import os
import time
from tqdm import tqdm
import math
from typing import Union
import torch
import torch.nn as nn
import random

from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, State
from ignite.metrics import RunningAverage, Loss

from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from typing import cast, List, Optional, Tuple, TypeVar
from slp.util import types
from slp.util.parallel import DataParallelModel, DataParallelCriterion

from slp.trainer.handlers import CheckpointHandler, EvaluationHandlerTxt
from slp.util import from_checkpoint, to_device
from slp.util import log
from slp.util import system


TrainerType = TypeVar('TrainerType', bound='Trainer')


class Trainer(object):
    def __init__(self: TrainerType,
                 model: nn.Module,
                 optimizer: Optimizer,
                 checkpoint_dir: str = '../../checkpoints',
                 experiment_name: str = 'experiment',
                 model_checkpoint: Optional[str] = None,
                 optimizer_checkpoint: Optional[str] = None,
                 metrics: types.GenericDict = None,
                 patience: int = 10,
                 validate_every: int = 1,
                 accumulation_steps: int = 1,
                 loss_fn: Union[_Loss, DataParallelCriterion] = None,
                 non_blocking: bool = True,
                 retain_graph: bool = False,
                 dtype: torch.dtype = torch.float,
                 device: str = 'cpu',
                 parallel: bool = False) -> None:
        self.dtype = dtype
        self.retain_graph = retain_graph
        self.non_blocking = non_blocking
        self.device = device
        self.loss_fn = loss_fn
        self.validate_every = validate_every
        self.patience = patience
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir

        model_checkpoint = self._check_checkpoint(model_checkpoint)
        optimizer_checkpoint = self._check_checkpoint(optimizer_checkpoint)

        self.model = cast(nn.Module, from_checkpoint(
                model_checkpoint, model, map_location=torch.device('cpu')))
        self.model = self.model.type(dtype).to(device)
        self.optimizer = from_checkpoint(optimizer_checkpoint, optimizer)
        self.parallel = parallel
        if parallel:
            if device == 'cpu':
                raise ValueError("parallel can be used only with cuda device")
            self.model = DataParallelModel(self.model).to(device)
            self.loss_fn = DataParallelCriterion(self.loss_fn)  # type: ignore
        if metrics is None:
            metrics = {}
        if 'loss' not in metrics:
            if self.parallel:
                metrics['loss'] = Loss(
                    lambda x, y: self.loss_fn(x, y).mean())  # type: ignore
            else:
                metrics['loss'] = Loss(self.loss_fn)
        self.trainer = Engine(self.train_step)
        self.train_evaluator = Engine(self.eval_step)
        self.valid_evaluator = Engine(self.eval_step)
        for name, metric in metrics.items():
            metric.attach(self.train_evaluator, name)
            metric.attach(self.valid_evaluator, name)

        self.pbar = ProgressBar()
        self.val_pbar = ProgressBar(desc='Validation')

        if checkpoint_dir is not None:
            self.checkpoint = CheckpointHandler(
                checkpoint_dir, experiment_name, score_name='validation_loss',
                score_function=self._score_fn, n_saved=2,
                require_empty=False, save_as_state_dict=True)

        self.early_stop = EarlyStopping(
            patience, self._score_fn, self.trainer)

        self.val_handler = EvaluationHandlerTxt(pbar=self.pbar,
                                             validate_every=1,
                                             early_stopping=self.early_stop,
                                            checkpointdir=self.checkpoint_dir)
        self.attach()
        log.info(
            f'Trainer configured to run {experiment_name}\n'
            f'\tpretrained model: {model_checkpoint} {optimizer_checkpoint}\n'
            f'\tcheckpoint directory: {checkpoint_dir}\n'
            f'\tpatience: {patience}\n'
            f'\taccumulation steps: {accumulation_steps}\n'
            f'\tnon blocking: {non_blocking}\n'
            f'\tretain graph: {retain_graph}\n'
            f'\tdevice: {device}\n'
            f'\tmodel dtype: {dtype}\n'
            f'\tparallel: {parallel}')

    def _check_checkpoint(self: TrainerType,
                          ckpt: Optional[str]) -> Optional[str]:
        if ckpt is None:
            return ckpt
        if system.is_url(ckpt):
            ckpt = system.download_url(cast(str, ckpt), self.checkpoint_dir)
        ckpt = os.path.join(self.checkpoint_dir, ckpt)
        return ckpt

    @staticmethod
    def _score_fn(engine: Engine) -> float:
        """Returns the scoring metric for checkpointing and early stopping

        Args:
            engine (ignite.engine.Engine): The engine that calculates
            the val loss

        Returns:
            (float): The validation loss
        """
        negloss: float = -engine.state.metrics['loss']
        return negloss

    def parse_batch(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        targets = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        return inputs, targets

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets = self.parse_batch(batch)
        y_pred = self.model(inputs)
        return y_pred, targets

    def train_step(self: TrainerType,
                   engine: Engine,
                   batch: List[torch.Tensor]) -> float:
        self.model.train()
        y_pred, targets = self.get_predictions_and_targets(batch)
        loss = self.loss_fn(y_pred, targets)  # type: ignore
        if self.parallel:
            loss = loss.mean()
        loss = loss / self.accumulation_steps
        loss.backward(retain_graph=self.retain_graph)
        if (self.trainer.state.iteration + 1) % self.accumulation_steps == 0:
            self.optimizer.step()  # type: ignore
            self.optimizer.zero_grad()
        loss_value: float = loss.item()
        return loss_value

    def eval_step(
            self: TrainerType,
            engine: Engine,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        self.model.eval()
        with torch.no_grad():
            y_pred, targets = self.get_predictions_and_targets(batch)
            return y_pred, targets

    def predict(self: TrainerType, dataloader: DataLoader) -> State:
        return self.valid_evaluator.run(dataloader)

    def fit(self: TrainerType,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 50) -> State:
        log.info(
            'Trainer will run for\n'
            f'model: {self.model}\n'
            f'optimizer: {self.optimizer}\n'
            f'loss: {self.loss_fn}')
        self.val_handler.attach(self.trainer,
                                self.train_evaluator,
                                train_loader,
                                validation=False)
        self.val_handler.attach(self.trainer,
                                self.valid_evaluator,
                                val_loader,
                                validation=True)
        self.model.zero_grad()
        self.trainer.run(train_loader, max_epochs=epochs)

    def overfit_single_batch(self: TrainerType,
                             train_loader: DataLoader) -> State:
        single_batch = [next(iter(train_loader))]

        if self.trainer.has_event_handler(self.val_handler, Events.EPOCH_COMPLETED):
            self.trainer.remove_event_handler(self.val_handler, Events.EPOCH_COMPLETED)

        self.val_handler.attach(self.trainer,
                                self.train_evaluator,
                                single_batch,  # type: ignore
                                validation=False)
        out = self.trainer.run(single_batch, max_epochs=100)
        return out

    def fit_debug(self: TrainerType,
                  train_loader: DataLoader,
                  val_loader: DataLoader) -> State:
        train_loader = iter(train_loader)
        train_subset = [next(train_loader), next(train_loader)]
        val_loader = iter(val_loader)  # type: ignore
        val_subset = [next(val_loader), next(val_loader)]  # type ignore
        out = self.fit(train_subset, val_subset, epochs=6)  # type: ignore
        return out

    def _attach_checkpoint(self: TrainerType) -> TrainerType:
        ckpt = {
            'model': self.model,
            'optimizer': self.optimizer
        }
        if self.checkpoint_dir is not None:
            self.valid_evaluator.add_event_handler(
                Events.COMPLETED, self.checkpoint, ckpt)
        return self

    def attach(self: TrainerType) -> TrainerType:
        ra = RunningAverage(output_transform=lambda x: x)
        ra.attach(self.trainer, "Train Loss")
        self.pbar.attach(self.trainer, ['Train Loss'])
        self.val_pbar.attach(self.train_evaluator)
        self.val_pbar.attach(self.valid_evaluator)
        self.valid_evaluator.add_event_handler(Events.COMPLETED,
                                               self.early_stop)
        self = self._attach_checkpoint()
        def graceful_exit(engine, e):
            if isinstance(e, KeyboardInterrupt):
                engine.terminate()
                log.warn("CTRL-C caught. Exiting gracefully...")
            else:
                raise(e)

        self.trainer.add_event_handler(Events.EXCEPTION_RAISED, graceful_exit)
        self.train_evaluator.add_event_handler(Events.EXCEPTION_RAISED,
                                               graceful_exit)
        self.valid_evaluator.add_event_handler(Events.EXCEPTION_RAISED,
                                               graceful_exit)
        return self


class AutoencoderTrainer(Trainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        return inputs, inputs


class SequentialTrainer(Trainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        targets = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        lengths = to_device(batch[2],
                            device=self.device,
                            non_blocking=self.non_blocking)
        return inputs, targets, lengths

    def get_predictions_and_targets(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets, lengths = self.parse_batch(batch)
        y_pred = self.model(inputs, lengths)
        return y_pred, targets


class Seq2seqTrainer(SequentialTrainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        lengths = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        return inputs, inputs, lengths


class TransformerTrainer(Trainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        targets = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        mask_inputs = to_device(batch[2],
                                device=self.device,
                                non_blocking=self.non_blocking)
        mask_targets = to_device(batch[3],
                                 device=self.device,
                                 non_blocking=self.non_blocking)
        return inputs, targets, mask_inputs, mask_targets

    def get_predictions_and_targets(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets, mask_inputs, mask_targets = self.parse_batch(batch)
        y_pred = self.model(inputs,
                            targets,
                            source_mask=mask_inputs,
                            target_mask=mask_targets)
        targets = targets.view(-1)
        y_pred = y_pred.view(targets.size(0), -1)
        # TODO: BEAMSEARCH!!
        return y_pred, targets


class HREDTrainer(Trainer):

    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1 = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        lengths1 = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        inputs2 = to_device(batch[2],
                           device=self.device,
                           non_blocking=self.non_blocking)
        lengths2 = to_device(batch[3],
                            device=self.device,
                            non_blocking=self.non_blocking)
        inputs3 = to_device(batch[4],
                           device=self.device,
                           non_blocking=self.non_blocking)
        lengths3 = to_device(batch[5],
                            device=self.device,
                            non_blocking=self.non_blocking)
        return inputs1, lengths1, inputs2, lengths2, inputs3, lengths3

    def get_predictions_and_targets(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1, lengths1, inputs2, lengths2, inputs3, lengths3 = \
            self.parse_batch(batch)
        y_pred = self.model(inputs1, lengths1, inputs2, lengths2, inputs3,
                            lengths3)
        # y_pred = self.model(batch)
        # TODO: BEAMSEARCH!!
        return y_pred, inputs3

    def train_step(self: TrainerType,
                   engine: Engine,
                   batch: List[torch.Tensor]) -> float:
        self.model.train()
        y_pred, targets = self.get_predictions_and_targets(batch)
        loss = self.loss_fn(y_pred, targets)  # type: ignore
        if self.parallel:
            loss = loss.mean()
        loss = loss / self.accumulation_steps
        loss.backward(retain_graph=self.retain_graph)
        if (self.trainer.state.iteration + 1) % self.accumulation_steps == 0:
            self.optimizer.step()  # type: ignore
            self.optimizer.zero_grad()
        loss_value: float = loss.item()
        return loss_value


class HREDIterationsTrainer:
    def __init__(self, model,
                 optimizer, criterion, metrics=None, scheduler=None,
                 checkpoint_dir=None, save_every=1000, validate_every=10,
                 print_every=200, clip=None, device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.validate_every = validate_every
        self.print_every = print_every
        self.clip = clip
        self.device = device

    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1 = to_device(batch[0], device=self.device)
        lengths1 = to_device(batch[1], device=self.device)
        inputs2 = to_device(batch[2], device=self.device)
        lengths2 = to_device(batch[3], device=self.device)
        inputs3 = to_device(batch[4], device=self.device)
        lengths3 = to_device(batch[5], device=self.device)
        return inputs1, lengths1, inputs2, lengths2, inputs3, lengths3

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1, lengths1, inputs2, lengths2, inputs3, lengths3 = \
            self.parse_batch(batch)
        y_pred = self.model(inputs1, lengths1, inputs2, lengths2, inputs3,
                            lengths3)
        return y_pred, inputs3

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        outputs, targets = self.get_predictions_and_targets(batch)
        loss = self.criterion(outputs, targets)

        metrics_res = []
        if self.metrics is not None:
            for metric in self.metrics:
                metrics_res.append(metric(outputs, targets).item())

        # Perform backpropagation
        loss.backward()

        # Clip gradients: gradients are modified in place
        if self.clip is not None:
            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.clip)

        # Adjust model weights
        self.optimizer.step()

        return loss.item(), metrics_res

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            outputs, targets = self.get_predictions_and_targets(batch)
        loss = self.criterion(outputs, targets)

        metrics_res = []
        if self.metrics is not None:
            for metric in self.metrics:
                metrics_res.append(metric(outputs, targets).item())

        return loss.item(), metrics_res

    def print_iter(self, print_loss, print_ppl, iteration, n_iterations):
        print_loss_avg = print_loss / self.print_every
        print_ppl_avg = print_ppl / self.print_every
        print("Training results")
        print(
            "Iteration: {}; Percent complete: {:.1f}%; Average train loss: {"
            ":.4f}".format(
                iteration, iteration / n_iterations * 100, print_loss_avg))
        print(
            "Iteration: {}; Percent complete: {:.1f}%; Average PPL: {:.4f}".format(
                iteration, iteration / n_iterations * 100, print_ppl_avg))
        print("++++++++++++++++++")

    def print_iter_val(self, print_loss, print_ppl, iteration, n_iterations):
        print_loss_avg = print_loss / self.print_every
        print_ppl_avg = print_ppl / self.print_every
        print("Validation results")
        print(
            "Iteration: {}; Percent complete: {:.1f}%; Average  loss: {"
            ":.4f}".format(
                iteration, iteration / n_iterations * 100, print_loss_avg))
        print(
            "Iteration: {}; Percent complete: {:.1f}%; Average PPL: {:.4f}".format(
                iteration, iteration / n_iterations * 100, print_ppl_avg))
        print("==============================================================")

    def save_iter(self, iteration, loss):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(iteration, 'checkpoint')))

    def train_Iterations(self, n_iterations, train_loader, val_loader):
        all_mini_batches_train = [batch for _, batch in enumerate(train_loader)]
        selected_batches_train = [random.choice(all_mini_batches_train) for _
                                  in range(n_iterations)]

        all_mini_batches_val = [batch for _, batch in enumerate(val_loader)]
        selected_batches_val = [random.choice(all_mini_batches_val) for _ in
                                range(n_iterations)]

        start_iter = 1
        train_print_loss = 0
        train_print_ppl = 0
        val_print_loss = 0
        val_print_ppl = 0

        print("Training model....")
        for iteration in range(start_iter, n_iterations + 1):

            # train step
            mini_batch = selected_batches_train[iteration - 1]
            loss, metrics_res = self.train_step(mini_batch)

            train_print_loss += loss
            train_print_ppl += metrics_res[0]

            # eval step
            mini_batch = selected_batches_val[iteration-1]
            loss_val, metrics_res = self.eval_step(mini_batch)
            val_print_loss += loss_val
            val_print_ppl += metrics_res[0]

            # Print progress
            if iteration % self.print_every == 0:
                self.print_iter(train_print_loss, train_print_ppl, iteration,
                                n_iterations)
                train_print_loss = 0
                train_print_ppl = 0

                self.print_iter_val(val_print_loss, val_print_ppl,
                                    iteration, n_iterations)
                val_print_loss = 0
                val_print_ppl = 0

            # Save checkpoint
            if self.checkpoint_dir is not None:
                if iteration % self.save_every == 0:
                    self.save_iter(iteration, loss)

    def fit(self, train_loader, val_loader, n_iters):
        self.train_Iterations(n_iters, train_loader, val_loader)


class HREDTrainerEpochs:

    def __init__(self, model,
                 optimizer, criterion, patience, metrics=None, scheduler=None,
                 checkpoint_dir=None,  clip=None, decreasing_tc=False,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.decreasing_tc = decreasing_tc

    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1 = to_device(batch[0], device=self.device)
        lengths1 = to_device(batch[1], device=self.device)
        inputs2 = to_device(batch[2], device=self.device)
        lengths2 = to_device(batch[3], device=self.device)
        inputs3 = to_device(batch[4], device=self.device)
        lengths3 = to_device(batch[5], device=self.device)
        return inputs1, lengths1, inputs2, lengths2, inputs3, lengths3

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1, lengths1, inputs2, lengths2, inputs3, lengths3 = \
            self.parse_batch(batch)
        y_pred = self.model(inputs1, lengths1, inputs2, lengths2, inputs3,
                            lengths3)
        return y_pred, inputs3

    def calc_val_loss(self, val_loader):
        self.model.eval()
        with torch.no_grad():

            # cur_tc = model.dec.get_teacher_forcing()
            # model.dec.set_teacher_forcing(True)

            val_loss, num_words = 0,0
            for index, batch in enumerate(tqdm(val_loader)):
                if self.decreasing_tc:
                    new_tc_ratio = 2100.0 / (2100.0 + math.exp(index/2100.0))
                    self.model.dec.set_tc_ratio(new_tc_ratio)

                preds, targets = self.get_predictions_and_targets(batch)


                # we want to find the perplexity or likelihood of the provided sequence


                preds = preds[:, :-1, :].contiguous().view(-1,preds.size(2))
                targets = targets[:, 1:].contiguous().view(-1)

                # do not include the lM loss, exp(loss) is perplexity
                loss = self.criterion(preds, targets)
                num_words += targets.ne(0).long().sum().item()
                val_loss += loss.item()

            # model.dec.set_teacher_forcing(cur_tc)

            return val_loss / num_words

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt, tc_ratio):

        print("Epoch {}:".format(epoch+1))
        print("Training loss: {} ".format(avg_train_epoch_loss))
        print("Training ppl: {} ".format(math.exp(avg_train_epoch_loss)))
        print("Validation loss: {} ".format(avg_val_epoch_loss))
        print("Validation ppl: {} ".format(math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("tc ratio", tc_ratio)
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(epoch,
                                                    'optimizer_checkpoint')))

    def clip_gnorm(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm()
                if param_norm > 1:
                    param.grad.data.mul_(1 / param_norm)

    def train_step(self, sample_batch):
        self.model.train()
        # new_tc_ratio = 2100.0 / (2100.0 + math.exp(batch_id / 2100.0))
        # self.model.dec.set_tc_ratio(new_tc_ratio)
        self.optimizer.zero_grad()

        preds, u3 = self.get_predictions_and_targets(sample_batch)

        # # neglect last timestep!
        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        # # neglect first timestep!!
        u3 = u3[:, 1:].contiguous().view(-1)
        loss = self.criterion(preds, u3)
        return loss, u3

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience, batch_id = 10000, 0, 0

        print("Training model....")
        self.model.train()
        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            train_epoch_loss, epoch_num_words = 0, 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):
                if self.decreasing_tc:
                    new_tc_ratio = 2100.0 / (2100.0 + math.exp(index/2100.0))
                    self.model.dec.set_tc_ratio(new_tc_ratio)

                loss, targets = self.train_step(sample_batch)
                # ne() because 0 is the pad idx
                target_toks = targets.ne(0).long().sum().item()

                epoch_num_words += target_toks
                train_epoch_loss += loss.item()
                loss = loss / target_toks

                # if options.lm:
                #     lmpreds = lmpreds[:, :-1, :].contiguous().view(-1,
                #                                                    lmpreds.size(
                #                                                        2))

                loss.backward(retain_graph=False)
                # if options.lm:
                #     lm_loss.backward()
                self.clip_gnorm()
                self.optimizer.step()

                batch_id += 1

            avg_val_loss = self.calc_val_loss(val_loader)
            avg_train_loss = train_epoch_loss / epoch_num_words
            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             cur_patience, strt, self.model.dec.get_tc_ratio())



    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class HREDTrainerEpochsTest:

    def __init__(self, model,
                 optimizer, criterion,patience, metrics=None, scheduler=None,
                 checkpoint_dir=None,  clip=None, device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience

    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1 = to_device(batch[0], device=self.device)
        lengths1 = to_device(batch[1], device=self.device)
        inputs2 = to_device(batch[2], device=self.device)
        lengths2 = to_device(batch[3], device=self.device)
        inputs3 = to_device(batch[4], device=self.device)
        lengths3 = to_device(batch[5], device=self.device)
        return inputs1, lengths1, inputs2, lengths2, inputs3, lengths3

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1, lengths1, inputs2, lengths2, inputs3, lengths3 = \
            self.parse_batch(batch)
        y_pred = self.model(batch)
        return y_pred, inputs3

    def calc_val_loss(self, val_loader):
        self.model.eval()
        with torch.no_grad():

            # cur_tc = model.dec.get_teacher_forcing()
            # model.dec.set_teacher_forcing(True)

            val_loss, num_words = 0,0
            for index, batch in enumerate(tqdm(val_loader)):

                preds,targets = self.get_predictions_and_targets(batch)


                # we want to find the perplexity or likelihood of the provided sequence


                preds = preds[:, :-1, :].contiguous().view(-1,preds.size(2))
                targets = targets[:, 1:].contiguous().view(-1)

                # do not include the lM loss, exp(loss) is perplexity
                loss = self.criterion(preds, targets)
                num_words += targets.ne(0).long().sum().item()
                val_loss += loss.item()

            # model.dec.set_teacher_forcing(cur_tc)

            return val_loss / num_words

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Training loss: {} ".format(avg_train_epoch_loss))
        print("Training ppl: {} ".format(math.exp(avg_train_epoch_loss)))
        print("Validation loss: {} ".format(avg_val_epoch_loss))
        print("Validation ppl: {} ".format(math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'checkpoint')))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'checkpoint')))

    def clip_gnorm(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm()
                if param_norm > 1:
                    param.grad.data.mul_(1 / param_norm)

    def train_step(self, sample_batch):
        self.model.train()
        # new_tc_ratio = 2100.0 / (2100.0 + math.exp(batch_id / 2100.0))
        # self.model.dec.set_tc_ratio(new_tc_ratio)
        self.optimizer.zero_grad()

        preds, u3 = self.get_predictions_and_targets(sample_batch)

        # # neglect last timestep!
        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        # # neglect first timestep!!
        u3 = u3[:, 1:].contiguous().view(-1)

        loss = self.criterion(preds, u3)
        return loss, u3

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience, batch_id = 10000, 0, 0

        print("Training model....")
        self.model.train()
        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            train_epoch_loss, epoch_num_words = 0, 0
            strt = time.time()

            for i_batch, sample_batch in enumerate(tqdm(train_loader)):

                loss, targets = self.train_step(sample_batch)
                # ne() because 0 is the pad idx
                target_toks = targets.ne(0).long().sum().item()

                epoch_num_words += target_toks
                train_epoch_loss += loss.item()
                loss = loss / target_toks

                # if options.lm:
                #     lmpreds = lmpreds[:, :-1, :].contiguous().view(-1,
                #                                                    lmpreds.size(
                #                                                        2))

                loss.backward(retain_graph=False)
                # if options.lm:
                #     lm_loss.backward()
                self.clip_gnorm()
                self.optimizer.step()

                batch_id += 1

            avg_val_loss = self.calc_val_loss(val_loader)
            avg_train_loss = train_epoch_loss / epoch_num_words
            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             cur_patience, strt)



    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)