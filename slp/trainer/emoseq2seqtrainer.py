import torch.nn as nn
import os
import torch
import torch.nn.functional as F
import math
import time
from tqdm import tqdm
from typing import cast, List, Optional, Tuple, TypeVar
from slp.util import from_checkpoint, to_device
from slp.util import types
import random

TrainerType = TypeVar('TrainerType', bound='Trainer')


class EmoSeq2SeqIterationsTrainerMultitask:
    def __init__(self, model,
                 optimizer, criterion1,criterion2, perplexity=True,
                 scheduler=None,
                 checkpoint_dir=None,  validate_every=400,
                 print_every=100,patience=3, clip=None, device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.perplexity = perplexity
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.validate_every = validate_every
        self.print_every = print_every
        self.clip = clip
        self.patience = patience
        self.device = device

    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1 = to_device(batch[0], device=self.device)
        lengths1 = to_device(batch[1], device=self.device)
        inputs2 = to_device(batch[2], device=self.device)
        lengths2 = to_device(batch[3], device=self.device)
        emo1 = to_device(batch[4], device=self.device)
        emo2 = to_device(batch[5], device=self.device)
        return inputs1, lengths1, inputs2, lengths2,emo1,emo2

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1, lengths1, inputs2, lengths2,emo1,emo2 = self.parse_batch(batch)
        y_pred,pred_emo1 = self.model(inputs1, lengths1, inputs2, lengths2)
        return y_pred,inputs2,pred_emo1,emo1

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        outputs, targets, pred_emo1, target_emo1 = \
            self.get_predictions_and_targets(batch)
        loss_lm = self.criterion1(outputs, targets)
        loss_emo1 = self.criterion2(pred_emo1, target_emo1)
        loss = 0.8*loss_lm + 0.2*loss_emo1
        # probs = F.softmax(pred_emo1)
        # pred_emo,x = torch.topk(probs,1,dim=-1)
        #
        # acc = sum([1 for pred,tar in zip(x.squeeze(1),target_emo1) if pred.item()==tar.item()])
        # acc = acc/x.shape[0]
        # print("Accuracy: ",acc)
        if self.perplexity:
            ppl = math.exp(loss.item())
        else:
            ppl=None
        # Perform backpropagation
        loss.backward()

        # Clip gradients: gradients are modified in place
        if self.clip is not None:
            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.clip)

        # Adjust model weights
        self.optimizer.step()

        return loss.item(), ppl

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            outputs, targets, pred_emo1, target_emo1 =  \
                self.get_predictions_and_targets(batch)
        loss_lm = self.criterion1(outputs, targets)
        loss_emo1 = self.criterion2(pred_emo1, target_emo1)
        loss = 0.8*loss_lm + 0.2*loss_emo1

        if self.perplexity:
            ppl = math.exp(loss.item())
        else:
            ppl=None

        return loss.item(), ppl

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
        val_ppl=0
        best_val_ppl, cur_patience = 10000, 0

        print("Training model....")
        for iteration in range(start_iter, n_iterations + 1):

            if cur_patience == self.patience:
                print("Breaking for 0 patience...")
                break

            # train step
            mini_batch = selected_batches_train[iteration - 1]
            loss, ppl = self.train_step(mini_batch)

            train_print_loss += loss
            train_print_ppl += ppl

            # eval step
            mini_batch = selected_batches_val[iteration-1]
            loss_val, ppl = self.eval_step(mini_batch)
            val_print_loss += loss_val
            val_print_ppl += ppl
            val_ppl += ppl

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

            # Checkpointing and early stopping
            if self.checkpoint_dir is not None:
                if iteration % self.validate_every == 0:
                    avg_val_ppl = val_ppl / self.validate_every
                    print("++++++++++++++++++++++++++")
                    print("Average val ppl: ",avg_val_ppl)
                    print("Best val ppl: ",best_val_ppl)
                    if avg_val_ppl < best_val_ppl:
                        self.save_iter(iteration, loss)
                        best_val_ppl=avg_val_ppl
                        cur_patience = 0
                    else:
                        cur_patience += 1
                    print("Patience is ",self.patience-cur_patience)
                    print("++++++++++++++++++++++++++")
                    val_ppl = 0

    def fit(self, train_loader, val_loader, n_iters):
        self.train_Iterations(n_iters, train_loader, val_loader)