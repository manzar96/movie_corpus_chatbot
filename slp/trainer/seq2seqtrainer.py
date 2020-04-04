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

def train(train_batches, model, model_optimizer, criterion, clip=None):
    """
    This function is used to train a Seq2Seq model.
    Model optimizer can be a list of optimizers if wanted(e.g. if we want to
    have different lr for encoder and decoder).
    """

    if not isinstance(model_optimizer, list):
        model_optimizer.zero_grad()
    else:
        for optimizer in model_optimizer:
            optimizer.zero_grad()
    epoch_loss = 0
    for index, batch in enumerate(train_batches):

        inputs, lengths_inputs, targets, masks_targets = batch
        inputs = inputs.long().cuda()
        targets = targets.long().cuda()
        lengths_inputs.cuda()
        masks_targets.cuda()

        if not isinstance(model_optimizer, list):
            model_optimizer.zero_grad()
        else:
            for optimizer in model_optimizer:
                optimizer.zero_grad()

        decoder_outputs = model(inputs, lengths_inputs, targets)

        # calculate and accumulate loss
        # loss = 0
        # n_totals = 0
        # for time in range(0, len(decoder_outputs)):
        #
        #     loss += criterion(decoder_outputs[time], targets[:, time].long())
        #     n_totals += 1
        # loss.backward()
        #
        # epoch_loss += loss.item() / n_totals

        loss = criterion(decoder_outputs,targets)
        epoch_loss += loss.item()
        loss.backward()
        # Clip gradients: gradients are modified in place
        if clip is not None:
            _ = nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Adjust model weights
        if not isinstance(model_optimizer, list):
            model_optimizer.step()
        else:
            for optimizer in model_optimizer:
                optimizer.step()

        last = index
    # we return average epoch loss
    return epoch_loss/(last+1)


def train_epochs(training_batches, model, model_optimizer,
                 criterion, num_epochs, print_every=1,
                 clip=None):

    print("Training...")
    model.train()

    # directory = os.path.join(save_dir, model_name)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    #
    # infofile = open(os.path.join(directory, 'model_info.txt'), 'w')
    # print(model_name, file=infofile)
    # print("Model architecture:  ", model,file=infofile)
    # print("Model optimizer:     ", model_optimizer,file=infofile)
    # print("Loss function:       ",criterion,file=infofile)
    # infofile.close()

    # logfile = open(os.path.join(directory, 'log_file.txt'), 'w')
    for epoch in range(num_epochs+1):
        avg_epoch_loss = 0

        # Train to all batches
        if clip is not None:
            avg_epoch_loss = train(training_batches, model, model_optimizer,
                                   criterion, clip)
        else:
            avg_epoch_loss = train(training_batches, model, model_optimizer,
                                   criterion)

        # Print progress
        if epoch % print_every == 0:
            print("Epoch {}; Percent complete: {:.1f}%; Average Epoch loss: {"
                  ":.4f}".format(epoch, epoch / num_epochs * 100,
                                 avg_epoch_loss))
                  #file=logfile)

    #     if save_every is not None:
    #         # Save checkpoint
    #         if epoch % save_every == 0:
    #
    #             if isinstance(model_optimizer, list):
    #                 torch.save({
    #                     'epoch': epoch,
    #                     'model': model.state_dict(),
    #                     'model_opt_enc': model_optimizer[0].state_dict(),
    #                     'model_opt_dec': model_optimizer[1].state_dict(),
    #                     'loss': avg_epoch_loss,
    #
    #                 }, os.path.join(directory, '{}_{}.tar'.format(epoch,
    #                                                               'checkpoint')))
    #             else:
    #                 torch.save({
    #                     'epoch': epoch,
    #                     'model': model.state_dict(),
    #                     'model_opt': model_optimizer.state_dict(),
    #                     'loss': avg_epoch_loss,
    #
    #                 }, os.path.join(directory, '{}_{}.tar'.format(epoch,
    #                                                               'checkpoint')))
    # logfile.close()


def validate( val_batches, model):
    """
    This function is used for validating the model!
    Model does not use "forward" but "evaluate , because we don't want to use
    teacher forcing!
    :param val_batches: batches given for validation
    :param model: trained model that need to have evaluate function (like
    forward)
    :return:
    """

    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(val_batches):
            inputs, lengths_inputs, targets, masks_targets = batch
            inputs = inputs.long().cuda()
            targets = targets.long().cuda()
            lengths_inputs.cuda()
            masks_targets.cuda()

            decoder_outputs, decoder_hidden = model.evaluate(inputs,
                                                             lengths_inputs)
            # decoder_outputs is a 3d tensor(batchsize,seq length,outputsize)

            for batch_index in range(decoder_outputs.shape[0]):
                out_logits = F.log_softmax(decoder_outputs[batch_index],dim=1)
                _,out_indexes = torch.max(out_logits,dim=1)

                print("Question: ", inputs[batch_index])
                print("Target: ",targets[batch_index])
                print("Response: ",out_indexes)

                print("+++++++++++++++++++++")


def inputInteraction(model, vocloader, text_preprocessor, text_tokenizer,
                     idx_loader, padder):
    max_len = model.max_target_len
    input_sentence = ""
    while True:
        try:
            # Get input response:
            input_sentence = input('>')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Process sentence
            input_filt = text_preprocessor.process_text([input_sentence])
            input_tokens = text_tokenizer.word_tokenization(input_filt)

            input_indexes = idx_loader.get_indexes(input_tokens)

            input_length = [len(input_indexes[0])]
            padded_input = padder.zeroPadding(input_indexes,max_len)
            padded_input = torch.LongTensor(padded_input).cuda()

            input_length = torch.LongTensor(input_length).cuda()

            dec_outs,dec_hidden = model.evaluate(padded_input,input_length)

            out_logits = F.log_softmax(dec_outs[0], dim=1)
            _, out_indexes = torch.max(out_logits, dim=1)
            #print(out_indexes)
            decoded_words = [vocloader.idx2word[int(index)] for index in
                             out_indexes]
            print("Response: ", decoded_words)

            print("+++++++++++++++++++++")

        except KeyError:
            print("Error:Encountered unknown word")


class Seq2SeqTrainerEpochs:

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
        return inputs1, lengths1, inputs2, lengths2

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1, lengths1, inputs2, lengths2 = self.parse_batch(batch)
        y_pred = self.model(inputs1, lengths1, inputs2, lengths2)
        return y_pred, inputs2

    def calc_val_loss(self, val_loader):
        curr_tc = self.model.decoder.get_tc_ratio()
        self.model.decoder.set_tc_ratio(1.0)
        self.model.eval()
        with torch.no_grad():

            # cur_tc = model.dec.get_teacher_forcing()
            # model.dec.set_teacher_forcing(True)

            val_loss, num_words = 0,0
            for index, batch in enumerate(tqdm(val_loader)):
                preds, targets = self.get_predictions_and_targets(batch)
                preds = preds[:, :-1, :].contiguous().view(-1,preds.size(2))
                targets = targets[:, 1:].contiguous().view(-1)

                # do not include the lM loss, exp(loss) is perplexity
                loss = self.criterion(preds, targets)
                num_words += targets.ne(0).long().sum().item()
                val_loss += loss.item()

            self.model.decoder.set_tc_ratio(curr_tc)
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
        self.optimizer.zero_grad()
        preds, target = self.get_predictions_and_targets(sample_batch)
        # # neglect last timestep!
        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        # # neglect first timestep!!
        target = target[:, 1:].contiguous().view(-1)
        loss = self.criterion(preds, target)
        return loss, target

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience, batch_id = 10000, 0, 0

        print("Training model....")
        self.model.train()

        if self.decreasing_tc:
            new_tc_ratio = 2100.0 / (2100.0 + math.exp(batch_id / 2100.0))
            self.model.decoder.set_tc_ratio(new_tc_ratio)

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            train_epoch_loss, epoch_num_words = 0, 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):
                if self.decreasing_tc:
                    new_tc_ratio = 2100.0 / (2100.0 + math.exp(batch_id / 2100.0))
                    self.model.decoder.set_tc_ratio(new_tc_ratio)

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
                             cur_patience, strt,
                             self.model.decoder.get_tc_ratio())

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class Seq2SeqIterationsTrainer:
    def __init__(self, model,
                 optimizer, criterion, metrics=None, scheduler=None,
                 checkpoint_dir=None,  validate_every=400,
                 print_every=100,patience=3, clip=None, device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
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

        return inputs1, lengths1, inputs2, lengths2

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1, lengths1, inputs2, lengths2 = self.parse_batch(batch)
        y_pred = self.model(inputs1, lengths1, inputs2, lengths2)
        return y_pred,inputs2

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
            loss, metrics_res = self.train_step(mini_batch)

            train_print_loss += loss
            train_print_ppl += metrics_res[0]


            # eval step
            mini_batch = selected_batches_val[iteration-1]
            loss_val, metrics_res = self.eval_step(mini_batch)
            val_print_loss += loss_val
            val_print_ppl += metrics_res[0]
            val_ppl += metrics_res[0]


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