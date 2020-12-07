import torch.nn as nn
import os
import torch
import torch.nn.functional as F
import math
import time
import random
import warnings

from tqdm import tqdm
from typing import cast, List, Optional, Tuple, TypeVar
from slp.util import from_checkpoint, to_device
from slp.util import types
from slp.data.vocab import tensor2text

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

    def __init__(self, model, optimizer, criterion, scheduler=None, patience=5,
                 checkpoint_dir=None,  clip=None, metrics=None,
                 vocab=None, genoptions=None,
                 best_metric='ppl',
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.patience = patience
        self.metrics = metrics
        self.genoptions = genoptions
        self.metrics_recoder = {}
        self.vocab = vocab
        if self.metrics is None:
            self.vocab = None
            warnings.warn("Model Generate will not be called!",RuntimeWarning)
        if self.genoptions and self.genoptions.skip_generation:
            warnings.warn("Model Generate will not be called!", RuntimeWarning)

        self.best_metric = best_metric
        if self.best_metric == 'ppl':
            warnings.warn("If you use ppl as bestmetric set "
                          "-skip_generation to True in options for better "
                          "speed", RuntimeWarning)
            self.op = min
        elif self.best_metric == 'bleu':
            self.op = max
        else:
            raise NotImplementedError
        self.device = device

    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs1 = to_device(batch[0], device=self.device)
        lengths1 = to_device(batch[1], device=self.device)
        inputs2 = to_device(batch[2], device=self.device)
        lengths2 = to_device(batch[3], device=self.device)
        return inputs1, lengths1, inputs2, lengths2

    def init_metrics(self):
        self.metrics_recorder_epoch = {'bleu1':0,'bleu2':0,'bleu3':0,
                                      'bleu4':0, 'distinct1':0,'distinct2':0,
                                      'distinct3':0}

    def update_metric_recorder(self, epoch, numberofbatches):
        self.metrics_recorder_epoch['bleu1'] /= numberofbatches
        self.metrics_recorder_epoch['bleu2'] /= numberofbatches
        self.metrics_recorder_epoch['bleu3'] /= numberofbatches
        self.metrics_recorder_epoch['bleu4'] /= numberofbatches
        self.metrics_recorder_epoch['distinct1'] /= numberofbatches
        self.metrics_recorder_epoch['distinct2'] /= numberofbatches
        self.metrics_recorder_epoch['distinct3'] /= numberofbatches
        self.metrics_recoder[epoch] = self.metrics_recorder_epoch

    def update_metrics(self, hyp_texts, ref_texts):
        bleu1 = 0
        bleu2 = 0
        bleu3 = 0
        bleu4 = 0
        distinct1 = 0
        distinct2 = 0
        distinct3 = 0
        for text, target in zip(hyp_texts, ref_texts):
            if 'bleu' in self.metrics.keys():
                bleu1 += self.metrics['bleu'].compute([text], target, k=1)
                bleu2 += self.metrics['bleu'].compute([text], target, k=2)
                bleu3 += self.metrics['bleu'].compute([text], target, k=3)
                bleu4 += self.metrics['bleu'].compute([text], target, k=4)
            if 'distinct' in self.metrics.keys():
                distinct1 += self.metrics['distinct'].distinct_n_sentence_level(
                    text, 1)
                distinct2 += self.metrics['distinct'].distinct_n_sentence_level(
                    text, 2)
                distinct3 += self.metrics['distinct'].distinct_n_sentence_level(
                    text, 3)

        bleu1 /= len(ref_texts)
        bleu2 /= len(ref_texts)
        bleu3 /= len(ref_texts)
        bleu4 /= len(ref_texts)
        distinct1 /= len(ref_texts)
        distinct2 /= len(ref_texts)
        distinct3 /= len(ref_texts)
        self.metrics_recorder_epoch['bleu1'] += bleu1
        self.metrics_recorder_epoch['bleu2'] += bleu2
        self.metrics_recorder_epoch['bleu3'] += bleu3
        self.metrics_recorder_epoch['bleu4'] += bleu4
        self.metrics_recorder_epoch['distinct1'] += distinct1
        self.metrics_recorder_epoch['distinct2'] += distinct2
        self.metrics_recorder_epoch['distinct3'] += distinct3

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    bleu4,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {} | Val BLEU4: {}".format(
            avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss), bleu4))
        print("Patience left: {}".format(self.patience-cur_patience))
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

    def train_step(self, batch):
        self.optimizer.zero_grad()
        inputs1, lengths1, targets, targets_lengths = self.parse_batch(batch)
        logits = self.model(inputs1, lengths1, targets, targets_lengths)
        loss = self.criterion(logits, targets)
        return loss

    def train_epoch(self, trainloader):
        self.model.train()
        train_epoch_loss = 0
        for index, sample_batch in enumerate(tqdm(trainloader)):
            loss = self.train_step(sample_batch)
            train_epoch_loss += loss.item()
            loss.backward(retain_graph=False)
            if self.clip is not None:
                _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return train_epoch_loss / len(trainloader)

    def eval_step(self, batch):

        # evaluating with full teacher forcing
        inputs1, lengths1, targets, targets_lengths = self.parse_batch(batch)
        logits = self.model(inputs1, lengths1, targets, targets_lengths)
        loss = self.criterion(logits, targets)

        if self.metrics is not None and not self.genoptions.skip_generation:
            # we run generate to report real result for our model (NO teacher
            # forcing)
            beam_preds_scores, _ = self.model.generate(inputs1,
                                                       lengths1,
                                                       self.genoptions,
                                                       self.vocab.start_idx,
                                                       self.vocab.end_idx,
                                                       self.vocab.pad_idx)
            preds, scores = zip(*beam_preds_scores)
            hyp_texts = [tensor2text(pred, self.vocab) for pred in preds]
            ref_texts = [tensor2text(target, self.vocab) for target in targets]
            self.update_metrics(hyp_texts, ref_texts)

        return loss

    def val_epoch(self, valloader):
        self.model.eval()
        self.init_metrics()
        with torch.no_grad():
            val_epoch_loss = 0
            for index, sample_batch in enumerate(tqdm(valloader)):
                loss = self.eval_step(sample_batch)
                val_epoch_loss += loss.item()
            return val_epoch_loss / len(valloader)

    def train(self, n_epochs, train_loader, val_loader):

        print("Training model....")
        self.model.train()

        cur_patience = 0
        if self.op is min:
            self.best_metric = 1e20
        else:
            self.best_metric = -1e20

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break
            strt = time.time()
            print(self.optimizer.load_state_dict)
            avg_epoch_train_loss = self.train_epoch(train_loader)
            avg_epoch_val_loss = self.val_epoch(val_loader)
            self.update_metric_recorder(epoch, len(val_loader))

            if self.best_metric is 'ppl':
                if avg_epoch_val_loss is self.op(avg_epoch_val_loss,
                                                 self.best_metric):
                    self.save_epoch(epoch)
                    self.best_metric = avg_epoch_val_loss
                    cur_patience = 0
                else:
                    cur_patience += 1
            else:
                if self.metrics_recoder[epoch]['bleu4'] is self.op(
                        self.metrics_recoder[epoch]['bleu4'],
                        self.best_metric):
                    self.save_epoch(epoch)
                    self.best_metric = self.metrics_recoder[epoch]['bleu4']
                    cur_patience = 0
                else:
                    cur_patience += 1

            self.print_epoch(epoch, avg_epoch_train_loss, avg_epoch_val_loss,
                             self.metrics_recoder[epoch]['bleu4'],
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train(epochs, train_loader, val_loader)


class Seq2SeqIterationsTrainer:
    def __init__(self, model,
                 optimizer, criterion, perplexity=True, scheduler=None,
                 checkpoint_dir=None,  validate_every=400,
                 print_every=100,patience=3, clip=None, device='cpu',
                 bleu=None):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.perplexity = perplexity
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.validate_every = validate_every
        self.print_every = print_every
        self.clip = clip
        self.patience = patience
        self.device = device
        self.bleu = bleu

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
        bleu = self.bleu(outputs,targets)
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

        return loss.item(), ppl,bleu

    def eval_step(self, batch):
        # set tc to zero
        tc_ratio = self.model.decoder.get_tc_ratio()
        self.model.decoder.set_tc_ratio(0)
        self.model.eval()
        with torch.no_grad():
            outputs, targets = self.get_predictions_and_targets(batch)
        loss = self.criterion(outputs, targets)
        bleu = self.bleu(outputs,targets)

        if self.perplexity:
            ppl = math.exp(loss.item())
        else:
            ppl=None

        # reset tc ratio
        self.model.decoder.set_tc_ratio(tc_ratio)
        return loss.item(), ppl,bleu

    def print_iter(self, print_loss, print_ppl, iteration, n_iterations):
        print_loss_avg = print_loss / self.print_every
        print_ppl_avg = print_ppl / self.print_every
        print("Iteration: {} | Percent complete: {:.1f}%".format(iteration,
                                                                 iteration /
                                                                 n_iterations
                                                                 * 100))
        print("Average train loss: {:.4f} | Average train ppl: {:.3f}".format(
                 print_loss_avg, print_ppl_avg))

    def print_iter_val(self, print_loss, print_ppl):
        print_loss_avg = print_loss / self.print_every
        print_ppl_avg = print_ppl / self.print_every
        print(
            "Average val loss: {:.4f} | Average val ppl: {:.3f}".format(
                print_loss_avg,
                print_ppl_avg))

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
        train_print_bleu = 0
        val_print_bleu = 0
        best_val_bleu = -1
        val_bleu = 0


        print("Training model....\n")
        for iteration in range(start_iter, n_iterations + 1):

            if cur_patience == self.patience:
                print("Breaking for 0 patience...")
                break

            # train step
            mini_batch = selected_batches_train[iteration - 1]
            loss, ppl, bleu = self.train_step(mini_batch)

            train_print_loss += loss
            train_print_ppl += ppl
            train_print_bleu += bleu

            # eval step
            mini_batch = selected_batches_val[iteration-1]
            loss_val, ppl, blue = self.eval_step(mini_batch)
            val_print_loss += loss_val
            val_print_ppl += ppl
            val_ppl += ppl
            val_print_bleu += bleu
            val_bleu += bleu

            # Print progress
            if iteration % self.print_every == 0:
                self.print_iter(train_print_loss, train_print_ppl, iteration,
                                n_iterations)
                print("Train BLEU: ", train_print_bleu / self.print_every)
                train_print_loss = 0
                train_print_ppl = 0

                self.print_iter_val(val_print_loss, val_print_ppl)
                val_print_loss = 0
                val_print_ppl = 0
                print("Val BLEU: ", val_print_bleu / self.print_every)

            if self.checkpoint_dir is not None:
                if iteration % self.validate_every == 0:
                    avg_val_bleu = val_bleu / self.validate_every
                    print("++++++++++++++++++++++++++")
                    print("Average val ppl: ", avg_val_bleu)
                    print("Best val ppl: ", best_val_bleu)
                    if avg_val_bleu > best_val_bleu:
                        self.save_iter(iteration, loss)
                        best_val_bleu = avg_val_bleu
                        cur_patience = 0
                    else:
                        cur_patience += 1
                    print("Patience is ", self.patience - cur_patience)
                    print("++++++++++++++++++++++++++")
                    val_bleu = 0

            # Checkpointing and early stopping
            # if self.checkpoint_dir is not None:
            #     if iteration % self.validate_every == 0:
            #         avg_val_ppl = val_ppl / self.validate_every
            #         print("++++++++++++++++++++++++++")
            #         print("Average val ppl: ",avg_val_ppl)
            #         print("Best val ppl: ",best_val_ppl)
            #         if avg_val_ppl < best_val_ppl:
            #             self.save_iter(iteration, loss)
            #             best_val_ppl=avg_val_ppl
            #             cur_patience = 0
            #         else:
            #             cur_patience += 1
            #         print("Patience is ",self.patience-cur_patience)
            #         print("++++++++++++++++++++++++++")
            #         val_ppl = 0

    def fit(self, train_loader, val_loader, n_iters):
        self.train_Iterations(n_iters, train_loader, val_loader)




class Seq2SeqTrainerEpochsAuxilary:

    def __init__(self, model,
                 optimizer, criterion1,criterion2, patience,
                 scheduler=None,
                 checkpoint_dir=None,  clip=None, decreasing_tc=False,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion1 = criterion1
        self.criterion2 = criterion2
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
        """Validation setting tc to 1(?)"""
        curr_tc = self.model.decoder.get_tc_ratio()
        self.model.decoder.set_tc_ratio(0)
        self.model.eval()
        with torch.no_grad():
            val_loss, num_word = 0, 0
            for index, batch in enumerate(tqdm(val_loader)):
                preds, targets = self.get_predictions_and_targets(batch)
                # preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
                # targets = targets[:, 1:].contiguous().view(-1)
                loss1 = self.criterion1(preds, targets)
                loss2 = self.criterion2(preds, targets)
                loss = 0.8*loss1+0.2*(1-loss2)
                val_loss += loss.item()
            self.model.decoder.set_tc_ratio(curr_tc)
            return val_loss / len(val_loader)

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt, tc_ratio):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("tc ratio: ", tc_ratio)
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

    def train_step(self, sample_batch):
        self.model.train()
        self.optimizer.zero_grad()
        preds, target = self.get_predictions_and_targets(sample_batch)
        loss1 = self.criterion1(preds, target)
        loss2 = self.criterion2(preds, target)
        loss = 0.8 * loss1 + 0.2 * (1-loss2)
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

            train_epoch_loss, epoch_num_words= 0, 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):
                if self.decreasing_tc:
                    new_tc_ratio = 2100.0 / (2100.0 + math.exp(batch_id / 2100.0))
                    self.model.decoder.set_tc_ratio(new_tc_ratio)

                loss, targets = self.train_step(sample_batch)
                train_epoch_loss += loss.item()
                # if index%500 == 0:
                #     print("running loss:  ",loss.item())

                loss.backward(retain_graph=False)
                if self.clip is not None:
                    _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.clip)
                self.optimizer.step()

                batch_id += 1
            avg_val_loss = self.calc_val_loss(val_loader)
            avg_train_loss = train_epoch_loss / len(train_loader)

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