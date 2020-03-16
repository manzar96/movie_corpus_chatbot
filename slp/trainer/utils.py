from slp.util import from_checkpoint, to_device

import torch
import os


class HREDIterationsTrainer:
    def __init__(self, model, 
                 optimizer, criterion, metrics=None, scheduler=None,
                 checkpoint_dir=None, save_every=1000, validate_every=10, 
                 print_every=200, clip=None, device='cpu'):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        if metrics is None:
            self.metrics = []
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.validate_every = validate_every
        self.print_every = print_every
        self.clip = clip
        self.device = device

    def train_step(self, batch):
    
        self.optimizer.zero_grad()
    
        inputs1 = to_device(batch[0], device=self.device)
        lengths1 = to_device(batch[1], device=self.device)
        inputs2 = to_device(batch[2], device=self.device)
        lengths2 = to_device(batch[3], device=self.device)
        inputs3 = to_device(batch[4], device=self.device)
        lengths3 = to_device(batch[5], device=self.device)
    
        # Initialize variables
        outputs = self.model(inputs1, lengths1, inputs2, lengths2, inputs3,
                             lengths3)
        loss = self.criterion(outputs, inputs3)
    
        metrics_res = []
        if self.metrics is not []:
            for metric in self.metrics:
                metrics_res.append(metric(outputs, inputs3).item())

        # Perform backpropatation
        loss.backward()
    
        # Clip gradients: gradients are modified in place
        if self.clip is not None:
            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.clip)
    
    
        # Adjust model weights
        self.optimizer.step()
    
        return loss.item(), metrics_res

    def print_iter(self, print_loss, print_ppl, iteration, n_iterations):
        print_loss_avg = print_loss / self.print_every
        print_ppl_avg = print_ppl / self.print_every
        print(
            "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iteration, iteration / n_iterations * 100, print_loss_avg))
        print(
            "Iteration: {}; Percent complete: {:.1f}%; Average PPL: {:.4f}".format(
                iteration, iteration / n_iterations * 100, print_ppl_avg))
        print_loss = 0
        print_ppl = 0
    
    def save_iter(self, iteration, loss):
    
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save({
            'iteration': iteration,
            'model': self.model.state_dict(),
            'en_opt': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(self.checkpoint_dir, '{}_{}.tar'.format(iteration,
                                                                'checkpoint')))

    def train_Iterations(self, n_iterations, train_loader):
    
        all_mini_batches = train_loader()
    
        start_iter = 1
        print_loss = 0
        print_ppl = 0
    
        for iteration in range(start_iter, n_iterations+1):
    
            mini_batch = all_mini_batches[iteration]
    
            loss, metrics_res = self.train_step(mini_batch)
    
            print_loss += loss
            print_ppl += metrics_res[0]
    
            # Print progress
            if iteration % self.print_every == 0:
                self.print_iter(print_loss, print_ppl, iteration, n_iterations)
                import ipdb;ipdb.set_trace()

            # Save checkpoint
            if self.checkpoint_dir is not None:
                if iteration % self.save_every == 0:
                    self.save_iter(iteration,  loss)

    def fit(self, train_loader, val_loader, n_iters):
        self.train_Iterations(n_iters, train_loader)
