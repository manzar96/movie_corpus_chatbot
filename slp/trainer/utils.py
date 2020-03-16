from slp.util import from_checkpoint, to_device

import torch
import os


def HREDtrain(batch, model, optim, scheduler, criterion, metrics, clip, device):

    optim.zero_grad()

    inputs1 = to_device(batch[0], device=device)
    lengths1 = to_device(batch[1], device=device)
    inputs2 = to_device(batch[2], device=device)
    lengths2 = to_device(batch[3], device=device)
    inputs3 = to_device(batch[4], device=device)
    lengths3 = to_device(batch[5], device=device)

    # Initialize variables
    loss = 0

    outputs = model(inputs1, lengths1, inputs2, lengths2, inputs3, lengths3)
    loss = criterion(outputs, inputs3)

    metrics_res = []
    if metrics is not []:
        for metric in metrics:
            metrics_res.append(metric(outputs, inputs3).item())


    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    if clip is not None:
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)


    # Adjust model weights
    optim.step()

    return loss.item(), metrics_res


def print_iter(print_loss, print_ppl, print_every, iteration, n_iterations):
    print_loss_avg = print_loss / print_every
    print_ppl_avg = print_ppl / print_every
    print(
        "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
            iteration, iteration / n_iterations * 100, print_loss_avg))
    print(
        "Iteration: {}; Percent complete: {:.1f}%; Average PPL: {:.4f}".format(
            iteration, iteration / n_iterations * 100, print_ppl_avg))
    print_loss = 0
    print_ppl = 0


def save_iter(directory, iteration, model, optimizer, loss):

    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
        'iteration': iteration,
        'model': model.state_dict(),
        'en_opt': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def train_Iterations(n_iterations, train_loader, val_loader, model, optimizer,
                     criterion, metrics, scheduler=None, checkpoint_dir=None,
                     save_every=1000, validate_every=10, print_every=200,
                     clip=None, device='cpu'):

    all_mini_batches = train_loader()

    start_iter = 1
    print_loss = 0
    print_ppl = 0

    for iteration in range(start_iter, n_iterations+1):

        mini_batch = all_mini_batches[iteration]

        loss, metrics_res = HREDtrain(mini_batch, model, optimizer,  scheduler,
                                  criterion, metrics, clip, device)

        print_loss += loss
        print_ppl += metrics_res[0]

        # Print progress
        if iteration % print_every == 0:
            print_iter(print_loss, print_ppl, print_every, iteration,
                       n_iterations)

        # Save checkpoint
        if checkpoint_dir is not None:
            if iteration % save_every == 0:
                save_iter(checkpoint_dir, iteration, model, optimizer, loss)
