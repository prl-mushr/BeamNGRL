import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import network_utils as nu
from utils.exp_utils import get_dataloaders, build_nets, get_loss_func, init_exp_dir
import argparse
import yaml
from BeamNGRL import *
from typing import Dict


def train(
        network,
        optimizer,
        loss_func,
        train_loader,
        valid_loader,
        exp_path,
        args,
        tn_args: Dict = None,
):

    writer = SummaryWriter(log_dir=os.path.join(exp_path))

    net_sched = None
    if args.scheduler:
        net_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                args.epochs,
                last_epoch=args.start_from,
        )
    torch.autograd.set_detect_anomaly(True)
    best_loss = np.inf
    try:
        if args.start_from != -1:
            start = args.start_from
        else:
            start = 1

        for epoch in range(start, args.n_epochs + 1):
            # if epoch > 1:
            if epoch > 0:
                network.train()
                train_average_loss = []

                for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(train_loader)):

                    # Set device
                    states_tn = states_tn.to(**tn_args)
                    controls_tn = controls_tn.to(**tn_args)
                    ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}
                    optimizer.zero_grad()

                    pred = network(
                        states_tn,
                        controls_tn,
                        ctx_tn_dict,
                    )

                    targets = network.process_targets(states_tn)
                    loss = loss_func(pred, targets)

                    # loss = loss_func(pred, states_tn)

                    loss.backward()
                    optimizer.step()

                    # Logging
                    train_average_loss.append(loss.detach().cpu().numpy())
                    grad_mag = nu.get_gradient_magnitude(network)
                    writer.add_scalar('Train/batchLoss', loss,
                        len(train_loader) * epoch + i)
                    writer.add_scalar('Train/gradient', grad_mag,
                        len(train_loader) * epoch + i)

                    # TODO: Visualization

                train_average_loss = np.asarray(train_average_loss).mean()
                writer.add_scalar('Train/Loss', train_average_loss, epoch)

            if args.save_each_epoch:
                nu._save_model(
                    network,
                    optimizer,
                    epoch,
                    None,
                    os.path.join(exp_path, "epoch_%d.pth" % epoch),
                )

            if not args.skip_valid_eval and epoch % args.valid_interval == 0:
                network.eval()
                test_avg_loss = []

                for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(valid_loader)):

                    with torch.no_grad():
                        # Set device
                        states_tn = states_tn.to(**tn_args)
                        controls_tn = controls_tn.to(**tn_args)
                        ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}

                        pred = network(
                            states_tn,
                            controls_tn,
                            ctx_tn_dict,
                        )

                        targets = network.process_targets(states_tn)
                        test_batch_loss = loss_func(pred, targets)

                    test_avg_loss.append(test_batch_loss.cpu().numpy())
                    writer.add_scalar('Valid/batchLoss', test_batch_loss,
                        len(valid_loader) * epoch + i)

                    # TODO: Visualization

                test_loss = np.asarray(test_avg_loss).mean()
                writer.add_scalar('Valid/Loss', test_loss, epoch)

                if test_loss < best_loss:
                    print('New best model found, saving...')
                    nu._save_model(
                        network,
                        optimizer,
                        epoch,
                        None,
                        os.path.join(exp_path, "best_%d.pth" % epoch),
                    )
                    best_loss = test_loss

            if net_sched is not None:
                net_sched.step()

    finally:
        writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file for training model')
    parser.add_argument('--output', type=str, required=False, help='location to store output - weights, log')
    parser.add_argument('--cpu', type=str, required=False, default=False, help='use cpu for training')
    parser.add_argument('--n_epochs', type=int, required=False, default=100, help='Number of training epochs.')
    parser.add_argument('--scheduler', action='store_true', help='use scheduler (cosine annealing)')
    parser.add_argument('--log_interval', type=int, required=False, default=10, help='model grad/weights log interval')
    parser.add_argument('--finetune', type=str, required=False, default = None, help='pretrained weights to finetune from')
    parser.add_argument('--shuffle', type=bool, required=False, default=True, help='shuffle data')
    parser.add_argument('--batchsize', type=int, required=False, default=16, help='training batch size')
    parser.add_argument('--start_from', type=int, required=False, default=-1, help='epoch to start from')
    parser.add_argument('--valid_interval', type=int, required=False, default=1, help='model grad/weights log interval')
    parser.add_argument('--skip_valid_eval', action='store_true', help='skip computation of validation error.')
    parser.add_argument('--save_each_epoch', type=bool, required=False, default=False, help='Save model after each epoch.')
    args = parser.parse_args()

    # Set torch params
    torch.manual_seed(0)
    torch.set_num_threads(1)
    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cpu:
        tensor_args = {'device': torch.device('cuda'), 'dtype': torch.float32}
    else:
        tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}

    # Load experiment config
    config = yaml.load(open(DYN_EXP_CONFIG / args.config).read(), Loader=yaml.SafeLoader)

    # Dataloaders
    train_loader, valid_loader, stats, data_cfg = get_dataloaders(args, config)

    # Model init.
    net, net_opt = build_nets(
        config, tensor_args,
        model_weight_file=args.finetune,
        data_stats=stats,
    )

    loss_func = get_loss_func(config)

    # Experiment init.
    exp_path = init_exp_dir(config, args)

    train(
        net, net_opt,
        loss_func,
        train_loader,
        valid_loader,
        exp_path,
        args,
        tensor_args,
    )