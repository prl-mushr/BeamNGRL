import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import network_utils as nu
import argparse
import yaml
from pathlib import Path


CONFIG_ROOT = Path('config')


def train(
        network,
        optimizer,
        train_loader,
        valid_loader,
        exp_path,
        args,
        loss_func=None
):

    writer = SummaryWriter(log_dir=os.path.join(exp_path))

    net_sched = None
    if args.scheduler:
        net_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                args.epochs,
                last_epoch=args.start_from,
        )

    best_loss = np.inf
    try:
        if args.start_from != -1:
            start = args.start_from
        else:
            start = 0

        for epoch in range(start, args.n_epochs + 1):
            if epoch > 1:
                network.train()
                train_average_loss = []

                for i, traj_data in enumerate(tqdm(train_loader)):

                    optimizer.zero_grad()

                    pred = network(traj_data)

                    loss = loss_func(pred, traj_data)

                    loss.backward()

                    # Batch logging
                    grad_mag = nu.get_gradient_magnitude(network)
                    writer.add_scalar(
                        'Train/gradient',
                        grad_mag,
                        len(train_loader) * epoch + i
                    )

                    train_average_loss.append(loss.detach().cpu().numpy())

                    writer.add_scalar(
                        'Train/batchLoss',
                        loss,
                        len(train_loader) * epoch + i)

                    # TODO: Visualization

                    optimizer.step()

                # Logging
                train_average_loss = np.asarray(train_average_loss).mean()
                writer.add_scalar(
                    'Train/Loss',
                    train_average_loss,
                    epoch)


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

                for i, traj_data in enumerate(tqdm(valid_loader)):

                    with torch.no_grad():
                        pred = network(traj_data)

                        loss = loss_func(pred, traj_data)

                    test_avg_loss.append(loss.cpu().numpy())
                    writer.add_scalar(
                        'Valid/batchLoss',
                        loss,
                        len(valid_loader) * epoch + i)

                    # TODO: Visualization

                test_loss = np.asarray(test_avg_loss).mean()

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

                writer.add_scalar(
                    'Valid/Loss',
                    test_loss,
                    epoch)

                if net_sched is not None:
                    net_sched.step()
    finally:
        writer.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, help='config file for training model')
    parser.add_argument('--output', type=str, required=False, help='location to store output - weights, log')
    parser.add_argument('--cpu', type=str, required=False, default=False, help='use cpu for training')
    parser.add_argument('--n_epochs', type=int, required=False, default=100, help='Number of training epochs.')
    parser.add_argument('--scheduler', action='store_true', help='use scheduler (cosine annealing)')
    parser.add_argument('--log_interval', type=int, required=False, default=10, help='model grad/weights log interval')
    parser.add_argument('--shuffle', type=bool, required=False, default=True, help='shuffle data')
    parser.add_argument('--batchsize', type=int, required=False, default=16, help='training batch size')
    parser.add_argument('--start_from', type=int, required=False, default=-1, help='epoch to start from')
    parser.add_argument('--valid_interval', type=int, required=False, default=1, help='model grad/weights log interval')
    args = parser.parse_args()

    # Set torch params
    torch.manual_seed(0)
    torch.set_num_threads(1)
    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cpu:
        tensor_args = {'device': torch.device('cuda'), 'dtype': torch.float32}
    else:
        tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}

    config = yaml.load(open(CONFIG_ROOT / args.config).read(), Loader=yaml.SafeLoader)

    train_loader, valid_loader, n_feat, data_cfg = get_dataloaders(args, config)

    net, net_opt = build_nets(n_feat, config, tensor_args['device'], model_weight_file=args.finetune, data_cfg=data_cfg)

    # Init. experiment
    exp_path = init_exp_dir(config, args)

    train(
        net, net_opt,
        train_loader,
        valid_loader,
        exp_path,
        args,
    )