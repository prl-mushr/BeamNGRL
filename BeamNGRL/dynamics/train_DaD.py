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
import traceback
import torch.nn.functional as F


def train(
        network,
        optimizer,
        loss_func,
        train_loader,
        valid_loader,
        exp_path,
        config,
        args,
        tn_args: Dict = None,
        skip = 1,
        best_loss = np.inf,
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
    best_model = None

    try:
        for epoch in range(args.n_epochs):
            network.train()
            train_average_loss = []
            for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()

                states_tn = states_tn.to(**tn_args)[:,::skip,:]
                controls_tn = controls_tn.to(**tn_args)[:,::skip,:]
                ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}
                targets = states_tn.clone()

                pred = network(
                    states_tn,
                    controls_tn,
                    ctx_tn_dict,
                )

                loss = loss_func(pred, targets, conf = config)
                loss.backward()
                optimizer.step()

                # Logging
                train_average_loss.append(loss.detach().cpu().numpy()) # scaling by mean of loss multiplier for visulization only.
                grad_mag = nu.get_gradient_magnitude(network)
                writer.add_scalar('Train/batchLoss', loss,
                    len(train_loader) * epoch + i)
                writer.add_scalar('Train/gradient', grad_mag,
                    len(train_loader) * epoch + i)

            train_average_loss = np.asarray(train_average_loss).mean()
            writer.add_scalar('Train/Loss', train_average_loss, epoch)

            
            network.eval()
            valid_avg_loss = []
            with torch.no_grad():
                for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(valid_loader)):
                    states_tn = states_tn.to(**tn_args)[:,::skip,:]
                    controls_tn = controls_tn.to(**tn_args)[:,::skip,:]
                    ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}
                    targets = states_tn.clone()

                    pred = network(
                        states_tn,
                        controls_tn,
                        ctx_tn_dict,
                    )

                    loss = loss_func(pred, targets, conf = config)
                    
                    valid_avg_loss.append(loss.cpu().numpy()) # this is for visualization only.
                    writer.add_scalar('Valid/batchLoss', loss,
                        len(valid_loader) * epoch + i)


            valid_loss = np.asarray(valid_avg_loss).mean()
            writer.add_scalar('Valid/Loss', valid_loss, epoch)

            if valid_loss < best_loss or epoch % args.save_each_n_epochs == 0:
                print('New best model found, saving...')
                nu._save_model(
                    network,
                    optimizer,
                    epoch,
                    None,
                    os.path.join(exp_path, "best_%d.pth" % epoch),
                )
                best_loss = valid_loss
                best_model = "{}/best_{}.pth".format(args.output, epoch)

            if net_sched is not None:
                net_sched.step()

    except Exception:
        print(traceback.format_exc())
        writer.close()
        exit()
    finally:
        writer.close()

    return best_model, best_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="baseline_mlp", help='config file for training model')
    parser.add_argument('--output', type=str, required=False, help='location to store output - weights, log')
    parser.add_argument('--cpu', type=str, required=False, default=False, help='use cpu for training') # why would you use cpu.
    parser.add_argument('--n_epochs', type=int, required=False, default=100, help='Number of training epochs.')
    parser.add_argument('--scheduler', action='store_true', help='use scheduler (cosine annealing)')
    parser.add_argument('--finetune', type=str, required=False, default = None, help='pretrained weights to finetune from')
    parser.add_argument('--shuffle', type=bool, required=False, default=True, help='shuffle data')
    parser.add_argument('--batchsize', type=int, required=False, default=1, help='training batch size')
    parser.add_argument('--start_from', type=int, required=False, default=-1, help='epoch to start from')
    parser.add_argument('--save_each_n_epochs', type=int, required=False, default=1000, help='Save model after each epoch.')
    args = parser.parse_args()
    # TODO: training config files should only be in the dynamics folder, not in the experiments folder.

    # Set torch params
    tn_args = {'device': torch.device('cuda'), 'dtype': torch.float32}

    # Load experiment config
    config_path = str(ROOT_PATH.parent) + "/Experiments/Configs/" + '{}.yaml'.format(args.config)
    config = yaml.load(open(config_path).read(), Loader=yaml.SafeLoader)

    torch.manual_seed(2) #config.seed) # TODO: add seed to config file.
    torch.set_num_threads(1) #config.num_threads)

    # override the map config using the specification provided by the dataset config.
    dataset_config = yaml.load(open( str(ROOT_PATH.parent) + "/data/datasets/" + config["dataset"]["name"] + "/config.yaml" ).read(), Loader=yaml.SafeLoader)
    config["Map_config"] = dataset_config["Map_config"] # copy from dataset config
    config["network"]["net_kwargs"]["BEVmap_size"] = config["Map_config"]["map_size"]
    config["network"]["net_kwargs"]["BEVmap_res"] = config["Map_config"]["map_res"]
    config["network"]["net_kwargs"]["patch_size"] = config["Dynamics_config"]["patch_size"]

    # Dataloaders
    train_loader, valid_loader, stats, data_cfg = get_dataloaders(args, config)

    # Model init.

    finetune_weights = None
    if args.finetune is not None:
        finetune_weights = str(ROOT_PATH.parent) + "/logs/" + args.finetune
    net, net_opt = build_nets(
        config, tn_args,
        model_weight_file=finetune_weights,
        data_stats=stats,
    )

    loss_func = get_loss_func(config)

    # Experiment init -- this is where the models and logs are stored
    exp_path = init_exp_dir(config, args)

    skip = 1
    try:
        skip = int(config["Dynamics_config"]["dt"]/config["dataset_dt"])
    except:
        print("please specify dataset_dt in config file")
        exit()

    best_loss = np.inf
    
    best_model_new, best_loss = train(
        net, net_opt,
        loss_func,
        train_loader,
        valid_loader,
        exp_path,
        config,
        args,
        tn_args,
        skip=skip,
        best_loss = best_loss
    )

# python3 train_DaD.py --config baseline_Config --output base --n_epochs 50 --batchsize=16