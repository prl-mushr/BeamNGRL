import os
import yaml
import BeamNGRL.dynamics.models as models
import BeamNGRL.dynamics.utils.loss_utils as losses
from BeamNGRL.dynamics.datasets import get_datasets
from torch import optim
import BeamNGRL.dynamics.utils.network_utils as nu
from BeamNGRL import *
import torch


def get_data_config(config):
    dataset_path = DATASETS_PATH / config['dataset']['name']
    dataset_config_path = dataset_path / 'config.yaml'
    data_config = yaml.load(open(dataset_config_path).read(), Loader=yaml.SafeLoader)
    return data_config, dataset_path


def get_dataloaders(args, config):
    spec = config['dataset'].copy()
    spec.pop('name')
    data_cfg, dataset_path = get_data_config(config)

    train_loader, valid_loader, data_stats = get_datasets(
        bs=args.batchsize,
        shuffle=args.shuffle,
        dataset_path=dataset_path,
        map_cfg=data_cfg['map'],
        **spec,
    )
    return train_loader, valid_loader, data_stats, data_cfg


def build_nets(config, tn_args, model_weight_file=None, data_stats=None):
    spec = config['network']
    net_class = getattr(models, spec['class'])
    net_args = spec.get('net_kwargs', {})
    net = net_class(
        state_feat=spec['state_feat'],
        ctrl_feat=spec['control_feat'],
        input_stats=data_stats,
        **net_args,
    ).to(**tn_args)

    opt = None
    if spec.get('opt') is not None:
        opt = getattr(optim, spec['opt'])(net.parameters(), **spec['opt_kwargs'])

    if model_weight_file is not None:
        nu.load_weights(model_weight_file, net, opt)

    print(net.standardizer)

    return net, opt


def get_loss_func(config):
    loss_func = getattr(losses, config['loss'])()
    return loss_func


def init_exp_dir(config, args):
    # Make exp dir
    log_dir = LOGS_PATH / args.output
    os.makedirs(log_dir, exist_ok=True)

    # Update and save config file
    kwargs = vars(args)
    config.update(kwargs)
    with open(log_dir / 'config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, sort_keys=False)
    return log_dir
