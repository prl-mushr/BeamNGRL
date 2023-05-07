import os
import yaml
import BeamNGRL.dynamics.models as models
from BeamNGRL.dynamics.datasets import get_datasets
from torch import optim
import BeamNGRL.dynamics.utils.network_utils as nu
from BeamNGRL import *


def get_data_config(config):
    dataset_path = DATASETS_PATH / config['dataset']['name']
    dataset_config_path = dataset_path / 'config.yaml'
    data_config = yaml.load(open(dataset_config_path).read(), Loader=yaml.SafeLoader)
    return data_config, dataset_path


def get_dataloaders(args, config):
    spec = config['dataset']
    data_cfg, dataset_path = get_data_config(config)

    train_loader, valid_loader = get_datasets(
        bs=args.batchsize,
        shuffle=args.shuffle,
        aug=spec['augment'],
        dataset_path=dataset_path,
        map_cfg=data_cfg['map'],
    )
    return train_loader, valid_loader, data_cfg


def build_nets(config, device, model_weight_file=None, data_cfg=None):
    spec = config['network']
    net_class = getattr(models, spec['class'])
    net_args = spec.get('net_kwargs', {})
    net = net_class(
        data_cfg=data_cfg,
        **net_args,
    ).to(device)

    opt = None
    if spec['opt'] != 'none':
        opt = getattr(optim, spec['opt'])(net.parameters(), **spec['opt_kwargs'])

    if model_weight_file is not None:
        nu.load_weights(model_weight_file, net, opt)

    return net, opt


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
