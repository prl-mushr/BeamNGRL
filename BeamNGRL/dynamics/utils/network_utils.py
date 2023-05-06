import torch
import os
import tabulate
import torch.nn as nn
import numpy as np
import glob


def load_model(dir, filename, step=None, load_to_cpu=False):
    import torch
    path = os.path.join(dir, filename)

    if os.path.isfile(path):
        if load_to_cpu:
            return torch.load(path, map_location=lambda storage, location: storage)
        else:
            return torch.load(path)
    else:
        print(path)
        raise Exception('Failed to load model')


def save_model(state, filename):
    torch.save(state, filename)


def load_weights(model_file, net, net_opt):
    state = load_model(os.path.dirname(model_file),
                       os.path.basename(model_file), load_to_cpu=True)
    net.load_state_dict(state['net'])
    net_opt.load_state_dict(state['optim'])
    # Move the parameters stored in the optimizer into gpu
    for opt_state in net_opt.state.values():
        for k, v in opt_state.items():
            if torch.is_tensor(v):
                opt_state[k] = v.to(device='cuda')
    return 0


def _save_model(net, net_opt, epoch, global_args, model_file):
    print('\nSaving model to: ', model_file)
    state = {
        'epoch': epoch,
        'global_args': global_args,
        'optim': net_opt.state_dict(),
        'net': net.state_dict()
    }
    save_model(state, model_file)


def make_train(nets):
    for module in nets.values():
        module.train()


def make_eval(nets):
    for module in nets.values():
        module.eval()


def set_device(nets, device):
    for module in nets.values():
        module.to(device)


def parameters(nets):
    params = []
    for net in nets.values():
        params += list(net.parameters())
    return params


def module_grad_stats(module):
    headers = ['layer', 'max', 'min']

    def maybe_max(x):
        return x.max() if x is not None else 'None'

    def maybe_min(x):
        return x.min() if x is not None else 'None'

    data = [
        (name, maybe_max(param.grad), maybe_min(param.grad))
        for name, param in module.named_parameters()
    ]
    return tabulate.tabulate(data, headers, tablefmt='psql')


def module_weights_stats(module):
    headers = ['layer', 'max', 'min']

    def maybe_max(x):
        return x.max() if x is not None else 'None'

    def maybe_min(x):
        return x.min() if x is not None else 'None'

    data = [
        (name, maybe_max(param), maybe_min(param))
        for name, param in module.named_parameters()
    ]
    return tabulate.tabulate(data, headers, tablefmt='psql')


def get_gradient_magnitude(module):
    gradients = []
    for name, param in module.named_parameters():
        if param.grad is not None:
            gradients.append(abs(float(param.grad.mean())))
    return np.mean(gradients)


def get_grad_mag(nets):
    stats = []
    for model in nets.values():
        stats.append(get_gradient_magnitude(model))
    return np.mean(stats)
