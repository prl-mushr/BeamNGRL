import torch
import os
import tabulate
import torch.nn as nn
import numpy as np
import glob
from typing import List


# states:
#  0: x,    1: y,    2: z,     (wf)
#  3: r,    4: p,    5: th,    (wf)
#  6: vx,   7: vy,   8: vz,    (bf)
#  9: ax,   10: ay,  11: az,   (bf)
#  12: wx,  13: wy,  14: wz    (bf)

state_idx_map = {
    'x':  0,
    'y':  1,
    'z':  2,
    'r':  3,
    'p':  4,
    'th': 5,
    'vx': 6,
    'vy': 7,
    'vz': 8,
    'ax': 9,
    'ay': 10,
    'az': 11,
    'wx': 12,
    'wy': 13,
    'wz': 14,
}

state_feat_map = {
    'x':        (lambda arr: arr[..., [0]]),
    'y':        (lambda arr: arr[..., [1]]),
    'z':        (lambda arr: arr[..., [2]]),
    'r':        (lambda arr: arr[..., [3]]),
    'p':        (lambda arr: arr[..., [4]]),
    'th':       (lambda arr: arr[..., [5]]),
    'vx':       (lambda arr: arr[..., [6]]),
    'vy':       (lambda arr: arr[..., [7]]),
    'vz':       (lambda arr: arr[..., [8]]),
    'ax':       (lambda arr: arr[..., [9]]),
    'ay':       (lambda arr: arr[..., [10]]),
    'az':       (lambda arr: arr[..., [11]]),
    'wx':       (lambda arr: arr[..., [12]]),
    'wy':       (lambda arr: arr[..., [13]]),
    'wz':       (lambda arr: arr[..., [14]]),

    'dvx_dt':       (lambda arr: arr[..., 1:, [6]] - arr[..., :-1, [6]]),
    'dvy_dt':       (lambda arr: arr[..., 1:, [7]] - arr[..., :-1, [7]]),

    'sin_th':   (lambda arr: torch.sin(arr[..., [5]])),
    'cos_th':   (lambda arr: torch.cos(arr[..., [5]])),
}

ctrl_feat_map = {
    'steer':    (lambda arr: arr[..., [0]]),
    'throttle': (lambda arr: arr[..., [1]]),
}


def get_feat_index_tn(
        feat_list: List,
):
    idx_list = []
    for feat in feat_list:
        idx_list += [int(state_idx_map[feat])]
    return torch.LongTensor(idx_list)


def get_state_features(
        states: torch.Tensor,
        feat_list: List,
        dt: float = None,
):
    state_feats = []
    for f in feat_list:
        feat = state_feat_map[f](states)
        if f in ['dvx_dt', 'dvx_dt']:
            feat = torch.cat((feat, feat[:, [-1], :]), dim=-1) # repeat last entry
            feat = feat / dt
        state_feats.append(feat)
    return torch.cat(state_feats, dim=-1)


def get_ctrl_features(
        controls: torch.Tensor,
        feat_list: List,
):
    ctrl_feats = []
    for f in feat_list:
        ctrl_feats.append(ctrl_feat_map[f](controls))
    return torch.cat(ctrl_feats, dim=-1)


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


def load_weights(model_file, net, net_opt=None):
    state = load_model(os.path.dirname(model_file),
                       os.path.basename(model_file), load_to_cpu=True)

    net.load_state_dict(state['net'])
    if net_opt is not None:
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
