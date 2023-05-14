# Dynamics Model Learning

This module is intended for: 
1. Collecting vehicle data,
2. Pre-processing data into a dataset of trajectory sequences,
3. Implementing and training dynamics models for MPPI
4. Testing models in-the-loop with the MPPI controller.

## Data Collection

Manual data collection example:
```bash
python manual_data_collection.py --output_dir train_smallgrid_manual --map_name smallgrid --duration 10 --start_pos -67 336 0.
python manual_data_collection.py --output_dir valid_smallgrid_manual --map_name smallgrid --duration 10 --start_pos -67 336 0.
```
Data will be stored under `$PKG_Path/data/manual_data` by default.

MPPI-based data collection example:
```bash
python mppi_data_collection.py  --output_dir train_smallgrid --map_name smallgrid --duration 100
python mppi_data_collection.py  --output_dir valid_smallgrid --map_name smallgrid --duration 30
```

Data will be stored under `$PKG_Path/data/mppi_data` by default.


## Dataset processing

To create a dataset for training, the collected data needs to be processed and split into
sequences of state-control trajectories with associated BEV maps and observations.

Define a dataset configuration file in the `config/datasets` director. Ex:

```yaml
raw_data_dir: 'mppi_data/'
# Total traj len = future + past + 1 (current)
future_traj_len: 50
past_traj_len: 50
skip_frames: 5

split:
  train:
    - 'train_smallgrid'
  valid:
    - 'valid_smallgrid'

map:
  width: 16 # in meters
  height: 16
  resolution: 0.05

```

where multipled sequences can be specified to create training/validation datasets.


Run data processing as follows, passing the config file and output directory as arugments:
```bash
python process_data.py --cfg small_grid_mppi --output_dir small_grid_mppi --save_vis True
```

Datasets will be stored under `$PKG_Path/data/mppi_data` by default.


## Model Definition
A base class for dynamics models is defined under `models/base.py`.
All other models should be derived from the base class, and implement the
`_forward` and `_rollout` methods, where the latter is used for inference.


## Training

Specify a configuration file for training, under `config/experiments`. Ex.:
```yaml
dataset:
  name: 'small_grid_mppi'
  augment: False
  state_input_key: 'future_states'
  control_input_key: 'future_ctrls'
  ctx_input_keys: ['state']

network:
  state_feat: ['x', 'y', 'th', 'vx', 'wy', 'ay', 'az']
  control_feat: ['steer', 'throttle']

  class: BasicMLP
  net_kwargs:
    hidden_depth: 1
    hidden_dim: 256
    batch_norm: False

  opt: Adam
  opt_kwargs:
    lr: 1.0e-4

loss: StatePredMSE
```

Run the `train.py` script, passing the config file name and output directory.

```bash
python train.py --config single_mlp.yaml --output small_grid --n_epochs 100
```

Model files and output will be stored under the `$PKG_Path/logs` directory by default.
To view training logs and outputs in tensorboard, run the following (in the root directory):

```bash
tensorboard --logdir='./logs' 
```

## Inference (WIP)

See examples under `scripts` for running trained models in-the-loop with the MPPI controller.

