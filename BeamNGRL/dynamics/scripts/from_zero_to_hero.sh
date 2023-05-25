#!/bin/bash
python ../sinusoidal_data_collection.py  --output_dir train_smallgrid --map_name smallgrid --duration 720
cp -r ../../../data/sinu_data/train_smallgrid/ ../../../data/sinu_data/valid_smallgrid/
python ../sinusoidal_data_collection.py  --output_dir train_smallgrid --map_name smallgrid --duration 720 --onlysteer True
cp -r ../../../data/sinu_data_steer/train_smallgrid/ ../../../data/sinu_data_steer/valid_smallgrid/
python ../sinusoidal_data_collection.py  --output_dir train_smallgrid --map_name smallgrid --duration 720 --onlyspeed True
cp -r ../../../data/sinu_data_speed/train_smallgrid/ ../../../data/sinu_data_speed/valid_smallgrid/
# sleep 60
# python ../process_data.py --cfg small_grid_sinu --output_dir small_grid_sinu --save_vis False

# python ../train.py --config single_residual_mlp.yaml --output small_grid --n_epochs 300