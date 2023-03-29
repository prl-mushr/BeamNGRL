# BeamNGRL

map data and Game files:
https://drive.google.com/drive/folders/18iNHC1p8-gldngf0vC5Z-OkKzgznyVw9?usp=sharing

```
BEV_data_generator.py
```
is for visualizing BEV for pre-recorded poses (found in the map data files)

```
Extract_map.py
```
Performs photogrammetry to obtain the maps from raw data (map_data_binary_50)

```
beamng_map_generator.py
```
Collects photos of the map from different locations for the photogrammetry. This has to be run on windows.

```
beamngpy_test.py
```
Is the file I use to just boot up the simulator and play around. It additionally demonstrates how to get car data (beyond just pose,twist,accel).
I use it for obtaining map positions where I can spawn the car.

I will add more documentation as I figure out how we're using this for IRL.
