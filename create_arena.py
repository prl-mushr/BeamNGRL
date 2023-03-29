# import cv2
import numpy as np


# center  = np.array([-302, -324])
# arena_start  = np.array([-340, -268])
# arena_end = np.array([-262, -376])

# mud_start = np.array([-336, -276])
# mud_end = np.array([-304, -372])

# sand_start = np.array([-300, -276])
# sand_end = np.array([-268, -372])
data = []
data.append([-302, -324,  100])

data.append([-304, -276,  100])
data.append([-336, -276,  100])
data.append([-336, -372,  100])
data.append([-304, -372,  100])

data.append([-268, -276,  100])
data.append([-300, -276,  100])
data.append([-300, -372,  100])
data.append([-268, -372,  100])

np.save("WP_file_arena.npy",data)