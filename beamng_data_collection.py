from beamng_interface import *
from key_check import key_press

class car():
    def __init__(self,manual_control='keyboard',run_NN=False,model="steering"):
        self.init = False # if the state has been initialized
        self.training_data_depth = [] # empty list for training data
        self.training_data_color = []
        self.training_data_segmt = []
        self.training_data_intrinsic = [] # empty list for training data
        self.Finish = False # whether we've finished recording or not
        self.Rec = False # whether we're recording or not
        self.HEIGHT = 300 #image dimensions
        self.WIDTH = 300
        self.dt = 0 # time step
        self.cam_img = None
        self.interface = beamng_interface()
        self.folder = 'raw_data/'
        self.index = 0
        # load scenario:
        proc_data = threading.Thread(target = self.update_state)
        proc_data.daemon = True
        proc_data.start()
        print("collector started")
        self.interface.load_scenario()

    def save_data(self):
        self.training_data_intrinsic = np.array(self.training_data_intrinsic)
        self.training_data_depth = np.array(self.training_data_depth)
        self.training_data_color = np.array(self.training_data_color)
        self.training_data_segmt = np.array(self.training_data_segmt)
        filename_intrinsic = self.folder + 'training_data_intrinsic_' + str(self.index) + '.npy'
        filename_depth = self.folder + 'training_data_depth_' + str(self.index) + '.npy'
        filename_color = self.folder + 'training_data_color_' + str(self.index) + '.npy'
        filename_segmt = self.folder + 'training_data_segmt_' + str(self.index) + '.npy'
        np.save(filename_intrinsic, self.training_data_intrinsic)
        np.save(filename_depth, self.training_data_depth)
        np.save(filename_color, self.training_data_color)
        np.save(filename_segmt, self.training_data_segmt)
        print("data saved")

    def update_state(self):
        self.Rec = True
        while(self.interface.depth is None or self.interface.pos is None):
            time.sleep(0.1)
        print("state initialized")
        for i in range(1000):
            try:
                self.now = time.time()
                if(self.interface.pos is None):
                    print("we got a problem")
                pos = np.hstack((self.interface.pos, self.interface.quat, self.interface.vel, self.interface.G, self.interface.A)) # pos refers to present operating state
                depth = self.interface.depth
                color = self.interface.color
                segmt = self.interface.segmt

                self.dt = time.time() - self.now
                time.sleep(0.1 - self.dt)  # record data at 10 fps.
                if(key_press() == ['O'] ):
                    print("recording aborted")
                    self.Rec = False
                    self.Finish = True
                if(key_press() == ['K'] ):
                    if self.Rec:
                        print("recording paused")
                        self.Rec = False
                        time.sleep(0.5)
                    elif not self.Rec :
                        print("recording continued")
                        self.Rec = True
                        time.sleep(0.5)

                if(self.Rec == True):
                    self.training_data_intrinsic.append(pos) # if recording and car is moving, append data to the list.
                    self.training_data_depth.append(depth)
                    self.training_data_color.append(color)
                    self.training_data_segmt.append(segmt)

                if(self.Finish == True or (i==999 and self.Rec == True)):
                    self.save_data()
                    exit()
            except KeyboardInterrupt:
                self.save_data()
                exit()
            except Exception as e:
                print(traceback.format_exc())
                time.sleep(0.1)

# run main code:
if __name__ == "__main__":
    # initialize car:
    car_ = car()