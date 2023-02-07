import numpy as np
import math as m
import cv2
from key_check import key_press
from controller import * # backend that contains the code for manual control 
from model_runner import model_runner # backend that contains the code for running the trained models.
import threading # we use threading to run stuff asynchronously 
import time


class driver():
    def __init__(self,manual_control='keyboard',run_NN=False,model="steering"):
        self.init = False # if the state has been initialized
        self.training_data = [] # empty list for training data
        self.Finish = False # whether we've finished recording or not
        self.Rec = False # whether we're recording or not
        self.HEIGHT = 240 #image dimensions
        self.WIDTH = 320
        self.CHANNELS = 1 #number of channels in the input image
        self.WHEELBASE = 1.56 # wheelbase of the car
        self.dt = 0 # time step
        self.cam_img = None

        print('INFO message: using keyboard control')
        self.input = keyboard_control()
        proc_remote_control = threading.Thread(target=self.input.read_loop) # daemon -> kills thread when main program is stopped
        proc_remote_control.setDaemon(True)
        proc_remote_control.start()

        self.NN = None
        #set model type:
        if(run_NN):
            if(model == "steering"):
                self.NN = model_runner('steering',self.WHEELBASE)
                proc_model_runner = threading.Thread(target=self.NN.run_model) # daemon -> kills thread when main program is stopped
                proc_model_runner.setDaemon(True)
                proc_model_runner.start()
            elif(model == "bezier"):
                self.NN = model_runner('bezier',self.WHEELBASE)
                proc_model_runner = threading.Thread(target=self.NN.run_model) # daemon -> kills thread when main program is stopped
                proc_model_runner.setDaemon(True)
                proc_model_runner.start()             
            elif(model=="image-image"):
                self.NN = model_runner('image_image',self.WHEELBASE)
                proc_model_runner = threading.Thread(target=self.NN.run_model) # daemon -> kills thread when main program is stopped
                proc_model_runner.setDaemon(True)
                proc_model_runner.start()
            else:
                print("incorrect model type!")
        else:
            print("running manual. no neural net initialized")
        self.now = time.time()

    def initialize(self, X, Y, head, speed,WB):
        self.X = X
        self.Y = Y
        self.last_X = X
        self.last_Y = Y
        self.mh = None
        self.speed = speed
        self.WB = WB
        self.init = True
        self.cam_img = None

    def calc_head(self): # calculates heading of the car by taking inverse tangent of dy/dx (movement)
        dy = self.Y - self.last_Y
        dx = self.X - self.last_X
        self.last_Y = self.Y
        self.last_X = self.X
        if(m.fabs(dx)<0.001 and m.fabs(dy)<0.001):
            self.mh = None
        else:
            self.mh = m.atan2(dy,dx)
            # print(self.mh*57.3)

    def update_state(self,X,Y,speed,steer,image_C,image_L,image_R,time_stamp):
        self.X = X
        self.Y = Y
        self.speed = speed
        self.calc_head()
        size = image_C.shape[0]
        # how much we want to trim the images. The image that comes in is a 320x320 image. we need 320x240
        top = size//8
        bottom = (size*7)//8
        self.cam_img = image_C[top:bottom,:]
        image_L = image_L[top:bottom,:]
        image_R = image_R[top:bottom,:]

        pos = [self.X,self.Y,self.mh,self.speed,steer] # pos refers to present operating state
        self.dt = time.time()-self.now
        self.now = time.time()
        #if neural net exists, run it
        if(self.NN):
            self.NN.update_model_input(self.cam_img, self.speed, self.input.th, self.dt,time_stamp)

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

        if(self.mh is not None and self.Rec == True):
            self.training_data.append([self.cam_img,image_L,image_R,pos]) # if recording and car is moving, append data to the list.
