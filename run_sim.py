import os
import random
import json
import time
from io import BytesIO
import base64
import argparse
from PIL import Image
import numpy as np
from gym_donkeycar.core.sim_client import SDClient
from CAR import * # this provides abstraction for control or I/O
import traceback # this is just for seeing the cause of the error if one is thrown during runtime without stopping the code
###########################################

steering_log = []

class SimpleClient(SDClient):

    def __init__(self, address, args, poll_socket_sleep_time=0.001):
        super().__init__(*address, poll_socket_sleep_time=poll_socket_sleep_time)
        self.last_image = None
        self.car_loaded = False
        self.car = driver(manual_control=args.manual_control,run_NN=args.test,model=args.model) # this class is defined in the file CAR.py

    def on_msg_recv(self, json_packet):
        # global car
        if json_packet['msg_type'] == "car_loaded":
            self.car_loaded = True
        
        if json_packet['msg_type'] == "telemetry":
            # the images that come in are grayscale except for the center image, which is RGB. This was done to maintain compatibility with Reinforcement learning script
            # the imitation learning model uses grayscale images and therefore in this particular script, all images are converted to grayscale (as even gray images come in as RGB images with all channels being equal)
            time_stamp = time.time() # this time stamp is useful for time-sensitive control algorithms.
            imgString = json_packet["image_C"] # center camera image
            image = Image.open(BytesIO(base64.b64decode(imgString)))
            image_C = np.asarray(image)
            image_C = cv2.cvtColor(image_C,cv2.COLOR_BGR2GRAY)
            imgString = json_packet["image_L"] #left camera image
            image = Image.open(BytesIO(base64.b64decode(imgString)))
            image_L = np.asarray(image)
            image_L = cv2.cvtColor(image_L,cv2.COLOR_BGR2GRAY)
            imgString = json_packet["image_R"] # right camera image
            image = Image.open(BytesIO(base64.b64decode(imgString)))
            image_R = np.asarray(image)
            image_R = cv2.cvtColor(image_R,cv2.COLOR_BGR2GRAY)

            if(not self.car.init):
                self.car.initialize(json_packet["pos_x"],json_packet["pos_z"],0,json_packet["speed"],0.5)
            else:
                self.car.update_state(json_packet["pos_x"],json_packet["pos_z"],json_packet["speed"],json_packet["steering_angle"],image_C,image_L,image_R,time_stamp)
            #don't have to, but to clean up the print, delete the image string.
            steering_log.append(json_packet["steering_angle"])
            del json_packet["image_C"]
            del json_packet["image_L"]
            del json_packet["image_R"]

    def send_controls(self, steering, throttle):
        p = { "msg_type" : "control",
                "steering" : steering.__str__(),
                "throttle" : throttle.__str__(),
                "brake" : "0.0" }
        msg = json.dumps(p)
        self.send(msg)
        #this sleep lets the SDClient thread poll our message and send it out.
        time.sleep(self.poll_socket_sleep_sec)

    def update(self):
        if(not self.car.input.autonomous):
            st = self.car.input.st
            th = self.car.input.th
        else:
            if(self.car.NN is not None):
                st = self.car.NN.st
                th = self.car.NN.th
            else:
                st = 0
                th = 0

        self.send_controls(st, th)



###########################################
## Make some clients and have them connect with the simulator

def test_clients(args):
    # test params
    host = "127.0.0.1" # local host
    port = 9091
    num_clients = 1
    clients = []
    file_name = "MUSHR_320x240_{}.npy".format(args.dataset_name)

    # Start Clients
    for _ in range(0, num_clients):
        c = SimpleClient(address=(host, port),args=args,poll_socket_sleep_time=0.0)
        clients.append(c)

    time.sleep(1)
    # Load Scene message. Only one client needs to send the load scene.
    # msg = '{ "msg_type" : "load_scene", "scene_name" : "generated_track" }'
    # msg = '{ "msg_type" : "load_scene", "scene_name" : "warehouse" }'
    # msg = '{ "msg_type" : "load_scene", "scene_name" : "avc2" }'
    # msg = '{ "msg_type" : "load_scene", "scene_name" : "generated_road" }'
    # msg = '{ "msg_type" : "load_scene", "scene_name" : "MUSHR_track" }'
    msg = '{ "msg_type" : "load_scene", "scene_name" : "'+args.env_name+ '" }'
    clients[0].send(msg)
    # Wait briefly for the scene to load.
    loaded = False
    while(not loaded):
        time.sleep(1.0)
        for c in clients:
            loaded = c.car_loaded           
    # Car config
    msg = '{ "msg_type" : "car_config", "body_style" : "mushr", "body_r" : "0", "body_g" : "0", "body_b" : "255", "car_name" : "MUSHR", "font_size" : "100" }' # do not change
    clients[0].send(msg)
    time.sleep(1)

    do_drive = True
    while not c.car.Finish and do_drive:
        try:
            for c in clients:
                c.update()
                if c.aborted or c.car.Finish:
                    print("Client socket problem, stopping driving.")
                    do_drive = False
        except KeyboardInterrupt:
            exit()
        
        except Exception as e:
            print(traceback.format_exc())

        except:
            pass

    print("saving")
    if(len(c.car.training_data)>10):
        np.save(file_name,c.car.training_data)
    time.sleep(1.0)
    log_name = args.log_name
    np.save("steering_log_{}.npy".format(log_name),np.array(steering_log))
    # Exist Scene
    msg = '{ "msg_type" : "exit_scene" }'
    clients[0].send(msg)

    time.sleep(1.0)

    # Close down clients
    print("waiting for msg loop to stop")
    for c in clients:
        c.stop()

    print("clients to stopped")



if __name__ == "__main__":
    env_list = [
       "warehouse",
       "generated_road",
       "avc2",
       "generated_track",
       "MUSHR_track",
       "MUSHR_benchmark"
    ]
    model_list = [
    "steering",
    "bezier",
    "image-image"
    ]

    parser = argparse.ArgumentParser(description='ddqn')
    parser.add_argument('--dataset_name', type=str, default="test", help='suffix for the dataset name')
    parser.add_argument('--model', type=str, default="steering", help='type of model', choices=model_list)
    parser.add_argument('--test', type=bool, default = False, help='agent uses learned model to navigate env')
    parser.add_argument('--manual_control', type=str, default="keyboard", help='port to use for websockets')
    parser.add_argument('--log_name', type=str, default="_", help='constant throttle for driving')
    parser.add_argument('--env_name', type=str, default='generated_road', help='name of donkey sim environment', choices=env_list)

    args = parser.parse_args()
    test_clients(args)
