import time
import traceback
import numpy as np
from key_check import *


def check_range(x,lim):
    if(x>lim):
        return lim
    elif(x<-lim):
        return -lim
    return x

class keyboard_control():
    def __init__(self):
        self.st = 0.0
        self.th = 0.0
        self.autonomous = False
        reset_mouse_pos()

    def read_loop(self):
        while 1:
            try:
                time.sleep(0.01)
                self.st,self.th = get_mouse_pos()
                if(key_press() == ['A']):
                    self.autonomous = True
                if(key_press() == ['M']):
                    self.autonomous = False
                self.st = check_range(self.st,1.0)
                self.th = check_range(self.th,1.0)
                # print(steering,throttle)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                # pass
                print(traceback.format_exc())
            except:
                pass

if __name__ == "__main__": 
    obj = keyboard_control()
    obj.read_loop()
