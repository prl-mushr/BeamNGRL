from pytorch_high_level import *
from pytorch_model import *
from Bezier import *
from point_warping import *
from scipy.ndimage.filters import gaussian_filter

class model_runner():
    def __init__(self,args,WHEELBASE):
        self.WHEELBASE = WHEELBASE
        self.model_name = args
        self.model_sleep = 1/30 # default sleep time. the models usually can execute very quickly (3-5 ms on gpu)
        self.cam_img = None 
        self.speed = 0
        self.dt = 1
        self.new_data = False
        self.st = 0
        self.th = 0
        self.time_stamp = time.time()
        self.WIDTH = 320
        self.HEIGHT = 240
        self.cam_height = 0.845 # found from 3d model of car
        self.X_max = self.WIDTH #image pixels in horizontal direction
        self.Y_max = self.HEIGHT #image pixels in vertical direction
        self.X_Center = self.X_max//2 #assuming image center is at geometric center (camera matrix should be used, but this will do for now)
        self.Y_Center = self.Y_max//2
        self.pitch = 0
        self.focalLength = 50 #hd1080
        self.fov_h = 90/57.3 #fov calculation (copied straight from zed's website, so if it fails, sue them not me!)
        self.K_h = m.tan(self.fov_h/2)
        self.fov_v = 58/57.3
        self.K_v = m.tan(self.fov_v/2)
        # selecting type of neural net:
        if (args == 'steering'):
            checkpoint = torch.load('steering.h5')
            self.model = checkpoint['net']
            self.model = self.model.to(device)
            print('steering model loaded')
        elif (args == 'bezier'):
            checkpoint = torch.load('bezier.h5')
            self.model = checkpoint['net']
            self.model = self.model.to(device)
            self.model_sleep = 1/120
            print('bezier model loaded')
        elif (args == 'image_image'):
            checkpoint = torch.load('trajectory_1.h5')
            self.model = checkpoint['net']
            self.model = self.model.to(device)
            self.roi_vert = np.array([[self.WIDTH*0.05,self.HEIGHT*0.99],[self.WIDTH*0.1,self.HEIGHT*0.6],[self.WIDTH*0.9,self.HEIGHT*0.6],[self.WIDTH*0.95,self.HEIGHT*0.99]],dtype=np.int32)
            self.kernel = np.ones((9,9),np.uint8)
            print('image-image model loaded')
        else:
            print("invalid model name")

    def update_model_input(self, img, speed, th, dt,time_stamp):
        self.cam_img = np.array(img,dtype=np.float32)
        self.speed = speed
        self.dt = dt
        self.new_data = True
        self.input_th = th
        self.time_stamp = time_stamp

    def roi(self,img,vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask,vertices,255)
        mask = cv2.bitwise_and(img,mask)
        return mask
    def run_model(self):
        with torch.no_grad(): # this is important!
            while True:
                if(self.new_data):
                    time_stamp = self.time_stamp
                    self.new_data = False
                    if(self.cam_img is not None and self.model is not None):
                        img = torch.as_tensor(self.cam_img).view(-1,1,240,320).to(device)
                        if(self.model_name == 'steering'):
                            output = self.model(img)[0]
                            self.st = float(output[1] - output[0])
                            self.th = self.input_th # feedforward on the throttle
                        if(self.model_name == 'bezier'):#bezier curve following using curvature controller
                            horizon = 6.5
                            traj = self.model(img)[0].cpu().numpy()
                            Px = np.array([0,0,traj[0]/2.5,traj[1]/2])*horizon
                            Py = np.array([0,traj[2]/2,traj[3],traj[4]])*horizon
                            for i in range(5):
                                delta = (time.time()- time_stamp)+0.025
                                # print(delta)
                                t = (self.speed*delta)/horizon
                                Curv = -get_Curvature(Px[0],Py[0],Px[1],Py[1],Px[2],Py[2],Px[3],Py[3],t)
                                self.st = (m.atan(self.WHEELBASE*Curv)*57.3)/16 #negative sign because the simulator is stupid
                                self.th = self.input_th #feedforward
                                time.sleep(0.01)
                        if(self.model_name == 'image_image'): #trajectory following using stanley controller
                            traj = self.model(img)[0].cpu().numpy()
                            traj = traj.astype(np.uint8)
                            traj = traj.reshape((240,320,1))
                            ret, traj = cv2.threshold(traj, 0, 255, cv2.THRESH_BINARY)
                            traj = self.roi(traj,[self.roi_vert])
                            traj = cv2.morphologyEx(traj, cv2.MORPH_OPEN, self.kernel)
                            traj = cv2.morphologyEx(traj, cv2.MORPH_CLOSE, self.kernel)
                            index = np.where(traj)
                            index = np.array(index)
                            top = np.min(index[0])
                            bottom = np.max(index[0])
                            x = []
                            y = []
                            for i in reversed(range(top,bottom)):
                                ind = np.where(index[0]==i)
                                array = index[1,ind]
                                pos = np.mean(array)
                                if not np.isnan(pos):
                                    x.append(pos)
                                else:
                                    x.append(x[-1])
                                y.append(i)
                            x = np.array(x)
                            y = np.array(y)
                            x = gaussian_filter(x,sigma=10)
                            x,y = img2XY(x, y,self.cam_height,self.K_v,self.K_h,self.X_max,self.Y_max,self.X_Center,self.Y_Center,self.pitch)
                            dx = np.diff(x)
                            dy = np.diff(y)
                            slope = dx/dy
                            # delta = time.time()-time_stamp
                            # dist = self.speed*delta
                            # pos = np.min(np.where(y>dist*0.9,y,y<dist*1.1))
                            cte = x[20]
                            theta = m.atan(slope[20])
                            intersect_dist = max(self.speed,6)
                            print(cte,theta)
                            self.st = (theta + m.atan(cte*0.1/intersect_dist))*57.3/16
                            self.th = self.input_th
                    else:
                        print("no model")
                
                else:
                    time.sleep(self.model_sleep)


