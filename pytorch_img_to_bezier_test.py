from pytorch_high_level import *
from pytorch_model import Bezier
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import matplotlib.pyplot as plt
from Bezier import *
import argparse
import os

parser = argparse.ArgumentParser(description='IL')
parser.add_argument('--dataset_name', type=str, default="4", help='suffix for the dataset name')
args = parser.parse_args()

HEIGHT = 240
WIDTH = 320
CHANNELS = 1
model_name = 'bezier.h5'

checkpoint = torch.load(model_name)
model = checkpoint['net']
train_log = []
output_path = os.getcwd()+"/Outputs_pytorch"

def plot_steering(s):
    WHEELBASE = 1
    Curvature = m.tan(s/57.3)/WHEELBASE
    if(m.fabs(Curvature)<1e-4):
        y = np.arange(0,1,0.05)
        x = np.zeros_like(y)
    else:
        radius = 1/Curvature
        yc = 0
        y = np.arange(0,1,0.05)
        x = np.sign(radius)*np.sqrt(radius**2 - y**2) - radius
    return x,y

counter = 0
torch.cuda.empty_cache()
train_data = np.load('MUSHR_320x240_shuffled_Bezier_{}.npy'.format(args.dataset_name),allow_pickle=True)
X = ([i[0] for i in train_data])
Y = ([i[1] for i in train_data])
st = ([i[2] for i in train_data])

with torch.no_grad():
    for j in range(0,len(X),1000):
        img = X[j]
        img = torch.Tensor(img).view(-1,1,240,320).to(device)
        expected = Y[j]
        angle = -st[j]
        now = time.time()
        traj = model(img)[0].cpu().numpy()
        dt = time.time()-now
        # print(dt*1000)
        print(traj,Y[j])
        Bx,By = plot_bezier_coeffs(traj)
        _Bx,_By = plot_bezier_coeffs(expected)
        st_x,st_y = plot_steering(angle)
        plt.plot(Bx*6.5,By*6.5,label='predicted')
        # plt.plot(_Bx,_By,label='ground_truth')
        # plt.plot(st_x,st_y,label='instantaneous steering trajectory')
        plt.xlabel('x axis (m)')
        plt.ylabel('y axis (m)')
        plt.legend()
        plt.axis('equal')
        # plt.show()
        plt.savefig(output_path+"/Bezier_{}.jpg".format(str(counter)))
        cv2.imwrite(output_path+"/Bezier_input_{}.jpg".format(str(counter)),X[j])
        plt.clf()
        counter += 1


loss_function = None
optimizer = None
train_data = None
X = None
Y = None
del train_data
del X
del Y
del loss_function
del optimizer
