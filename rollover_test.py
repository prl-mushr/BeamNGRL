# file for processing data from the rollover test
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# load numpy array from folder data. file name is episode_{episode_number}.npy
from sklearn.metrics import confusion_matrix

size = 10
ground_truth = []
prediction = []

def metrics(prediction, ground_truth, folder_name):
    for i in range(200):
        # load data from episode i:
        episode = np.load(str(folder_name+"/episode_{}.npy".format(i+1)))
        # first 3 elements are position, next 3 are rpy, next 3 are velocity, next 3 are acceleration, next 3 are gyro, next 1 is time, next 1 is dt, next 3 are raw acceleration, next 1 is start turning,
        pos = episode[:,0:3]
        rpy = episode[:,3:6]*57.3
        vel = episode[:,6:9]
        acc = episode[:,9:12]
        gyro = episode[:,12:15]
        t = episode[:,15]
        dt = episode[:,16]
        raw_acc = episode[:,17:20]
        start_turning = episode[:,20]

        critical_angle = 60/57.3 # is in degrees
        def butter_lowpass_filter(data, cutoff, fs, order):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            # Get the filter coefficients 
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data)
            return y

        # calculate force angle margin based on body frame acceleration and critical angle
        # force angle margin is defined as the tan of the ratio of the acceleration in y direction to the acceleration in z direction
        # this is the angle between the force vector and the vertical axis
        # if the force angle margin is greater than the critical angle, the vehicle will roll over
        FAM = np.arctan2(acc[:,1], np.fabs(acc[:,2]))
        FAM = np.fabs(FAM)

        # FAM indicator is 1 when FAM is greater than the critical angle and 0 otherwise:
        FAM_indicator = -np.ones_like(FAM)
        FAM_indicator[np.where(FAM > critical_angle - rpy[:,0]/57.3)] = 1
        ground_truth.append(np.any(rpy[:,0] > 120))
        prediction.append(np.any(FAM_indicator == 1))
        try:
            critical_index = np.where(np.fabs(rpy[:,0]) > 10)[0][0]
        except:
            critical_index = len(rpy) -1 
        t_critical = t[critical_index]
        TTR_gt = t_critical - t
        # apply a 10 Hz IIR filter to FAM
        # FAM = butter_lowpass_filter(FAM, 5, 40, 1)
        # find the first derivative of Force Angle Margin
        dFAM = np.gradient(FAM, 1/50)
        dFAM = butter_lowpass_filter(dFAM, 10, 40, 2)
        # find the second derivative of Force Angle Margin
        ddFAM = np.gradient(dFAM, dt)

        def calculate_ttr(FAM, dFAM, G, critical_angle, t):
            margin = critical_angle - FAM
            # denominator is the max of the absolute value of dFAM and the absolute value of the first column of G:
            denominator = np.maximum(np.fabs(dFAM), np.fabs(G[:,0]))
            T = margin/dFAM
            T = np.minimum(T, 2)
            T = np.maximum(T, -2)
            # filter T:
            T = butter_lowpass_filter(T, 10, 40, 2)
            return T
            
        TTR = calculate_ttr(FAM, dFAM, gyro, critical_angle, t)
        # find the time when Force Angle Margin is greater than the critical angle
        # convert rpy to degrees from radians by multiplying by 57.3
        # find first index where roll is greater than 60 degrees
        try:
            index_stop = np.where(rpy[:,0] > 60)[0][0]
        except:
            index_stop = len(rpy) -1
        # find first index where time is greater than 5 seconds
        index_start = np.where(start_turning == 1)[0][0]
        # plot TTR, FAM, vs time from 5 seconds to 60 degrees with lables and legends:
        # plt.figure()
        # plt.plot(t[index_start:index_stop], TTR_gt[index_start:index_stop], label="TTR_gt")
        # plt.plot(t[index_start:index_stop], TTR[index_start:index_stop], label="TTR")
        # plt.plot(t[index_start:index_stop], np.zeros_like(t[index_start:index_stop]) )
        # # plt.plot(t[index_start:index_stop], FAM[index_start:index_stop], label="FAM")
        # plt.plot(t[index_start:index_stop], rpy[index_start:index_stop,0]/57.3, label="Roll")
        # #plot FAM_indicator:
        # plt.plot(t[index_start:index_stop], FAM_indicator[index_start:index_stop], label="FAM_indicator")
        # plt.plot(t[index_start:index_stop], rpy[index_start:index_stop,1], label="pitch")

        # plot accelerations:
        # plt.plot(t[index_start:index_stop], acc[index_start:index_stop,0], label="Acc_x")
        # plt.plot(t[index_start:index_stop], acc[index_start:index_stop,1], label="Acc_y")
        # plt.plot(t[index_start:index_stop], acc[index_start:index_stop,2], label="Acc_z")
        # plot velocity:
        # plt.plot(t[index_start:index_stop], vel[index_start:index_stop,0], label="Vel_x")
        # plt.plot(t[index_start:index_stop], vel[index_start:index_stop,1], label="Vel_y")
        # plt.plot(t[index_start:index_stop], vel[index_start:index_stop,2], label="Vel_z")


        # plt.xlabel("Time (s)")
        # plt.ylabel("Angle (deg)")
        # plt.legend()
        # plt.show()

prediction = []
ground_truth = []
metrics(prediction, ground_truth,"untripped_flat_with_correction")
p,g = prediction, ground_truth
print(np.array(g).mean())
# cm = confusion_matrix(np.array(g), np.array(p))
# # print interpreted confusion matrix:
# print("True Positive: ", cm[1,1])
# print("False Positive: ", cm[0,1])
# print("True Negative: ", cm[0,0])
# print("False Negative: ", cm[1,0])
# print("Accuracy: ", (cm[1,1] + cm[0,0])/np.sum(cm))

prediction = []
ground_truth = []
metrics(prediction, ground_truth,"tripped_flat_with_correction")
p,g = prediction, ground_truth
print(np.array(g).mean())
# cm = confusion_matrix(np.array(g), np.array(p))
# # print interpreted confusion matrix:
# print("True Positive: ", cm[1,1])
# print("False Positive: ", cm[0,1])
# print("True Negative: ", cm[0,0])
# print("False Negative: ", cm[1,0])
# print("Accuracy: ", (cm[1,1] + cm[0,0])/np.sum(cm))

prediction = []
ground_truth = []
metrics(prediction, ground_truth,"mixed_offroad_with_correction")
p,g = prediction, ground_truth
print(np.mean(np.array(g)))
# cm = confusion_matrix(np.array(g), np.array(p))
# # print interpreted confusion matrix:
# print("True Positive: ", cm[1,1])
# print("False Positive: ", cm[0,1])
# print("True Negative: ", cm[0,0])
# print("False Negative: ", cm[1,0])
# print("Accuracy: ", (cm[1,1] + cm[0,0])/np.sum(cm))

# print overall metrics:

# print(ground_truth)
# print(prediction)
# plot confusion matrix:
# cm = confusion_matrix(np.array(ground_truth), np.array(prediction))
# # print interpreted confusion matrix:
# print("True Positive: ", cm[1,1])
# print("False Positive: ", cm[0,1])
# print("True Negative: ", cm[0,0])
# print("False Negative: ", cm[1,0])
# print("Accuracy: ", (cm[1,1] + cm[0,0])/np.sum(cm))
# print("Precision: ", cm[1,1]/(cm[1,1] + cm[0,1]))
# print("Recall: ", cm[1,1]/(cm[1,1] + cm[1,0]))
# print("F1 Score: ", 2*cm[1,1]/(2*cm[1,1] + cm[0,1] + cm[1,0]))
# print("Specificity: ", cm[0,0]/(cm[0,0] + cm[0,1]))
# print("False Positive Rate: ", cm[0,1]/(cm[0,1] + cm[0,0]))
# print("False Negative Rate: ", cm[1,0]/(cm[1,0] + cm[1,1]))
# print("True Negative Rate: ", cm[0,0]/(cm[0,0] + cm[0,1]))
# print("True Positive Rate: ", cm[1,1]/(cm[1,1] + cm[1,0]))
# print("False Discovery Rate: ", cm[0,1]/(cm[0,1] + cm[1,1]))
# print("False Omission Rate: ", cm[1,0]/(cm[1,0] + cm[0,0]))
# print("Negative Predictive Value: ", cm[0,0]/(cm[0,0] + cm[1,0]))
# print("Prevalence: ", (cm[1,1] + cm[1,0])/np.sum(cm))
# print("Detection Rate: ", cm[1,1]/(cm[1,1] + cm[1,0]))
# print("Detection Prevalence: ", cm[1,1]/np.sum(cm))
# print("Balanced Accuracy: ", (cm[1,1]/(cm[1,1] + cm[1,0]) + cm[0,0]/(cm[0,0] + cm[0,1]))/2)






