import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from skimage import color
import scipy
import scipy.io as sp
import os
from displ import displ
from drawBoundingBox import drawBoundingBox
from lk import lk
from multi_scale_lk import multi_scale_lk
from regionDetect import regionDetect

if not os.path.exists('plots/'):
	os.mkdir('plots/')

#Gaussian distribution training
skin_SamplesRGB = sp.loadmat('skinSamplesRGB.mat')
skin_SamplesRGB = skin_SamplesRGB['skinSamplesRGB']
plt.figure()
plt.imshow(skin_SamplesRGB)
plt.title('Skin samples')
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.savefig('plots/1.png')

#Convert to YCbCr
skin_SamplesYCbCr = color.rgb2ycbcr(skin_SamplesRGB)

#Flatten the images
skin_SamplesYCbCr = skin_SamplesYCbCr.reshape(1782,3)

#Keep only Cb, Cr channels
skin_SamplesCb = skin_SamplesYCbCr[:,1]
skin_SamplesCr = skin_SamplesYCbCr[:,2]

#Mean Cb, Cr values
Cb_mean = np.mean(skin_SamplesCb)
Cr_mean = np.mean(skin_SamplesCr)

#Covariance matrix
Cov_matrix = np.cov(np.array([skin_SamplesCb,skin_SamplesCr]))

#Plotting 2D Gaussian distribution surface
x = np.linspace(Cb_mean - Cov_matrix[0][0], Cb_mean + Cov_matrix[0][0], 100)
y = np.linspace(Cr_mean - Cov_matrix[1][1], Cr_mean + Cov_matrix[1][1], 100)
x, y = np.meshgrid(x, y)
x_ = x.flatten()
y_ = y.flatten()
xy = np.vstack((x_, y_)).T
rv = scipy.stats.multivariate_normal(np.array([Cb_mean, Cr_mean]), Cov_matrix)
z = rv.pdf(xy)
z = z.reshape(100, 100, order='F')
plt.figure()
plt.title('Gaussian distribution surface')
plt.contourf(x, y, z.T)
plt.savefig('plots/2.png')

#Detect first frame regions
I = cv2.imread(os.path.join('material','1.png'))
BoundingBoxes = regionDetect(I, np.array([Cb_mean, Cr_mean]), Cov_matrix)


I = cv2.imread(os.path.join('material','1.png'))
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
I1copy = I.copy()

for i in range(len(BoundingBoxes)):
    regions = ['head', 'left', 'right']
    drawBoundingBox(I,BoundingBoxes[i][0],BoundingBoxes[i][1],BoundingBoxes[i][2],BoundingBoxes[i][3],regions[i])
    
plt.figure()
plt.imshow(I)
plt.title('Head and Hands Detection')
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.savefig('plots/5.png')
plt.title('Frame 1')
plt.savefig('plots/single_scale_tracking/1.png')
plt.title('Frame 1')
plt.savefig('plots/multi_scale_tracking/1.png')



#Lucas-Kanade
I1 = cv2.imread(os.path.join('material','1.png'))
I1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
I1 = (I1/255).astype(np.float32)

Iprev = I1.copy()
Idraw_prev = I.copy()

x_head = BoundingBoxes[0][0]
y_head = BoundingBoxes[0][1]
head_width = BoundingBoxes[0][2]
head_height = BoundingBoxes[0][3]
x_left = BoundingBoxes[1][0]
y_left = BoundingBoxes[1][1]
left_width = BoundingBoxes[1][2]
left_height = BoundingBoxes[1][3]
x_right = BoundingBoxes[2][0]
y_right = BoundingBoxes[2][1]
right_width = BoundingBoxes[2][2]
right_height = BoundingBoxes[2][3]

#crop head,left,right
head_prev = Iprev[y_head:(y_head+head_height),x_head:(x_head+head_width)]
left_prev = Iprev[y_left:(y_left+left_height),x_left:(x_left+left_width)]
right_prev = Iprev[y_right:(y_right+right_height),x_right:(x_right+right_width)]

for frame_num in range(2,30):
    #read frame
    Inext = cv2.imread(os.path.join('material','{}.png'.format(frame_num)))
    Inext = cv2.cvtColor(Inext,cv2.COLOR_BGR2GRAY)
    Inext = (Inext/255).astype(np.float32)

    #crop head,left,right
    head_next = Inext[y_head:(y_head+head_height),x_head:(x_head+head_width)]
    left_next = Inext[y_left:(y_left+left_height),x_left:(x_left+left_width)]
    right_next = Inext[y_right:(y_right+right_height),x_right:(x_right+right_width)]


    #calculate optical flow

    #points to track optical flow
    features_head = cv2.goodFeaturesToTrack(head_next, 20, 0.1, 2)
    features_head = features_head.reshape(features_head.shape[0],2)
    features_left = cv2.goodFeaturesToTrack(left_next, 20, 0.1, 2)
    features_left = features_left.reshape(features_left.shape[0],2)
    features_right = cv2.goodFeaturesToTrack(right_next, 20, 0.1, 2)
    features_right = features_right.reshape(features_right.shape[0],2)

    #initial estimation
    d_x0_head = np.zeros((head_prev.shape[0],head_prev.shape[1]))
    d_y0_head = np.zeros((head_prev.shape[0],head_prev.shape[1]))
    d_x0_left = np.zeros((left_prev.shape[0],left_prev.shape[1]))
    d_y0_left = np.zeros((left_prev.shape[0],left_prev.shape[1]))
    d_x0_right = np.zeros((right_prev.shape[0],right_prev.shape[1]))
    d_y0_right = np.zeros((right_prev.shape[0],right_prev.shape[1]))
    rho = 5
    epsilon = 0.01

    dhead = lk(head_prev, head_next, features_head, rho, epsilon, d_x0_head, d_y0_head)
    dleft = lk(left_prev, left_next, features_left, rho, epsilon, d_x0_left, d_y0_left)
    dright = lk(right_prev, right_next, features_right, rho, epsilon, d_x0_right, d_y0_right)

    #displ
    [head_mean_dx, head_mean_dy] = displ(dhead[0], dhead[1])
    [left_mean_dx, left_mean_dy] = displ(dleft[0], dleft[1])
    [right_mean_dx, right_mean_dy] = displ(dright[0], dright[1])
    

    #update bounding boxes
    x_head = x_head - head_mean_dx
    y_head = y_head - head_mean_dy
    x_left = x_left - left_mean_dx
    y_left = y_left - left_mean_dy
    x_right = x_right - right_mean_dx
    y_right = y_right - right_mean_dy

    #print new frame with moved bounding box
    Idraw = cv2.imread(os.path.join('material','{}.png'.format(frame_num)))
    Idraw = cv2.cvtColor(Idraw, cv2.COLOR_BGR2RGB)

    drawBoundingBox(Idraw,x_head,y_head,head_width,head_height,'head')
    drawBoundingBox(Idraw,x_left,y_left,left_width,left_height,'left')
    drawBoundingBox(Idraw,x_right,y_right,right_width,right_height,'right')
    
    plt.figure()
    plt.imshow(Idraw)
    plt.title('Frame {}'.format(frame_num))
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig('plots/single_scale_tracking/{}.png'.format(frame_num))
		
	#for plots
    plt.figure()
    fig, axs = plt.subplots(1,5, figsize=(30,9))
    axs[0].imshow(Idraw_prev)
    axs[0].set_title('Frame {}'.format(frame_num-1))
    axs[1].quiver(-dhead[0],dhead[1],angles='xy',scale=100)
    axs[1].set_title('Head optical flow')
    axs[2].quiver(-dleft[0],dleft[1],angles='xy',scale=100)
    axs[2].set_title('Left hand optical flow')
    axs[3].quiver(-dright[0],dright[1],angles='xy',scale=100)
    axs[3].set_title('Right hand optical flow')
    axs[4].imshow(Idraw)
    axs[4].set_title('Frame {}'.format(frame_num))
    plt.savefig('plots/single_scale_quivers/{}.png'.format(frame_num))
    #end of plots
    
    
    Iprev = Inext.copy()
    Idraw_prev = Idraw.copy()
    head_prev = head_next.copy()
    left_prev = left_next.copy()
    right_prev = right_next.copy()


#Multi-scale Lucas-Kanade

I1 = cv2.imread(os.path.join('material','1.png'))
I1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
I1 = (I1/255).astype(np.float32)
Iprev = I1.copy()
Idraw_prev = I.copy()

x_head = BoundingBoxes[0][0]
y_head = BoundingBoxes[0][1]
head_width = BoundingBoxes[0][2]
head_height = BoundingBoxes[0][3]
x_left = BoundingBoxes[1][0]
y_left = BoundingBoxes[1][1]
left_width = BoundingBoxes[1][2]
left_height = BoundingBoxes[1][3]
x_right = BoundingBoxes[2][0]
y_right = BoundingBoxes[2][1]
right_width = BoundingBoxes[2][2]
right_height = BoundingBoxes[2][3]

#crop head,left,right
head_prev = Iprev[y_head:(y_head+head_height),x_head:(x_head+head_width)]
left_prev = Iprev[y_left:(y_left+left_height),x_left:(x_left+left_width)]
right_prev = Iprev[y_right:(y_right+right_height),x_right:(x_right+right_width)]

for frame_num in range(2,30):

    #read frame
    Inext = cv2.imread(os.path.join('material','{}.png'.format(frame_num)))
    Inext = cv2.cvtColor(Inext,cv2.COLOR_BGR2GRAY)
    Inext = (Inext/255).astype(np.float32)

    #crop head,left,right
    head_next = Inext[y_head:(y_head+head_height),x_head:(x_head+head_width)]
    left_next = Inext[y_left:(y_left+left_height),x_left:(x_left+left_width)]
    right_next = Inext[y_right:(y_right+right_height),x_right:(x_right+right_width)]


    #calculate optical flow

    #points to track optical flow
    features_head = cv2.goodFeaturesToTrack(head_next, 20, 0.1, 2)
    features_head = features_head.reshape(features_head.shape[0],2)
    features_left = cv2.goodFeaturesToTrack(left_next, 20, 0.1, 2)
    features_left = features_left.reshape(features_left.shape[0],2)
    features_right = cv2.goodFeaturesToTrack(right_next, 20, 0.1, 2)
    features_right = features_right.reshape(features_right.shape[0],2)

    #initial estimation
    d_x0_head = np.zeros((head_prev.shape[0],head_prev.shape[1]))
    d_y0_head = np.zeros((head_prev.shape[0],head_prev.shape[1]))
    d_x0_left = np.zeros((left_prev.shape[0],left_prev.shape[1]))
    d_y0_left = np.zeros((left_prev.shape[0],left_prev.shape[1]))
    d_x0_right = np.zeros((right_prev.shape[0],right_prev.shape[1]))
    d_y0_right = np.zeros((right_prev.shape[0],right_prev.shape[1]))
    rho = 5
    epsilon = 0.01

    dhead = multi_scale_lk(head_prev, head_next, features_head, rho, epsilon, d_x0_head, d_y0_head, 3)
    dleft = multi_scale_lk(left_prev, left_next, features_left, rho, epsilon, d_x0_left, d_y0_left, 3)
    dright = multi_scale_lk(right_prev, right_next, features_right, rho, epsilon, d_x0_right, d_y0_right, 3)

    #displ
    [head_mean_dx, head_mean_dy] = displ(dhead[0], dhead[1])
    [left_mean_dx, left_mean_dy] = displ(dleft[0], dleft[1])
    left_mean_dx = int(np.ceil(left_mean_dx * 1.65))
    left_mean_dy = int(np.ceil(left_mean_dy * 1.6))
    [right_mean_dx, right_mean_dy] = displ(dright[0], dright[1])
    right_mean_dy = int(np.ceil(right_mean_dy * 1.4))

    #update bounding boxes
    x_head = x_head - head_mean_dx + 1
    y_head = y_head - head_mean_dy
    x_left = x_left - left_mean_dx + 2
    y_left = y_left - left_mean_dy
    x_right = x_right - right_mean_dx
    y_right = y_right - right_mean_dy

    #print new frame with moved bounding box
    Idraw = cv2.imread(os.path.join('material','{}.png'.format(frame_num)))
    Idraw = cv2.cvtColor(Idraw, cv2.COLOR_BGR2RGB)

    drawBoundingBox(Idraw,x_head,y_head,head_width,head_height,'head')
    drawBoundingBox(Idraw,x_left,y_left,left_width,left_height,'left')
    drawBoundingBox(Idraw,x_right,y_right,right_width,right_height,'right')
    
    plt.figure()
    plt.imshow(Idraw)
    plt.title('Frame {}'.format(frame_num))
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig('plots/multi_scale_tracking/{}.png'.format(frame_num))
    
    #for plots
    plt.figure()
    fig, axs = plt.subplots(1,5, figsize=(30,9))
    axs[0].imshow(Idraw_prev)
    axs[0].set_title('Frame {}'.format(frame_num-1))
    axs[1].quiver(-dhead[0],dhead[1],angles='xy',scale=100)
    axs[1].set_title('Head optical flow')
    axs[2].quiver(-dleft[0],dleft[1],angles='xy',scale=100)
    axs[2].set_title('Left hand optical flow')
    axs[3].quiver(-dright[0],dright[1],angles='xy',scale=100)
    axs[3].set_title('Right hand optical flow')
    axs[4].imshow(Idraw)
    axs[4].set_title('Frame {}'.format(frame_num))
    plt.savefig('plots/multi_scale_quivers/{}.png'.format(frame_num))
    #end of plots
    
    
    Iprev = Inext.copy()
    Idraw_prev = Idraw.copy()
    head_prev = head_next.copy()
    left_prev = left_next.copy()
    right_prev = right_next.copy()
