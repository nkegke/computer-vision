import numpy as np
import cv2
from lk import lk

#reduce function 
def GREDUCE(I,h):
    Y_filtered = cv2.filter2D(I,-1,h) #convolve image with kernel h 
    downsampled = cv2.resize(Y_filtered, (Y_filtered.shape[1]//2, Y_filtered.shape[0]//2), interpolation=cv2.INTER_AREA)#downsampling by keeping only even rows and columns
    return downsampled
    

#gaussian pyramid with depth levels (excluding image)
def GPyramid(I,sigma,depth):
    L = [I]
    n = int(2*np.ceil(3*sigma)+1)
    gauss1D = cv2.getGaussianKernel(n, sigma)
    h = gauss1D @ gauss1D.T 
    for i in range(depth):
        t = GREDUCE(L[-1],h)
        L.append(t)
    return L


#multi-scale LK
def multi_scale_lk(I1, I2, features, rho, epsilon, d_x0, d_y0, scales):

    #construct pyramid with 3 levels above original image
    pyramid1 = GPyramid(I1,3,scales)
    pyramid2 = GPyramid(I2,3,scales)
    
    #new size of optical flows (for lower scale)
    d_xL_exp = np.zeros(pyramid1[-1].shape)
    d_yL_exp = np.zeros(pyramid1[-1].shape)

    #go in higher scales
    for i in reversed(range(1,scales+1)):

        #find features in new scale
        features_L = cv2.goodFeaturesToTrack(pyramid2[i], 20, 0.1, 2)
        features_L = features_L.reshape(features_L.shape[0],2)

        #find optical flow in current scale 
        [d_xL, d_yL] = lk(pyramid1[i], pyramid2[i], features_L, rho, epsilon, d_xL_exp, d_yL_exp)

        #renew indices of points of interest
        [new_d_xL_y, new_d_xL_x] = np.where(d_xL != 0)
        new_d_xL_x = new_d_xL_x *2
        new_d_xL_y = new_d_xL_y *2
        [new_d_yL_y, new_d_yL_x] = np.where(d_yL != 0)
        new_d_yL_x = new_d_yL_x *2
        new_d_yL_y = new_d_yL_y *2

        #double size of optical flow's table
        d_xL_exp = np.zeros(pyramid1[i-1].shape)
        d_yL_exp = np.zeros(pyramid1[i-1].shape)

        #double optical flow values
        for j in range(features_L.shape[0]):
            d_xL_exp[int(new_d_xL_y[j])][int(new_d_xL_x[j])] = d_xL_exp[int(new_d_xL_y[j])][int(new_d_xL_x[j])] + d_xL[ int(features_L[j][1]) ][ int(features_L[j][0]) ]*2
            d_yL_exp[int(new_d_yL_y[j])][int(new_d_yL_x[j])] = d_yL_exp[int(new_d_yL_y[j])][int(new_d_yL_x[j])] + d_yL[ int(features_L[j][1]) ][ int(features_L[j][0]) ]*2

    return d_xL_exp, d_yL_exp
