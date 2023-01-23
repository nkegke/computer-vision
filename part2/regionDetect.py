import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats
from pointDetection import pointDetection

def disk_strel(n):
    #Return a structural element, which is a disk of radius n.
    r = int(np.round(n))
    d = 2*r+1
    x = np.arange(d) - r
    y = np.arange(d) - r
    x, y = np.meshgrid(x,y)
    strel = x**2 + y**2 <= r**2
    return strel.astype(np.uint8)

def regionDetect(I, mu, cov):
    #Read image
    IYCrCb = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)

    #Keep Cb,Cr channels
    ICbCr = IYCrCb[:,:,::-1]
    ICbCr = ICbCr[:,:,:2]

    #Compute skin pdf
    skin_probability = stats.multivariate_normal.pdf(ICbCr, mean=mu, cov=cov)
    #Regularization
    skin_probability /= skin_probability.max()

    #Threshold
    threshold = 0.08
    skin = (skin_probability >= threshold).astype(np.uint8)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Skin probability')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(skin_probability, cmap='gray')
    plt.subplot(1,2,2)
    plt.title('Binary skin detection image')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(skin,cmap='gray')
    plt.savefig('plots/3.png')
	
	
    opening_kernel = disk_strel(2)
    Opened_skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, opening_kernel)
    closing_kernel = disk_strel(30)
    Closed_skin = cv2.morphologyEx(Opened_skin, cv2.MORPH_CLOSE, closing_kernel)
    plt.figure()
    plt.subplot(1,3,1)
    plt.title('Before filtering')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(skin, cmap='gray')
    plt.subplot(1,3,2)
    plt.title('After Opening')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(Opened_skin,cmap='gray')
    plt.subplot(1,3,3)
    plt.title('After Closing')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(Closed_skin,cmap='gray')
    plt.savefig('plots/4.png')
    
    return pointDetection(Closed_skin)