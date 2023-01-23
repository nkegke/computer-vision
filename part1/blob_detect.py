import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
from visualize_corners import *
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings("ignore")
from edge_detect import create_kernels


def BlobDetection(I, sigma, theta_blob, plot='no'):                        

    I = I.astype(np.float)/255
    
    #2.3.1
    gaussian = create_kernels(sigma)[0]
    filtered = cv2.filter2D(I, -1, gaussian)
    N = (np.ceil(3*sigma)*2+1).astype(int)
    
    Lxx = np.gradient(np.gradient(filtered)[0])[0]
    Lyy = np.gradient(np.gradient(filtered)[1])[1]  
    Lxy = np.gradient(np.gradient(filtered)[0])[1]  
    Det = Lxx*Lyy - Lxy*Lxy                         #det of Hessian table

    #visualizing
    # if plot=='plot':
#         sigm = np.around(sigma, decimals=1)
#         plt.title('Hessian, sigma='+str(sigm))
#         plt.imshow(Det,cmap='gray')
#         plt.show()

    #2.3.2
    B_sq = disk_strel(N)
    Cond1 = ( Det==cv2.dilate(Det,B_sq) )           #condition for maximum value in an area of B
    Cond2 = ( Det>theta_blob*(Det.max()))           #condition for higher value than a percentage of max
    blobs = (Cond1&Cond2).astype(int)               #area is blob if both conditions are true
    total_blobs = []
    for i in range(len(blobs)):
        for j in range(len(blobs[i])):              #creating a list of lists
            if blobs[i][j]==1:                      #with coordinates of center of blobs
                total_blobs.append([j,i,sigma])     #and scale sigma

    return Det, np.array(total_blobs)




def MultiscaleBlob(I, N, s, sigma_0, theta_blob, plot='no'):

    I = I.astype(np.float)/255
    
    dif_scales = [sigma_0*(s**i) for i in range(N)]

    hess = []   #List with det of hessian table for every scale and for every pixel
    total_blobs = []   #List with all centers and scales of blobs
    for i in range(N):
        blob = BlobDetection(I, dif_scales[i], theta_blob, plot)
        hess.append(blob[0])
        total_blobs.extend((blob[1]).tolist())

    final_coordinates_table = np.zeros(len(total_blobs))    #shows which blobs to discard
    for i in range(len(total_blobs)):
        x = int(total_blobs[i][1])
        y = int(total_blobs[i][0])
        scale = total_blobs[i][2]
        if scale == dif_scales[0]:        #boundary case
            if np.abs(hess[0][x][y]) > np.abs(hess[1][x][y]): #Check if det of Hessian in first scale is higher than in second
                final_coordinates_table[i] = 1
        elif scale == dif_scales[N-1]:    #boundary case
            if np.abs(hess[N-1][x][y]) > np.abs(hess[N-2][x][y]): #Check if det of Hessian in last scale is higher than in second to last
                final_coordinates_table[i] = 1
        else:                             #middle case            Check if LoG in a middle scale is higher than a scale higher and a scale lower
            if (np.abs(hess[dif_scales.index(scale)][x][y]) > np.abs(hess[dif_scales.index(scale)-1][x][y]) ) and (np.abs(hess[dif_scales.index(scale)][x][y]) > np.abs(hess[dif_scales.index(scale)+1][x][y])):
                final_coordinates_table[i] = 1

    counter = 0 #discarding blobs
    for i in range(len(final_coordinates_table)):
        if final_coordinates_table[i] == 0:
            total_blobs.pop(i-counter)
            counter += 1
    
    return np.array(total_blobs)






#2.5 Speed up with box filters and integral images
def BoxFilters(I, sigma, theta_blob, plot='no'):
    I = I.astype(np.float)/255

    gauss2D = create_kernels(sigma)[0]
    Is = cv2.filter2D(I, -1, gauss2D)
    n = int(2*np.ceil(3*sigma)+1)

    #2.5.1
    integral = np.cumsum(np.cumsum(Is,1),0).astype(float)

    #2.5.2
    #Dxx
    #height of window
    height = (4*np.floor(n/6)+1).astype(int)
    #width of window
    width = (2*np.floor(n/6)+1).astype(int) 

    #(width x height) center window of filter
    #Find center of window
    jc = np.floor(width/2).astype(int)
    ic = np.floor(height/2).astype(int)

    #Pad top and left edges with zeros
    Ip = np.pad(integral, [[0,ic+1], [0,jc+1]], 'edge')
    #Pad right and bottom edges with edge values
    Ip = np.pad(Ip, [[ic+1,0], [jc+1,0]])

    #Creating 4 new arrays, each contains Sa, Sb, Sc, Sd of every Dxx window of every pixel
    Sa = np.roll(Ip, ic+1, 0)
    Sa = np.roll(Sa, jc+1, 1)

    Sb = np.roll(Ip, ic+1, 0)
    Sb = np.roll(Sb, -jc, 1)

    Sd = np.roll(Ip, -ic, 0)
    Sd = np.roll(Ip, jc+1, 1)

    Sc = np.roll(Ip, -ic, 0)
    Sc = np.roll(Ip, -jc, 1)

    #Computing the area of every window
    area = Sa + Sc - Sb - Sd;

    #Crop the padding
    #area = area[2*ic+1:area.shape[0]-1-2*ic, 2*jc+1:area.shape[1]-1-2*jc]
    area = area[ic+1:area.shape[0]-1-ic, jc+1:area.shape[1]-1-jc]

    #Pad x axis with zeros (center x-shifting by width and -width)
    area = np.pad(area, [[0, 0],[width, width]])
    smiddle = area.copy()
    #Shifting to compute white windows
    sleft = np.roll(area , [0, -width])
    sright = np.roll(area , [0, width])

    Dxx = sleft - 2.0*smiddle + sright
    #Unpad to get rid of out of bounds areas
    Dxx = Dxx[:, width:Dxx.shape[1]-width]




    #Dyy, same logic
    width = (4*np.floor(n/6)+1).astype(int)
    height = (2*np.floor(n/6)+1).astype(int)       

    jc = np.floor(width/2).astype(int)
    ic = np.floor(height/2).astype(int)

    Ip = np.pad(integral, [[0,ic+1], [0,jc+1]], 'edge')
    Ip = np.pad(Ip, [[ic+1,0], [jc+1,0]])

    Sa = np.roll(Ip, ic+1, 0)
    Sa = np.roll(Sa, jc+1, 1)

    Sb = np.roll(Ip, ic+1, 0)
    Sb = np.roll(Sb, -jc, 1)

    Sd = np.roll(Ip, -ic, 0)
    Sd = np.roll(Ip, jc+1, 1)

    Sc = np.roll(Ip, -ic, 0)
    Sc = np.roll(Ip, -jc, 1)
    area = Sa - Sb - Sd + Sc;

    area = area[ic+1:area.shape[0]-1-ic, jc+1:area.shape[1]-1-jc]
    #Pad y axis with zeros (center y-shifting by height and -height)
    area = np.pad(area, [[height, height],[0,0]])

    smiddle = area.copy()
    sup = np.roll(area , [-height,0])
    sdown = np.roll(area , [height,0])

    Dyy = sup - 2.0*smiddle + sdown
    Dyy = Dyy[height:Dyy.shape[0]-height,:]




    #Dxy, minor differences
    width = (2*np.floor(n/6)+1).astype(int)
    height = (2*np.floor(n/6)+1).astype(int)

    jc = np.floor(width/2).astype(int)
    ic = np.floor(height/2).astype(int)

    Ip = np.pad(integral, [[0,ic+1], [0,jc+1]], 'edge')
    Ip = np.pad(Ip, [[ic+1,0], [jc+1,0]])

    Sa = np.roll(Ip, ic+1, 0)
    Sa = np.roll(Sa, jc+1, 1)

    Sb = np.roll(Ip, ic+1, 0)
    Sb = np.roll(Sb, -jc, 1)

    Sd = np.roll(Ip, -ic, 0)
    Sd = np.roll(Ip, jc+1, 1)

    Sc = np.roll(Ip, -ic, 0)
    Sc = np.roll(Ip, -jc, 1)
    area = Sa - Sb - Sd + Sc;

    area = area[ic+1:area.shape[0]-1-ic, jc+1:area.shape[1]-1-jc]
    #Pad x and y axes with zeros (center x-shifting by -jc-1 and jc+1, y-shifting by -ic-1 and ic+1)
    area = np.pad(area, [ic+1, jc+1])

    #centers of each of Dxy filter 4 windows
    sleft_up = np.roll(area, [-ic-1,-jc-1])
    sright_down = np.roll(area, [ic+1,jc+1])
    sright_up = np.roll(area , [ic+1,-jc-1])
    sleft_down = np.roll(area , [-ic-1,jc+1])

    Dxy = sleft_up + sright_down - sright_up - sleft_down
    Dxy = Dxy[ic+1:Dxy.shape[0]-1-ic, jc+1 : Dxy.shape[1]-1-jc]

    #2.5.3
    R = (Dxx*Dyy - (0.9*Dxy)**2)
    # if plot=='plot':
#         sigm = np.around(sigma, decimals=1)
#         plt.title('Hessian approx, sigma='+str(sigm))
#         plt.imshow(R,cmap='gray')
#         plt.show()
    B_sq = disk_strel(n)
    Cond1 = ( R==cv2.dilate(R,B_sq) )
    Cond2 = ( R>theta_blob*(R.max()))
    blobs = (Cond1&Cond2).astype(int)
    total_blobs = []
    for i in range(len(blobs)):
        for j in range(len(blobs[i])):
            if blobs[i][j]==1:
                total_blobs.append([j,i,sigma])
    return R, np.array(total_blobs)



#2.5.4
def MultiscaleBoxFilt(I, N, s, sigma_0, theta_blob, plot='no'):

    I = I.astype(np.float)/255
    
    dif_scales = [sigma_0*(s**i) for i in range(N)]

    hess = []   #List with det of hessian table for every scale and for every pixel
    total_blobs = []   #List with all centers and scales of blobs
    for i in range(N):
        box = BoxFilters(I, dif_scales[i], theta_blob, plot)
        hess.append(box[0])
        total_blobs.extend((box[1]).tolist())


    final_coordinates_table = np.zeros(len(total_blobs))    #shows which blobs to discard
    for i in range(len(total_blobs)):
        x = int(total_blobs[i][1])
        y = int(total_blobs[i][0])
        scale = total_blobs[i][2]
        if scale == dif_scales[0]:        #boundary case
            if np.abs(hess[0][x][y]) > np.abs(hess[1][x][y]): #Check if det of Hessian in first scale is higher than in second
                final_coordinates_table[i] = 1
        elif scale == dif_scales[N-1]:    #boundary case
            if np.abs(hess[N-1][x][y]) > np.abs(hess[N-2][x][y]): #Check if det of Hessian in last scale is higher than in second to last
                final_coordinates_table[i] = 1
        else:                             #middle case            Check if LoG in a middle scale is higher than a scale higher and a scale lower
            if (np.abs(hess[dif_scales.index(scale)][x][y]) > np.abs(hess[dif_scales.index(scale)-1][x][y]) ) and (np.abs(hess[dif_scales.index(scale)][x][y]) > np.abs(hess[dif_scales.index(scale)+1][x][y])):
                final_coordinates_table[i] = 1

    counter = 0 #discarding blobs
    for i in range(len(final_coordinates_table)):
        if final_coordinates_table[i] == 0:
            total_blobs.pop(i-counter)
            counter += 1
    return np.array(total_blobs)



if __name__='main':

	img_gray = cv2.imread(IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
	img = cv2.imread(IMAGE_NAME)
	img_colored = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	sigma, theta_blob = 2.5, 0.02
	img_coords = BlobDetection(img_gray, sigma, theta_blob)[1]
	interest_points_visualization(img_colored, img_coords).set_title('Blob Detection, sigma=2.5, theta_blob=0.02')
	plt.show()

	N, s, sigma_0, theta_blob = 4, 1.7, 2, 0.02
	total_coordinates = MultiscaleBlob(img_gray, N, s, sigma_0, theta_blob, 'plot')
	interest_points_visualization(img_colored,total_coordinates).set_title('Multiscale Blob Detection, N=4, sigma_0=2.5, s=1.7, theta_blob=0.02')
	plt.show()

	sigma, theta_blob = 1.5, 0.005
	img_coords = BoxFilters(img_gray, sigma, theta_blob, [])[1]
	interest_points_visualization(img_colored, img_coords)
	plt.show()

	N, s, sigma_0, theta_blob = 4, 1.7, 2, 0.003
	total_coordinates = MultiscaleBoxFilt(img_gray, N, s, sigma_0, theta_blob, 'plot')
	interest_points_visualization(img_colored,total_coordinates).set_title('Speeded up Multiscale Blob Detection N=4, sigma_0=2, s=1.7, theta_blob=0.003')
	plt.show()
	