import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
from cv21_lab1_part2_utils import interest_points_visualization, disk_strel
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings("ignore")
from edge_detect import create_kernels


def HarrisStephens(I, sigma, rho, k, theta_corn, plot='no'):
    I = I.astype(np.float)/255

    Gr = create_kernels(rho)[0]
    Gs = create_kernels(sigma)[0]

    Is = cv2.filter2D(I, -1, Gs)               #Is = Gs * I

    Isgradx = np.gradient(Is)[0] 
    Isgrady = np.gradient(Is)[1]

    J1 = cv2.filter2D(Isgradx*Isgradx, -1, Gr) #J1 = Gr * (dIs/dx . dIs/dx)
    J2 = cv2.filter2D(Isgradx*Isgrady, -1, Gr) #J2 = Gr * (dIs/dx . dIs/dy)
    J3 = cv2.filter2D(Isgrady*Isgrady, -1, Gr) #J1 = Gr * (dIs/dy . dIs/dy)

    lambda_plus = 0.5*(J1 + J3 + np.sqrt((J1-J3)**2+4*(J2**2)))  #eigenvalues of J
    lambda_minus = 0.5*(J1 + J3 - np.sqrt((J1-J3)**2+4*(J2**2))) #needed for cornerness criterion
    if plot=='plot':
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(lambda_plus, cmap = 'gray')
        axs[0].set_title('lambda+')
        axs[1].imshow(lambda_minus, cmap = 'gray')
        axs[1].set_title('lambda-')
        plt.show()

    R = lambda_minus*lambda_plus - k*((lambda_minus+lambda_plus)**2) #cornerness criterion
    ns = np.ceil(3*sigma)*2+1
    B_sq = disk_strel(ns)
    Cond1 = ( R==cv2.dilate(R,B_sq) )   #condition for maximum pixel in an area of B
    Cond2 = ( R>theta_corn*(R.max()) )  #condition for higher value than a percentage of max
    corners = (Cond1&Cond2).astype(int) #area is corner if both conditions are true
    total_coordinates = []
    for i in range(len(corners)):
        for j in range(len(corners[i])):                #creating a list of lists
            if corners[i][j]==1:                        #with coordinates of center of corners
                total_coordinates.append([j,i,sigma])   #and scale sigma

    return np.array(total_coordinates)


def MultiscaleHarris(I, N, s, sigma_0, rho_0, k, theta_corn):

    I = I.astype(np.float)/255
    
    dif_scales = [sigma_0*(s**i) for i in range(N)]
    integ_scales = [rho_0*(s**i) for i in range(N)]

    total_coordinates = []
    for i in range(N):
    	total_coordinates.extend(HarrisStephens(I, dif_scales[i], integ_scales[i], k, theta_corn).tolist())
    
    LoGs = []
    for i in range(N):
        LoGs.append((dif_scales[i]**2)*cv2.filter2D(I, -1, create_kernels(i)[1]))

    final_coordinates_table = np.zeros(len(total_coordinates)) #shows which corners to discard
    for i in range(len(total_coordinates)):
        x = int(total_coordinates[i][1])
        y = int(total_coordinates[i][0])
        scale = total_coordinates[i][2]
        if scale == dif_scales[0]:        #left boundary case
            if np.abs(LoGs[0][x][y]) > np.abs(LoGs[1][x][y]):     #Check if LoG in first scale is higher than in second
                final_coordinates_table[i] = 1
        elif scale == dif_scales[N-1]:    #right boundary case
            if np.abs(LoGs[N-1][x][y]) > np.abs(LoGs[N-2][x][y]): #Check if LoG in last scale is higher than in second to last
                final_coordinates_table[i] = 1
        else:                             #middle case             Check if LoG in a middle scale is higher than a scale higher and a scale lower
            if (np.abs(LoGs[dif_scales.index(scale)][x][y]) > np.abs(LoGs[dif_scales.index(scale)-1][x][y])) and (np.abs(LoGs[dif_scales.index(scale)][x][y]) > np.abs(LoGs[dif_scales.index(scale)+1][x][y])):
                final_coordinates_table[i] = 1
    counter = 0
    for i in range(len(final_coordinates_table)):
        if final_coordinates_table[i] == 0:
            total_coordinates.pop(i-counter)
            counter += 1
    return np.array(total_coordinates)
    

if __name__='main':
	Urban_gray = cv2.imread(IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
	Urban = cv2.imread(IMAGE_NAME)
	Urban_colored = cv2.cvtColor(Urban, cv2.COLOR_BGR2RGB)
	coordinates = HarrisStephens(Urban_gray, 2, 2.5, 0.05, 0.005,'plot')
	interest_points_visualization(Urban_colored,coordinates).set_title('Harris Corner Detection')
	plt.show()

	N, s, sigma_0, rho_0, k, theta_corn = 4, 1.5, 2, 2.5, 0.05, 0.005
	total_coordinates = MultiscaleHarris(Urban_gray, N, s, sigma_0, rho_0, k, theta_corn)
	interest_points_visualization(Urban_colored,total_coordinates).set_title('Multiscale Corner Detection')
	plt.show()