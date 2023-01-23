import numpy as np
import cv2
import scipy

def lk(I1, I2, features, rho, epsilon, d_x0, d_y0):

    #Gaussian kernel
    n = int(2*np.ceil(3*rho)+1)
    gauss1D = cv2.getGaussianKernel(n, rho)
    Gr = gauss1D @ gauss1D.T

    #grid for interpolation later
    x_0,y_0 = np.meshgrid(np.arange(I1.shape[1]), np.arange(I1.shape[0]))

    #partial derivatives
    [I1_y, I1_x] = np.gradient(I1)

    #initial estimation
    dx_i = d_x0.copy()
    dy_i = d_y0.copy()

    #second frame
    In = I2.copy()

    #define when to stop
    threshold = 0.5
    mean_difference_x = mean_difference_y = 1742 #random big value
    max_iterations = 350
    i = 0

    #while i <= max_iterations:
    while mean_difference_x >= threshold and mean_difference_y >= threshold and i < max_iterations:
        i += 1
        #interpolating for non-integer indices
        height = In.shape[0]
        width = In.shape[1]

        In_1 = scipy.ndimage.map_coordinates(I1, [np.ravel(y_0+dy_i), np.ravel(x_0+dx_i)], order=1)
        In_1 = In_1.reshape(height,width)
        In_1x = scipy.ndimage.map_coordinates(I1_x, [np.ravel(y_0+dy_i), np.ravel(x_0+dx_i)], order=1)
        In_1x = In_1x.reshape(height,width)
        In_1y = scipy.ndimage.map_coordinates(I1_y, [np.ravel(y_0+dy_i), np.ravel(x_0+dx_i)], order=1)
        In_1y = In_1y.reshape(height,width)

        #partial derivatives
        A1 = In_1x.copy()
        A2 = In_1y.copy()
        E = In - In_1

        #compute u
        u11 = cv2.filter2D(A1**2, -1, Gr) + epsilon
        u22 = cv2.filter2D(A2**2, -1, Gr) + epsilon
        u12 = cv2.filter2D(A1*A2, -1, Gr)

        u1 = cv2.filter2D(A1*E, -1, Gr)
        u2 = cv2.filter2D(A2*E, -1, Gr)

        det = u11*u22-u12**2
        u_x = (u22/det) * u1 - (u12/det) * u2
        u_y = (-u12/det) * u1 + (u11/det) * u2

        #update d
        ################################
        dx_i = dx_i + u_x
        dy_i = dy_i + u_y
        ################################
        mean_difference_x = np.linalg.norm(u_x)
        mean_difference_y = np.linalg.norm(u_y)

    dx = np.zeros((dx_i.shape[0],dx_i.shape[1]))
    dy = np.zeros((dy_i.shape[0],dy_i.shape[1]))
    for coords in features:
        coords_y = int(coords[1])
        coords_x = int(coords[0])
        dx[coords_y][coords_x] = dx[coords_y][coords_x] + dx_i[coords_y][coords_x]
        dy[coords_y][coords_x] = dy[coords_y][coords_x] + dy_i[coords_y][coords_x]

    #print(i)
    d = []
    d.append(dx)
    d.append(dy)
    return np.array(d)