import cv2
import matplotlib.pyplot as plt
import numpy as np


#Returns 2D-Gaussian and Laplacian of Gaussian (13x13) kernels
# sigma:	gaussian std
def create_kernels(sigma):

    #2D Gaussian
    n = int(2*np.ceil(3*sigma)+1)
    gauss1D = cv2.getGaussianKernel(n, sigma) # Column vector
    gauss2D = gauss1D @ gauss1D.T # Symmetric gaussian kernel

    #LoG
    x = y = np.arange(np.ceil(-n/2), np.ceil(n/2), 1) #[-n/2 ... n/2] 1-D vector
    xx, yy = np.meshgrid(x,y, sparse=True) # Create a 2D-array, xx is row and yy is column
    LoG = ((xx**2+yy**2-2*sigma**2)/(2*np.pi*sigma**6))*np.exp((-xx**2-yy**2)/(2*sigma**2)) # Evaluate LoG function to grid

    return gauss2D, LoG


# I: 			input image
# sigma:		gaussian std (finetune-able)
# theta_edge:	threshold parameter to cancel low slope edges (finetune-able)
# approach:		LoG (linear) or dilation-erosion (non-linear/morphological)
# plot:			plot stages of detection
def EdgeDetect(I, sigma, theta_edge=0.16, approach, plot='no'):

    gauss2D, LoG = create_kernels(sigma)

    morph_kern = np.array([ #Morphological kernel
            [0,1,0],
            [1,1,1],
            [0,1,0]
        ], dtype=np.uint8)
    if (approach == 'linear'):
        
        L = cv2.filter2D(I, -1, LoG) #Applying Laplacian of Gaussian operator 
        I = cv2.filter2D(I, -1, gauss2D) #denoising original image for next steps
    elif (approach == 'non-linear'):
        I = cv2.filter2D(I, -1, gauss2D) #denoising image
        Ldil = cv2.dilate(I, morph_kern) #dilation
        Ler = cv2.erode(I, morph_kern) #erosion
        L = Ldil + Ler - 2*I #Laplacian = dilation + erosion - 2*image

    X = (L >= 0).astype(float) #Binary sign image I
    Y = cv2.dilate(X,morph_kern) - cv2.erode(X,morph_kern) #Zero crossings

    i = np.gradient(I)[0]
    j = np.gradient(I)[1]
    normgrad = np.sqrt(i**2+j**2)
    maxgrad = normgrad.max()
    BigSlope = (normgrad > theta_edge * maxgrad) #||GradI||>theta*max(||GradI||)
    Z = (Y.astype(int) & BigSlope)

    #Plotting
    if (plot=='plot'):
        fig, axs = plt.subplots(1,4)
        axs[0].imshow(I, cmap='gray')
        axs[0].set_title('Original image')
        axs[1].imshow(L, cmap='gray')
        axs[1].set_title('LoG')
        axs[2].imshow(X, cmap='gray')
        axs[2].set_title('Sign of LoG')
        axs[3].imshow(Z, cmap='gray')
        axs[3].set_title('Edges')
        plt.show()
    return Z
    

# img:	image to add noise and experiment with edge_detection
# dB:	noise in dB
def add_noise_to_img(img, dB):

	Imax = img.max()
	Imin = img.min()
	denom = 10**(dB/20)
	sigma_n = (Imax-Imin)/denom
	noise = np.random.normal(0,sigma_n,size = img.shape)
	img_noise = img + noise
	return img_noise

if __name__='main':
	img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
	img = img.astype(np.float)/255
	img_noise = add_noise_to_img(img, 10)
	D = EdgeDetect(img_noise, 2, 0.16, 'non-linear', 'plot')
