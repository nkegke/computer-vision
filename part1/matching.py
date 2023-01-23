import corner_detect import as det
import descriptors as desc

if __name__='main':
	
	det_fun = lambda I: det.HarrisStephens(I, 1.5, 2, 0.05, 0.005)
	# detect_fun = lambda I: p2.MultiscaleBoxFilt(I, 3, 1.3, 2, 0.002)
	
	desc_fun = lambda I, kp: desc.featuresSURF(I,kp)
	# desc_fun = lambda I, kp: desc.featuresHOG(I,kp)
	
	avg_scale_errors, avg_theta_errors = desc.matching_evaluation(det_fun, desc_fun)
	
	print('HarrisStephens SURF Avg. Scale Error for Images 1,2,3: {:.3f}'.format(np.average(avg_scale_errors)))
	print('HarrisStephens SURF Avg. Theta Error for Images 1,2,3: {:.3f}'.format(np.average(avg_theta_errors)))
	# print('Multiscale Box Filters HOG Avg. Scale Error for Images 1,2,3: {:.3f}'.format(np.average(avg_scale_errors)))
	# print('Multiscale Box Filters HOG Avg. Theta Error for Images 1,2,3: {:.3f}'.format(np.average(avg_theta_errors)))