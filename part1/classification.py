import import corner_detect import as det
import descriptors as desc
import numpy as np
from bagOfVisualWords import BagOfVisualWords

if __name__='main':

	detect_fun = lambda I: det.MultiscaleBoxFilt(I, 3, 1.3, 2, 0.002)
	desc_fun = lambda I, kp: desc.featuresSURF(I,kp)
	feats = p3.FeatureExtraction(detect_fun, desc_fun, saveFile = 'BFSURFfeats')
	feats = p3.FeatureExtraction(detect_fun, desc_fun, loadFile = 'BFSURFfeats')
	
	accs = []
	for k in range(5):
		data_train, label_train, data_test, label_test = p3.createTrainTest(feats, k)
	
		#K-means
		BOF_tr, BOF_ts = BagOfVisualWords(data_train, data_test)
	
		#SVM
		acc, preds, probas = desc.svm(BOF_tr, label_train, BOF_ts, label_test)
		accs.append(acc)
		
	print('Mean accuracy for Multiscale Harris with SURF descriptors: {:.3f}%'.format(100.0*np.mean(accs)))