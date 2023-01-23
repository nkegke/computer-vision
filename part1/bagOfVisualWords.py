import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def histogram(descriptors, centroids):
    
    #compute minimum euclidean distance between each descriptor and centroid
    eucl = cdist(descriptors,centroids)
    min_dist = np.argmin(eucl, axis=1)

    #compute histogram, bins: centroids, input: descriptors
    bins = np.arange(centroids.shape[0]+1)
    hist, bin_edges = np.histogram(min_dist, bins)
    
    #normalize by L2 norm
    L2 = np.linalg.norm(hist)
    return hist/L2


def BagOfVisualWords(data_train, data_test):
    
    #data_train is list of 2D arrays
    #concatenating into 1 2D array
    feature_vector = data_train[0].copy()
    for photo_desc in data_train[1:]:
        feature_vector = np.concatenate((feature_vector,photo_desc), axis=0)

    #choosing half of feature_vector to be our subset
    idx = np.random.choice(range(np.shape(feature_vector)[0]), np.shape(feature_vector)[0]//2 + 1, replace=False)
    random_subset = feature_vector[idx]

    #perform k-means with 500 centroids
    kmeans = KMeans(n_clusters=500)
    kmeans.fit(random_subset)
    centroids = kmeans.cluster_centers_

    #apply histogram function on every image
    BOF_tr = np.array([histogram(img,centroids) for img in data_train])
    BOF_ts = np.array([histogram(img,centroids) for img in data_test])

    return BOF_tr, BOF_ts


