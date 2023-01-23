import numpy as np
import scipy

def pointDetection(skin):

    labeled_array, num_features = scipy.ndimage.label(skin)
    table = np.where(labeled_array == 1)
    y_head = table[0].min()
    x_head = table[1].min()
    head_height = table[0].max() - y_head
    head_width = table[1].max() - x_head
    #print(x_head,y_head,head_height,head_width)

    labeled_array, num_features = scipy.ndimage.label(skin)
    table = np.where(labeled_array == 2)
    y_left = table[0].min() - 15
    x_left = table[1].min() - 15
    left_width = table[0].max() - y_left + 15
    left_height = table[1].max() - x_left + 15

    labeled_array, num_features = scipy.ndimage.label(skin)
    table = np.where(labeled_array == 3)
    y_right = table[0].min() - 15
    x_right = table[1].min() - 15
    right_height = table[0].max() - y_right + 15
    right_width = table[1].max() - x_right + 15
    
    return [[x_head,y_head,head_width,head_height],[x_left,y_left,left_width,left_height],[x_right,y_right,right_width,right_height]]