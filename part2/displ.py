import numpy as np

def displ(d_x, d_y):
    #print('max=',np.max(np.abs(d_x)), np.max(np.abs(d_y)))
    energy = d_x**2 + d_y**2
    max_energy = np.max(energy)
    threshold = 0.95*max_energy
    
    [asd_y, asd_x] = np.where(energy > threshold)

    mean_dx = 0
    mean_dy = 0
    for i in range(len(asd_x)):
        mean_dx = mean_dx + d_x[asd_y[i]][asd_x[i]]
        mean_dy = mean_dy + d_y[asd_y[i]][asd_x[i]]
    total_nonzero = i+1

    mean_dx = mean_dx/(total_nonzero+1)
    mean_dy = mean_dy/(total_nonzero+1)
    #print('we return=', mean_dx, mean_dy)
    return int(np.ceil(mean_dx)), int(np.ceil(mean_dy))