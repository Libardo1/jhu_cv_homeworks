import skimage.io as io
from skimage.util import pad, view_as_windows
import numpy as np
from scipy.ndimage.filters import generic_filter

from matplotlib import pyplot as plt

def normalize_matrix(matrix):
    
    matrix = matrix - matrix.mean()
    
    # Stand. deviation
    matrix_std = np.sqrt((matrix**2).sum())

    matrix = matrix / matrix_std
    
    return matrix

window_size = 15

img_l = io.imread('HW3_images/scene_l.bmp')
img_r = io.imread('HW3_images/scene_r.bmp')

# Integer division
pad_width = window_size / 2
window_shape = (window_size, window_size)

img_l_padded = pad(img_l, pad_width=pad_width, mode='symmetric')
img_r_padded = pad(img_r, pad_width=pad_width, mode='symmetric')

img_l_view = view_as_windows(img_l_padded, window_shape=window_shape, step=1)
img_r_view = view_as_windows(img_r_padded, window_shape=window_shape, step=1)

img_height, img_width = img_l.shape

result = np.zeros(img_l.shape)

for row in xrange(img_height):
    for col in xrange(img_width):
        
        first_patch = img_l_view[row, col]
        
        first_patch_norm = normalize_matrix(first_patch)
        
        current_scanline = img_r_view[row, :]
        scores = np.zeros(img_width)
        
        for pixel in xrange(img_width):
            
            second_patch = current_scanline[pixel]
            
            second_patch_norm = normalize_matrix(second_patch)
            
            scores[pixel] = (first_patch_norm * second_patch_norm).sum()
            
        best_match = np.argmax(scores)
        
        result[row, col] = best_match

# pixels
f = 400

#mm
b = 100

horiz_span = np.arange(img_width)
u_l = np.tile(horiz_span, (img_height, 1))


vert_span = np.arange(img_height).reshape((-1, 1))
v_l = np.tile(vert_span, (1, img_width))

z = f*b / (u_l - result)
z[z > 10000] = 0

x = b* ((u_l + result) / 2*(u_l - result))
y = b* ((v_l + v_l) / 2*(u_l - result))


io.imshow(z, cmap=plt.get_cmap('gray'))
io.show()

strings_to_write = []

for row in xrange(img_height):
    for col in xrange(img_width):
        
        data = [str(x[row, col]), str(y[row, col]), str(z[row, col])]
        new_string = " ".join(data)
        strings_to_write.append(new_string)
        
file_contents = '\n'.join(strings_to_write)

with open('point_cloud.txt', 'w') as the_file:
    the_file.write(file_contents)