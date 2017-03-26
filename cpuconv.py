import numpy as np


def convolve(a, b):
    
    image, kernel = [np.array(i).astype(np.float32) for i in [a, b]]
    
    # Sum of absolute values in kernel matrix
    kernel_sum = np.absolute(kernel).sum()
    
    # Calculate the dimensions for iteration over the pixels and weights
    i_width, i_height = image.shape[1], image.shape[0]
    k_width, k_height = kernel.shape[1], kernel.shape[0]

    if k_width % 2 == 0 or k_height % 2 == 0:
        print("Warning: Kernel dimensions not odd. Centre point ambiguous, could break code.")
    
    padding_w = k_width-1
    padding_h = k_height-1
    
    f_width = i_width - padding_w
    f_height = i_height - padding_h
    
    # Prepare the output array
    filtered = np.zeros((f_height, f_width))
    
    # Iterate over image
    for y in range(f_height):
        for x in range(f_width):
            
            weighted_pixel_sum = 0  # Initial pixel value
    
            # Iterate over kernel
            for ky in range(-int(padding_h/2), int(padding_h/2)+1):
                for kx in range(-int(padding_w/2), int(padding_w/2)+1):
                    
                    # Coordinates of pixel on original image (for each kernel element)
                    pixel_y = int(y - ky + padding_h/2)
                    pixel_x = int(x - kx + padding_w/2)
        
                    # Set value of pixel based on coordinates
                    pixel = image[pixel_y, pixel_x] 

                    # Get weight of this pixel from kernel matrix
                    weight = kernel[ky + int(k_height / 2), kx + int(k_width / 2)]
    
                    # Weigh the pixel value and sum, update pixel value for this image coordinate
                    weighted_pixel_sum += pixel * weight
    
            # Set pixel at location (x,y) in output to sum of the weighed neighborhood
            filtered[y, x] = weighted_pixel_sum / kernel_sum
    
    return filtered
