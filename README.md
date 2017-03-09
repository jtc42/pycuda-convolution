# cuconv
Simple image convolutions in PyCUDA. Mainly a learning exercise. 

## Requirements
* CUDA
* PyCUDA

## Usage
* For a speed comparison between cuconv, Scipy.signal, and a basic CPU convolution, run 'main.py'. 
* Results are stored in the 'results' folder.
* Basic usage:
```
cuconv.convolve(input_image_as_array, conv_kernel_as_array)
```
* The function expects a pair of 2D numpy-arrays, with the first corresponding to the input image, and the second being an odd-dimensioned convolution kernel.

## Existing data
* See 'results' folder for image results of speed comparison in 'main.py', on horizontal and vertical Sobel edge detection, and 9x9 box blur kernels.
