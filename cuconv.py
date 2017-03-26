import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# DEVICE SETUP
BLOCK_SIZE = 32  # Max 32. 32**2 = 1024, max for GTX1060
    
# Compile kernel
mod = SourceModule(open("kernel.cu", "r").read())

# Get functions
conv = mod.get_function("conv")


def convolve(a, b):
    global BLOCK_SIZE
    global conv
    
    a, b = [np.array(i).astype(np.float32) for i in [a, b]]
    
    # Matrix A 
    aw = np.int32(a.shape[1])  # Widthof in matrix
    ah = np.int32(a.shape[0])  # Height of in matrix
    
    # Matrix B (kernel)
    bw = np.int32(b.shape[1])  # Widthof in matrix
    if bw % 2 == 0:
        print("Kernel width is not an odd number! Strange things will happen...")
    bh = np.int32(b.shape[0])  # Height of in matrix
    if bh % 2 == 0:
        print("Kernel height is not an odd number! Strange things will happen...")
    b_sum = np.int32(np.absolute(b).sum())
    
    # Matrix C, subtract 2*padding, *2 because it's taken off all sides
    c = np.empty([ah-(bh-1), aw-(bw-1)])
    c = c.astype(np.float32)
    
    # Allocate memory on device
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)
    
    # Copy matrix to memory
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # Set grid size from A matrix
    grid = (int(aw/BLOCK_SIZE+(0 if aw % BLOCK_SIZE is 0 else 1)), 
            int(ah/BLOCK_SIZE+(0 if ah % BLOCK_SIZE is 0 else 1)), 
                          1)
    
    # Call gpu function
    conv(a_gpu, b_gpu, aw, ah, bw, bh, b_sum, c_gpu, block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=grid)
    
    # Copy back the result
    cuda.memcpy_dtoh(c, c_gpu)
    
    # Free memory. May not be useful? Ask about this.
    a_gpu.free()
    b_gpu.free()
    c_gpu.free()
    
    # Return the result
    return c
