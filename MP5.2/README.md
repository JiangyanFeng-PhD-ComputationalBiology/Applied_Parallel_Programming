# MP5.2: List Scan


The purpose of this lab is to implement one or more kernels and their associated host code to perform parallel scan on a 1D list. The scan operator used will be addition. 

## The code performs the following:
* allocate device memory
* copy host memory to device
* initialize thread block and kernel grid dimensions
* invoke CUDA kernel
* copy results from device to host
* deallocate device memory
* implement the work-efficient scan kernel to generate per-block scan array and store the block sums into an auxiliary block sum array.
* use shared memory to reduce the number of global memory accesses, handle the boundary conditions when loading input list elements into the shared memory
* reuse the kernel to perform scan on the auxiliary block sum array to translate the elements into accumulative block sums. Note that this kernel will be launched with only one block.
* implement the kernel that adds the accumulative block sums to the appropriate elements of the per-block scan array to complete the scan for all the elements.

## Authors

Jiangyan Feng - jf8@illinois.edu

Ph.D. Candidate

Chemical and Bimolecular Engineering

University of Illinois at Urbana-Champaign

200 RAL, 600 S. Mathews Ave., Urbana IL 61801
