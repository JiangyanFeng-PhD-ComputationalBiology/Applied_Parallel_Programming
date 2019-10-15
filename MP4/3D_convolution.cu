// Implementing 3D convolution in CUDA
// @Jiangyan Feng, jf8@illinois.edu

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define BLOCK_WIDTH 8
#define MASK_WIDTH 3
#define radius (MASK_WIDTH-1)/2
#define TILE_WIDTH (BLOCK_WIDTH + MASK_WIDTH -1)

//@@ Define constant memory for device kernel here
__constant__ float M_c[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int z = blockIdx.z*blockDim.z + threadIdx.z;
  
  __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
  // Loading part 1
  int tid = threadIdx.x + threadIdx.y*BLOCK_WIDTH + threadIdx.z*BLOCK_WIDTH*BLOCK_WIDTH;
  int tidx = tid % TILE_WIDTH;
  int tidy = (tid / TILE_WIDTH) % TILE_WIDTH;
  int tidz = (tid / TILE_WIDTH) / TILE_WIDTH;
  
  int inx = tidx + (blockIdx.x*BLOCK_WIDTH) - radius;
  int iny = tidy + (blockIdx.y*BLOCK_WIDTH) - radius;
  int inz = tidz + (blockIdx.z*BLOCK_WIDTH) - radius;
  int inid = inx + iny*x_size + inz*x_size*y_size;
  
  if (inx >= 0 && inx < x_size && iny >= 0 && iny < y_size && inz >= 0 && inz < z_size){
    N_ds[tidz][tidy][tidx] = input[inid];
  }
  else {
    N_ds[tidz][tidy][tidx] = 0;
  }
  __syncthreads();
  
  // Loading part 2
  tid = threadIdx.x + threadIdx.y*BLOCK_WIDTH + threadIdx.z*BLOCK_WIDTH*BLOCK_WIDTH + BLOCK_WIDTH*BLOCK_WIDTH*BLOCK_WIDTH;
  tidx = tid % TILE_WIDTH;
  tidy = (tid / TILE_WIDTH) % TILE_WIDTH;
  tidz = (tid / TILE_WIDTH) / TILE_WIDTH;
  
  inx = tidx + (blockIdx.x*BLOCK_WIDTH) - radius;
  iny = tidy + (blockIdx.y*BLOCK_WIDTH) - radius;
  inz = tidz + (blockIdx.z*BLOCK_WIDTH) - radius;
  inid = inx + iny*x_size + inz*x_size*y_size;
  
  if (tidz < TILE_WIDTH){
     if (inx >= 0 && inx < x_size && iny >= 0 && iny < y_size && inz >= 0 && inz < z_size){
       N_ds[tidz][tidy][tidx] = input[inid];
     }
    else{
      N_ds[tidz][tidy][tidx] = 0;
    }
  }
  __syncthreads();
  
  // Calculating 
  float sum = 0.0;
  for (int m = 0; m < MASK_WIDTH; m++){
    for (int j = 0; j < MASK_WIDTH; j++){
      for (int i = 0; i < MASK_WIDTH; i++){
        sum += N_ds[threadIdx.z + m][threadIdx.y + j][threadIdx.x + i]*M_c[m*MASK_WIDTH*MASK_WIDTH + j*MASK_WIDTH +i]; 
      }
    }
  }
 
 // Saving
 if (x < x_size && y < y_size && z < z_size){
   output[x + y*x_size + z*x_size*y_size] = sum;
   __syncthreads();  
 }
}
  
int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int inputSize = inputLength - 3;
  cudaMalloc((void **)&deviceInput, inputSize*sizeof(float));
  cudaMalloc((void **)&deviceOutput, inputSize*sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpyToSymbol(M_c, hostKernel, kernelLength*sizeof(float));
  cudaMemcpy(deviceInput, &hostInput[3], inputSize*sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((1.0*x_size)/BLOCK_WIDTH), ceil((1.0*y_size)/BLOCK_WIDTH), ceil((1.0*z_size)/BLOCK_WIDTH));
  dim3 dimBLOCK(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
  
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBLOCK>>> (deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, inputSize*sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
