// Implementing list scan in CUDA
// @Jiangyan Feng, jf8@illinois.edu

// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *temp, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[BLOCK_SIZE*2];
  int tx = threadIdx.x;
  int start = 2*blockIdx.x*blockDim.x;

  if (start + 2*tx < len) 
    XY[2*tx] = input[start + 2*tx];
  else
    XY[2*tx] = 0.;
  if (start + 2*tx + 1 < len) 
    XY[2*tx + 1] = input[start + 2*tx + 1];
  else
    XY[2*tx + 1] = 0.;
  
  for (unsigned int stride = 1; stride <= 2*blockDim.x; stride *= 2){
    __syncthreads();
    int index = (tx + 1) * 2 * stride - 1;
    if (index < 2*BLOCK_SIZE){
      XY[index] += XY[index - stride];
    } 
  }
  for (int stride = (2*BLOCK_SIZE)/4; stride > 0; stride /= 2){
    __syncthreads();
    int index = (tx + 1) * 2 * stride - 1;
    if (index + stride < 2*BLOCK_SIZE)
      XY[index + stride] += XY[index];
  }
  __syncthreads();
  if (tx == 0) temp[blockIdx.x] = XY[2*BLOCK_SIZE - 1];
  if (start + 2*tx < len) output[start + 2*tx] = XY[2*tx];
  if (start + 2*tx + 1 < len) output[start + 2*tx + 1] = XY[2*tx + 1];
}

__global__ void add(float *input1, float *input2, int len){
  int tx = threadIdx.x;
  int start = (blockIdx.x + 1)*blockDim.x*2;
  if (start + 2*tx < len)
    input1[start + 2*tx] += input2[blockIdx.x];
  if (start + 2*tx + 1< len)
    input1[start + 2*tx + 1] += input2[blockIdx.x];
}
  
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceTemp1;
  float *deviceTemp2;
  float *deviceSum;
  int numElements; // number of elements in the list
  
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  int numTempElements = (numElements + 2*BLOCK_SIZE - 1)/(2*BLOCK_SIZE);
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceTemp1, numTempElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceTemp2, 1 * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceSum, numTempElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(deviceTemp1, 0, numTempElements * sizeof(float)));
  wbCheck(cudaMemset(deviceTemp2, 0, 1 * sizeof(float)));
  wbCheck(cudaMemset(deviceSum, 0, numTempElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid1(numTempElements, 1, 1);
  dim3 dimGrid2(1, 1, 1);
  dim3 dimGrid3(numTempElements - 1, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan <<<dimGrid1, dimBlock>>> (deviceInput, deviceTemp1, deviceOutput, numElements);
  if (numTempElements > 1){
    scan <<<dimGrid2, dimBlock>>> (deviceTemp1, deviceTemp2, deviceSum, numTempElements);
    add <<<dimGrid3, dimBlock>>> (deviceOutput, deviceSum, numElements);
  }
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceTemp1);
  cudaFree(deviceTemp2);
  cudaFree(deviceSum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
