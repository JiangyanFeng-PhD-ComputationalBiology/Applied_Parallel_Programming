// Implementing histogram equalization in CUDA
// @Jiangyan Feng, jf8@illinois.edu


// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 1024
//@@ insert code here
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Step 1: cast the image from float to unsigned char
__global__ void cast(float *in, unsigned char *out, int size){
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size)
    out[i] = (unsigned char) (255 * in[i]);
}
// Step 2: convert the image tp grayscale
__global__ void convert(unsigned char *in, unsigned char *out, int size1, int size2){
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if ((3*i + 2 < size1) && (i < size2)){
  unsigned char r = in[3*i];
  unsigned char g = in[3*i + 1];
  unsigned char b = in[3*i + 2];
  out[i] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

// Step 3: compute histogram
__global__ void hist(unsigned char *in, unsigned int *out, int size){
unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
__shared__ unsigned int histo_s[HISTOGRAM_LENGTH];
if (threadIdx.x < HISTOGRAM_LENGTH)
  histo_s[threadIdx.x] = 0;
__syncthreads();

if (i < size){
  unsigned char pos = in[i];
  atomicAdd(&(histo_s[pos]), 1);
}
__syncthreads();

if (threadIdx.x < HISTOGRAM_LENGTH){
  atomicAdd(&(out[threadIdx.x]), histo_s[threadIdx.x]);
}
}

// Step 4: cumulative distribution
__global__ void cdf(unsigned int *in, float *out, int size1, int size2){
  //loading
  __shared__ float blockSum[HISTOGRAM_LENGTH];
  unsigned int tx = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;
  if (start + 2*tx < size1)
    blockSum[2*tx] = in[start + 2*tx]/((float) size2);
  else
    blockSum[2*tx] = 0.;
    
  if (start + 2*tx + 1< size1)
    blockSum[2*tx + 1] = in[start + 2*tx + 1]/((float) size2);
  else
    blockSum[2*tx + 1] = 0.;  
  
  //reduction
  int stride = 1;
  while (stride < 2*blockDim.x){
  __syncthreads();
  int index = (tx + 1)*stride*2 - 1;
  if (index < 2*blockDim.x && (index - stride) >= 0){
    blockSum[index] += blockSum[index - stride];
  }
  stride *= 2;
  }
 //post scan
 stride = blockDim.x / 2;
 while (stride > 0){
   __syncthreads();
   int index = (tx + 1)*stride*2 - 1;
   if (index + stride < 2 * blockDim.x){
     blockSum[index + stride] += blockSum[index];
   }
   stride /= 2;
 }
 __syncthreads();
 //output
 if (start + 2*tx < size1){
   out[start + 2*tx] = blockSum[2 * tx];
 }
 if (start + 2*tx + 1< size1){
   out[start + 2*tx + 1] = blockSum[2 * tx + 1];
 }
}

// Step 5: correct 

__global__ void correct(unsigned char *in1, float *in2, unsigned char *out, int size){
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  float cdfmin = in2[0];
  if (i < size){
    float res = 255 * (in2[in1[i]] - cdfmin) / (1.0 - cdfmin);
    out[i] = (unsigned char) (min(max(res, 0.0), 255.0));
  }
}

// Step 6: cast back
__global__ void cast_back(unsigned char *in, float *out, int size){
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size)
    out[i] = (float) (in[i] / 255.0);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceCDF;
  unsigned char *deviceRGB;
  unsigned char *deviceGrey;
  unsigned char *deviceCorrect;
  unsigned int *deviceHisto;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int size1 = imageWidth * imageHeight * imageChannels;
  int size2 = imageWidth * imageHeight;
  
  wbCheck(cudaMalloc((void **)&deviceInputImageData, size1 * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, size1 * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceRGB, size1 * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceGrey, size2 * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceCorrect, size1 * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceHisto, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, size1*sizeof(float), cudaMemcpyHostToDevice));
  int section1 = (size1 + BLOCK_SIZE - 1)/BLOCK_SIZE;
  int section2 = (size2 + BLOCK_SIZE - 1)/BLOCK_SIZE;
  dim3 dimGrid1(section1,1,1);
  dim3 dimGrid2(section2,1,1);
  dim3 dimGrid3(1,1,1);
  dim3 dimBlock1(BLOCK_SIZE, 1, 1);
  dim3 dimBlock2(HISTOGRAM_LENGTH/2, 1, 1); 
  
  cast <<< dimGrid1, dimBlock1 >>> (deviceInputImageData, deviceRGB, size1);
  convert <<< dimGrid2, dimBlock1 >>> (deviceRGB, deviceGrey, size1, size2);
  hist <<< dimGrid2, dimBlock1 >>> (deviceGrey, deviceHisto, size2);
  cdf <<< dimGrid3, dimBlock2 >>> (deviceHisto, deviceCDF, HISTOGRAM_LENGTH, size2);
  correct <<< dimGrid1, dimBlock1 >>> (deviceRGB, deviceCDF, deviceCorrect, size1);
  cast_back <<< dimGrid1, dimBlock1 >>> (deviceCorrect, deviceOutputImageData, size1);
  cudaDeviceSynchronize();
  
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, size1*sizeof(float), cudaMemcpyDeviceToHost));

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceCDF);
  cudaFree(deviceRGB);
  cudaFree(deviceGrey);
  cudaFree(deviceCorrect);
  cudaFree(deviceHisto);

  free(hostInputImageData);
  free(hostOutputImageData);
  return 0;
}
