#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 8
#define MAX_THREADS 1024
#define MAX_KERNEL_SIZE 7200

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float kernel[MAX_KERNEL_SIZE];

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil((1.0*W_out) / TILE_WIDTH);
    const int H_grid = ceil((1.0*H_out) / TILE_WIDTH);

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b, m, h, w, c, p, q;
    b = blockIdx.x;
    m = blockIdx.y;
    h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
    if (h >= H_out || w >= W_out) return; //boundary checking

    float acc = 0.;
    for (c = 0; c < C; c++){ //sum over all input channels
        for (p = 0; p < K; p++){ //loop over K*K filter
            for (q = 0; q < K; q++){
                acc += x4d(b, c, h + p, w + q)*k4d(m, c, p, q);
            }
        }
    }
    y4d(b, m, h, w) = acc; 

#undef y4d
#undef x4d
#undef k4d
}

__global__ void unroll_kernel(float* x, float* x_unroll, int C, int H, int W, int K) {
/*
   Code to unroll input matrix to recast convolution layer as matrix multiply
*/
	int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

#define x3d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + i0]
#define x_unroll2d(i1, i0) x_unroll[(i1) * (W_out*H_out) + i0]
	
    int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
    //int b = blockIdx.x;
    int t = blockIdx.x*blockDim.x + threadIdx.x;
//	if (t == 0) {
//		printf("K:%d", K);
//		printf("C:%d", C);
//		printf("H:%d", H);
//		printf("W:%d", W);
//	}

    if (t < C * W_unroll) {
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        h_unroll = h_out * W_out + w_out;
        w_base = c * K * K;
        for(p = 0; p < K; p++){
            for(q=0; q<K; q++){
                w_unroll = w_base + p * K + q;
                x_unroll2d(w_unroll, h_unroll) = x3d(c, h_out + p, w_out + q); //Need to compute proper indices for this line!!
            }

        }
    }
#undef x3d
#undef x_unroll2d
}
__global__ void simple_matrix_mul(float* A, float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns,
                                  int numCRows, int numCColumns){

    int row  = blockIdx.y*blockDim.y + threadIdx.y;
    int col  = blockIdx.x*blockDim.x + threadIdx.x;
    int width = numAColumns;
  
  	if ((row < numARows) && (col < numBColumns)){
    	float Pvalue = 0;
    	for (int k=0; k < width; ++k){
      		Pvalue += A[row*numAColumns+k] * B[k*numBColumns+col];
    	}
    	C[row*numBColumns+col] = Pvalue;
  	}
}

__global__ void forward_matmul_kernel(float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns,
                                     int numCRows, int numCColumns){
/* Forward pass using matrix multiplication
*/

    //__shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

  // Loop over the M and N tiles required to compute the P element
  // The code assumes that the Width is a multiple of TILE_WIDTH!
    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH + 1; ++m) {
        // Collaborative loading of M and N tiles into shared memory
        //if(Row < numARows && m*TILE_WIDTH+tx < numAColumns) {
        //  subTileA[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
        //}
        //else {
        //  subTileA[ty][tx] = 0;
        //}
        if (m*TILE_WIDTH+ty < numBRows && Col < numBColumns) {
          subTileB[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
        }
        else {
          subTileB[ty][tx] = 0;
        }
        __syncthreads();
        if (Row < numCRows && Col < numCColumns) {
          for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue += kernel[Row*numAColumns+m*TILE_WIDTH+k] * subTileB[k][tx];
          }
        }
        __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns) {
		C[Row*numCColumns+Col] = Pvalue;
    }
	__syncthreads();
	//if ((bx == 0) && (by == 0) && (tx == 0) && (ty == 0)) {
		//printf("C: %f\n",C[0]);
		//printf("%d,%d\n",Row,Col);
	//}
	//__syncthreads();
}

__global__ void conv_layer_kernel(int H, int W, int M, int C, int K, int W_out, int H_out, float* x, float* y){

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    //__shared__ float tileMatA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileMatB[TILE_WIDTH][TILE_WIDTH];

    int b = blockIdx.z;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int column = blockIdx.x * TILE_WIDTH + tx;
    int numMatAColumns = C*K*K;

    float acc = 0.0;

    int num_iterations = ceil(numMatAColumns/(1.0*TILE_WIDTH));

    for (int i = 0; i < num_iterations; i++) {
        //int temp_col = i*TILE_WIDTH + tx;
        int temp_row = i*TILE_WIDTH + ty;

        //tileMatA[ty][tx] = 0;
        tileMatB[ty][tx] = 0;

        //int W_m = row;
        //int W_c = temp_col/(K*K);
        //int W_h = (temp_col%(K*K))/K;
        //int W_w = (temp_col%(K*K))%K;
        //if (temp_col < numMatAColumns && row < M)
        //    tileMatA[ty][tx] = k4d(W_m, W_c, W_h, W_w);
        //else
        //    tileMatA[ty][tx] = 0;

        int X_b = b;
        int X_c = temp_row/(K*K);
        int X_p = temp_row%(K*K)/K, X_q = (temp_row%(K*K))%K; 
        int X_h = column/W_out, X_w = column%W_out;
        if (temp_row < numMatAColumns && column < H_out*W_out)
            tileMatB[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
        else
            tileMatB[ty][tx] = 0;

        __syncthreads();
        if (row < M && column < W_out*H_out){ 
            for (int q = 0; q < TILE_WIDTH; q++)
                acc += kernel[row*numMatAColumns+i*TILE_WIDTH+q] * tileMatB[q][tx];
        }
        __syncthreads();
    }

        int Y_b = b;
        int Y_m = row;
        int Y_h = column / W_out, Y_w = column % W_out;
    
        if (row < M && column < W_out*H_out)
            y4d(Y_b, Y_m, Y_h, Y_w) = acc;
}
/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Set the kernel dimensions
    //const int W_grid = ceil((1.0*W_out) / TILE_WIDTH);
    //const int H_grid = ceil((1.0*H_out) / TILE_WIDTH);
    //const int Z = H_grid * W_grid;

    //Unroll matrix
    //int W_unroll = C * K * K;
    //int H_unroll = H_out * W_out;

    //int num_threads_unroll = C * H_out * W_out;
    //int num_blocks_unroll = ceil(float(1.0*num_threads_unroll)/MAX_THREADS);

    //float *x_unroll_host; //Unrolled x matrix
    //float *x_unroll_device;
    //cudaMalloc((void**) &x_unroll_device, W_unroll * H_unroll * sizeof(float));

	int numARows = M;
	int numACols = C*K*K;
	//int numBRows = C*K*K;
	//int numBCols = H_out*W_out;
	//int numCRows = M;
	//int numCCols = H_out*W_out;
	//float* Y_ptr = y.dptr_;
	//float* X_ptr = x.dptr_;
    //float* W_ptr = w.dptr_;

    //printf("numARows is %d\n", numARows);
    //printf("numACols is %d\n", numACols);
    //printf("K is %d\n", K);

    cudaMemcpyToSymbol(kernel, w.dptr_, sizeof(float)*numARows*numACols); //Store kernel in constant memory

    dim3 gridDim(ceil((1.0*H_out*W_out)/TILE_WIDTH), ceil(M/(1.0*TILE_WIDTH)), B);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
	conv_layer_kernel<<<gridDim, blockDim>>>(H, W, M, C, K, W_out, H_out, x.dptr_, y.dptr_);
		//cudaDeviceSynchronize();
	
    
	MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

	//printf("unroll:%f\n", w.dptr_[0]);

    //Initialize block and grid dimensions for matrix mul


    

    //dim3 gridDim(B, M, Z);
    //dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    //forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
