#include <numeric>     // std::accumulate
#include <algorithm>   // std::transform, std::copy
#include <vector>      // std::vector
#include <cmath>       // std::sqrt, std::ceil
#include "cp.h"
#include <cuda_runtime.h>

#define TILE 32

//Kernel code
__global__ void correlationKernel(float* input, float* output, int nx, int ny){
  int bx = blockDim.x, by = blockDim.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int y = ty + blockIdx.y * TILE;
  int x = tx + blockIdx.x * TILE;
  __shared__ float sub1[TILE][TILE];
  __shared__ float sub2[TILE][TILE];
  float sum = 0.0;
  for (int i = 0; i < ((nx -1)/TILE)+1; ++i){   //rounding up
    if ((i * TILE + tx) < nx && y < ny)			
      sub1[ty][tx] = input[y*nx + i*TILE + tx];
    else
      sub1[ty][tx] = 0;
    if ((i * TILE + ty) < nx && x < ny)			
      sub2[ty][tx] = input[x*nx + i*TILE + ty];
    else
      sub2[ty][tx] = 0;
    __syncthreads();
    for (int j = 0; j < TILE; ++j)
      sum += sub1[ty][j]*sub2[j][tx];
    __syncthreads();
  }
  if (y < ny && x < ny)
    output[((blockIdx.y*by+ty)*ny)+(blockIdx.x*bx)+tx] = sum;
}

void correlate(int ny, int nx, const float* data, float* result) {
  float rowMean, normFactor;
  int rowStart, rowEnd;
  size_t inputSize = ny * nx;
  size_t outputSize = ny * ny;
  float* hostIn = 0;
  float* deviceIn = 0;
  float* deviceOut = 0;
  std::vector<float> zeroMeanVec(nx), elemSqrdVec(nx);
  cudaMallocHost((void**) &hostIn, inputSize * sizeof(float));
  cudaMalloc((void**) &deviceIn, inputSize * sizeof(float));
  cudaMalloc((void**) &deviceOut, outputSize * sizeof(float));
  dim3 blockSize(TILE,TILE);                       
  dim3 gridSize(std::ceil(float(ny)/blockSize.x), std::ceil(float(ny)/blockSize.y));
  for(int y = 0; y < ny; ++y){
    rowStart = y*nx;
    rowEnd = nx+rowStart;
    //Find mean of the current row
    rowMean = std::accumulate(data+rowStart, data+rowEnd, 0.0) / float(nx);
    //Subtract each element of the current row from mean to make row zero mean
    std::transform(data+rowStart, data+rowEnd, zeroMeanVec.begin(), [&rowMean](float val){ return (val - rowMean);});
    //Find square of each element of the current row
    std::transform(zeroMeanVec.begin(), zeroMeanVec.end(), elemSqrdVec.begin(), [](float val){ return (val * val);});
    //Find normalization factor  of the current row
    normFactor = std::sqrt(std::accumulate(elemSqrdVec.begin(), elemSqrdVec.end(), 0.0));
    //Normalize the current row so that the sum of the squares of the elements of the row is 1 with zero mean
    std::transform(zeroMeanVec.begin(), zeroMeanVec.end(), zeroMeanVec.begin(), [&normFactor](float val){ return (val / normFactor);});
    //Save the normalized result in a matrix of dimension ny*nx
    std::copy(zeroMeanVec.begin(), zeroMeanVec.end(), hostIn+rowStart);
  }
  //Copy host data to GPU
  cudaMemcpy(deviceIn, hostIn, inputSize * sizeof(float), cudaMemcpyHostToDevice);
  //Kernel call
  correlationKernel<<<gridSize, blockSize>>>(deviceIn, deviceOut, nx, ny);
  //Copy GPU data to host
  cudaMemcpy(result, deviceOut, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
  //Free memory
  cudaFreeHost(hostIn);
  cudaFree(deviceIn);
  cudaFree(deviceOut);
}
