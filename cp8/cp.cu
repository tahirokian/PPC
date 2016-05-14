#include <numeric>     // std::accumulate
#include <algorithm>   // std::transform, std::copy
#include <vector>      // std::vector
#include <cmath>       // std::sqrt, std::ceil
#include "cp.h"
#include <cuda_runtime.h>

//Kernel code
__global__ void correlationKernel(double* input, float* output, int nx, int ny){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  double sum = 0;
  if (x >= ny || y >= ny || x < y) // Exit if outside, do not calculate lower triangle
    return;
  for (int i = 0; i < nx; ++i)
    sum += (input[(y*nx)+i] * input[(x*nx)+i]);
  output[(y*ny)+x] = float(sum);
}

void correlate(int ny, int nx, const float* data, float* result) {
  double rowMean, normFactor;
  int rowStart, rowEnd;
  size_t inputSize = ny * nx;
  size_t outputSize = ny * ny;
  double* hostData = 0;
  double* dataGPU = 0;
  float* resultGPU = 0;
  std::vector<double> zeroMeanVec(nx), elemSqrdVec(nx);
  cudaMallocHost((void**) &hostData, inputSize * sizeof(double));
  cudaMalloc((void**) &dataGPU, inputSize * sizeof(double));
  cudaMalloc((void**) &resultGPU, outputSize * sizeof(float));
  dim3 blockSize(8,8);                                                 			//block of 8x8x1
  dim3 gridSize(std::ceil(double(ny)/blockSize.x), std::ceil(double(ny)/blockSize.y));	//grid of (ny/8)x(ny/8)x1
  for(int y = 0; y < ny; ++y){
    rowStart = y*nx;
    rowEnd = nx+rowStart;
    //Find mean of the current row
    rowMean = std::accumulate(data+rowStart, data+rowEnd, 0.0) / double(nx);
    //Subtract each element of the current row from mean to make row zero mean
    std::transform(data+rowStart, data+rowEnd, zeroMeanVec.begin(), [&rowMean](double val){ return (val - rowMean);});
    //Find square of each element of the current row
    std::transform(zeroMeanVec.begin(), zeroMeanVec.end(), elemSqrdVec.begin(), [](double val){ return (val * val);});
    //Find normalization factor  of the current row
    normFactor = std::sqrt(std::accumulate(elemSqrdVec.begin(), elemSqrdVec.end(), 0.0));
    //Normalize the current row so that the sum of the squares of the elements of the row is 1 with zero mean
    std::transform(zeroMeanVec.begin(), zeroMeanVec.end(), zeroMeanVec.begin(), [&normFactor](double val){ return (val / normFactor);});
    //Save the normalized result in a matrix of dimension ny*nx
    std::copy(zeroMeanVec.begin(), zeroMeanVec.end(), hostData+rowStart);
  }
  //Copy host data to gpu
  cudaMemcpy(dataGPU, hostData, inputSize * sizeof(double), cudaMemcpyHostToDevice);
  //Kernel call
  correlationKernel<<<gridSize, blockSize>>>(dataGPU, resultGPU, nx, ny);
  //Copy gpu data to host
  cudaMemcpy(result, resultGPU, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(hostData);
  cudaFree(dataGPU);
  cudaFree(resultGPU);
}
