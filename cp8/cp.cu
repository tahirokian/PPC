#include <numeric>     // std::accumulate
#include <algorithm>   // std::transform, std::copy, std::max
#include <vector>      // std::vector
#include <cmath>       // std::sqrt
#include "cp.h"
#include <cuda_runtime.h>
#include <iostream>

//Kernel code
__global__ void correlationKernel(double* input, float* output, int nx, int ny){
  double sum = 0;
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= ny || y >= ny) // Exit if outside
    return;
  for (int i = 0; i < nx; ++i){
    sum += (input[(y*nx)+i] * input[(x*nx)+i]);
  }
  output[(y*ny)+x] = float(sum);
}

void correlate(int ny, int nx, const float* data, float* result) {
  double rowMean, normFactor;
  int rowStart, rowEnd;
  int inputSize = ny * nx;
  int outputSize = ny * ny;
  double* hostData = 0;
  double* dataGPU = 0;
  float* resultGPU = 0;
  float* hostResult = 0;
  std::vector<double> zeroMeanVec(nx), elemSqrdVec(nx);
  cudaHostAlloc((void**) &hostData, inputSize * sizeof(double), cudaHostAllocMapped);
  cudaHostAlloc((void**) &hostResult, outputSize * sizeof(float), cudaHostAllocMapped);
  cudaMalloc((void**) &dataGPU, inputSize * sizeof(double));
  cudaMalloc((void**) &resultGPU, outputSize * sizeof(float));
  dim3 blockSize(8,8);                                                 //block of 8x8x1
  dim3 gridSize(std::ceil(ny/blockSize.x), std::ceil(ny/blockSize.y)); //grid of (ny/8)x(ny/8)x1, each thread calculates one element of output matrix
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
  cudaMemcpy(hostResult, resultGPU, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

  //Matrix multiplication on CPU for comparison with GPU
  for (int j=0; j<ny; ++j){           //Move through rows of X
    for (int i=j; i<ny; ++i){         //Move through columns of XT
      double sum = 0;
      for (int k=0; k<nx; ++k){       //Move through column of X and rows of XT
         sum += (hostData[(j*nx)+k] * hostData[(i*nx)+k]);
      }
      result[i + (j*ny)] = float(sum);
    }
  }
  
  std::cout << "\nresult[0], hostResult[0]: " << result[0] << " " << hostResult[0] << std::endl;
  std::cout << "result[ny-1], hostResult[ny-1]: " << result[ny-1] << " " << hostResult[ny-1] << std::endl;
  cudaFree(hostData);
  cudaFree(hostResult);
  cudaFree(dataGPU);
  cudaFree(resultGPU);
}
