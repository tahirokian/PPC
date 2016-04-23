#include <numeric>     // std::accumulate
#include <algorithm>   // std::transform, std::copy, std::max
#include <vector>      // std::vector
#include <cmath>       // std::sqrt
#include "cp.h"

void correlate(int ny, int nx, const float* data, float* result){
  int i, j, k, y;
  double rowMean, sum, normFactor;
  int rowStart, rowEnd, rowNumber;
  std::vector<double> zeroMeanVec(nx), elemSqrdVec(nx);
  std::vector<std::vector<double>> X(ny, std::vector<double>(nx));
  for(y = 0; y < ny; ++y){
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
    std::copy(zeroMeanVec.begin(), zeroMeanVec.end(), X[y].begin());
  }
  //Matrix multiplication
  #pragma omp parallel for schedule(static,1) private(i, k, sum, rowNumber)
  for (j=0; j<ny; ++j){           //Move through rows of X
    rowNumber = j * ny;
    for (i=j; i<ny; ++i){         //Move through rows of X for i >= j
      sum = 0;
      for (k=0; k<nx; ++k){       //Move through columns of X
        sum += (X[j][k] * X[i][k]);
      }
      result[i + rowNumber] = sum;
    }
  }
}
