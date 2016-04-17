#include <numeric>     // std::accumulate
#include <algorithm>   // std::transform, std::copy
#include <vector>      // std::vector
#include <cmath>       // std::sqrt

#include "cp.h"

void correlate(int ny, int nx, const float* data, float* result){
  int i, j, k, y;
  double rowMean;
  int rowStart;
  int rowEnd;
  double normFactor;
  std::vector<double> zeroMeanVec(nx), elemSqrdVec(nx);
  std::vector<std::vector<double>> X(ny, std::vector<double>(nx));
  std::vector<std::vector<double>> XT(nx, std::vector<double>(ny));
  for(y = 0; y < ny; ++y){
    rowStart = y*nx;
    rowEnd = nx+rowStart;
    //Find mean of the current row
    rowMean = std::accumulate(data+rowStart, data+rowEnd, 0.0) / double(nx);
    //subtract each element of the row from mean. Normalize to make the row with mean of 0
    std::transform(data+rowStart, data+rowEnd, zeroMeanVec.begin(), [&rowMean](double val){ return (val - rowMean);});
    //Find square of each element of the row
    std::transform(zeroMeanVec.begin(), zeroMeanVec.end(), elemSqrdVec.begin(), [](double val){ return (val * val);});
    //Find sample standard deviation of the row
    normFactor = std::sqrt(std::accumulate(elemSqrdVec.begin(), elemSqrdVec.end(), 0.0));
    //Normalize it so that the sum of the squares of the elements of the row is 1
    std::transform(zeroMeanVec.begin(), zeroMeanVec.end(), zeroMeanVec.begin(), [&normFactor](double val){ return (val / normFactor);});
    //Save the normalized result in a matriz of dimension ny*nx
    std::copy(zeroMeanVec.begin(), zeroMeanVec.end(), X[y].begin());
  }
  for (j=0; j<ny; ++j){
    for (i=0; i<nx; ++i){
      XT[i][j] = X[j][i];
    }
  }
  for (j=0; j<ny; ++j){           //Move on rows of X
    for (k=0; k<nx; ++k){         //Move on column of X and rows of XT
      for (i=j; i<ny; ++i){       //Move on columns of XT
        result[i + (j*ny)] += (X[j][k] * XT[k][i]);
      }
    }
  }
}

