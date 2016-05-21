#include <numeric>     // std::accumulate
#include <algorithm>   // std::transform, std::copy
#include <vector>      // std::vector
#include <cmath>       // std::sqrt, std::ceil
#include "cp.h"
#include "vector.h"

void correlate(int ny, int nx, const float* data, float* result){
  int i, j, k, y, ii=0, vecPerRow;
  double rowMean, normFactor;
  int rowStart, rowEnd, rowNum, jRowNum, iRowNum;
  std::vector<double> zeroMeanVec(nx), elemSqrdVec(nx);
  std::vector<std::vector<double>> X(ny, std::vector<double>(nx));
  double4_t* matVec;
  double4_t s;
  vecPerRow = std::ceil(nx/4.0);
  matVec = double4_alloc(ny * vecPerRow);
  /* Finding normalized matrix */
  for (y = 0; y < ny; ++y){
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
  /* Pre-processing - Aliging data to use in vector instructions */
  for (j=0; j<ny; ++j){
    for (i=0; i<nx; i+=4){
      for (k=0; k<4; ++k){
        if ( (i+k) < nx ){
          matVec[ii][k] = X[j][i+k];
        }
        else{
          matVec[ii][k] = 0;
        }
      }
      ++ii;
    }
  }
  #pragma omp parallel for schedule(dynamic) private(i, k, s, rowNum, jRowNum, iRowNum)
  for (j = 0; j < ny; ++j){
    rowNum = j * ny;
    jRowNum = j * vecPerRow;
    for (i = j; i < ny; ++i){
      iRowNum = i * vecPerRow;
      s[0] = 0; s[1] = 0; s[2] = 0; s[3] = 0;
      asm("#foo");
      for (k = 0; k < vecPerRow; ++k){
        s += matVec[jRowNum + k] * matVec[iRowNum + k];
      }
      asm("#bar");
      result[rowNum + i] = s[0] + s[1] + s[2] + s[3];
    }
  }
  free(matVec);
}
