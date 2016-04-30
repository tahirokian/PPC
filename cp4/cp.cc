#include <numeric>     // std::accumulate
#include <algorithm>   // std::transform, std::copy, std::max
#include <vector>      // std::vector
#include <cmath>       // std::sqrt
#include <iostream>
#include "cp.h"
#include "vector.h"

void correlate(int ny, int nx, const float* data, float* result){
  int i, j, k, y, ii=0, vecPerRow;
  float rowMean, normFactor;
  int rowStart, rowEnd, outRowNum0, outRowNum1, outRowNum2, jRowNum0, jRowNum1, jRowNum2, iRowNum0, iRowNum1, iRowNum2;
  std::vector<float> zeroMeanVec(nx), elemSqrdVec(nx);
  std::vector<std::vector<float>> X(ny, std::vector<float>(nx));
  float8_t* matVec;
  float8_t s[9];
  vecPerRow = std::ceil(nx/8.0);
  matVec = float8_alloc(ny * vecPerRow);
  /* Finding normalized matrix */
  for (y = 0; y < ny; ++y){
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
    std::copy(zeroMeanVec.begin(), zeroMeanVec.end(), X[y].begin());
  }
  /* Pre-processing for vector instructions */
  for (j=0; j<ny; ++j){
    for (i=0; i<nx; i+=8){
      for (k=0; k<8; ++k){
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
  #pragma omp parallel for schedule(dynamic) private(i,k,s,outRowNum0,outRowNum1,outRowNum2,jRowNum0,jRowNum1,jRowNum2,iRowNum0,iRowNum1,iRowNum2)
  for (j = 0; j < ny; j+=3){
    outRowNum0 = j * ny;
    jRowNum0 = j * vecPerRow;
    outRowNum1 = (j+1) * ny;
    jRowNum1 = (j+1) * vecPerRow;
    outRowNum2 = (j+2) * ny;
    jRowNum2 = (j+2) * vecPerRow;
    for (i = j; i < ny; i+=3){
      iRowNum0 = i * vecPerRow;
      iRowNum1 = (i+1) * vecPerRow;
      iRowNum2 = (i+2) * vecPerRow;
      for (k = 0; k < 9; ++k){
        for (int kk = 0; kk < 8; ++kk){
          s[k][kk] = 0;
        }
      }
      asm("#foo");
      for (k = 0; k < vecPerRow; ++k){
        s[0] += matVec[jRowNum0 + k] * matVec[iRowNum0 + k];
        if ((ny-i) > 1){
          s[1] += matVec[jRowNum0 + k] * matVec[iRowNum1 + k];
          s[3] += matVec[jRowNum1 + k] * matVec[iRowNum0 + k];
          s[4] += matVec[jRowNum1 + k] * matVec[iRowNum1 + k];
        }
        if ((ny-i) > 2){
          s[2] += matVec[jRowNum0 + k] * matVec[iRowNum2 + k];
          s[5] += matVec[jRowNum1 + k] * matVec[iRowNum2 + k];
          s[6] += matVec[jRowNum2 + k] * matVec[iRowNum0 + k];
          s[7] += matVec[jRowNum2 + k] * matVec[iRowNum1 + k];
          s[8] += matVec[jRowNum2 + k] * matVec[iRowNum2 + k];
        }
      }
      asm("#bar");
      result[outRowNum0 + i]   = s[0][0] + s[0][1] + s[0][2] + s[0][3] + s[0][4] + s[0][5] + s[0][6] + s[0][7];
      if ((ny-i) > 1){
        result[outRowNum0 + i+1] = s[1][0] + s[1][1] + s[1][2] + s[1][3] + s[1][4] + s[1][5] + s[1][6] + s[1][7];
        result[outRowNum1 + i]   = s[3][0] + s[3][1] + s[3][2] + s[3][3] + s[3][4] + s[3][5] + s[3][6] + s[3][7];
        result[outRowNum1 + i+1] = s[4][0] + s[4][1] + s[4][2] + s[4][3] + s[4][4] + s[4][5] + s[4][6] + s[4][7];
      }
      if ((ny-i) > 2){
        result[outRowNum0 + i+2] = s[2][0] + s[2][1] + s[2][2] + s[2][3] + s[2][4] + s[2][5] + s[2][6] + s[2][7];
        result[outRowNum1 + i+2] = s[5][0] + s[5][1] + s[5][2] + s[5][3] + s[5][4] + s[5][5] + s[5][6] + s[5][7];
        result[outRowNum2 + i]   = s[6][0] + s[6][1] + s[6][2] + s[6][3] + s[6][4] + s[6][5] + s[6][6] + s[6][7];
        result[outRowNum2 + i+1] = s[7][0] + s[7][1] + s[7][2] + s[7][3] + s[7][4] + s[7][5] + s[7][6] + s[7][7];
        result[outRowNum2 + i+2] = s[8][0] + s[8][1] + s[8][2] + s[8][3] + s[8][4] + s[8][5] + s[8][6] + s[8][7];
      }
    }
  }
  free(matVec);
}
