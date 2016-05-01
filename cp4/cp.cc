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
  int rowStart, rowEnd;
  std::vector<float> zeroMeanVec(nx), elemSqrdVec(nx);
  std::vector<std::vector<float>> X(ny, std::vector<float>(nx));
  float8_t* matVec;
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
  /* Pre-processing - Alligning  data for vector instructions */
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
  #pragma omp parallel private(i,k)
  {
    int kk;
    int outRow0=0, outRow1=0, outRow2=0;	//outRowx point a row in result (C = A*B), three rows selected at one time
    int jRow0=0, jRow1=0, jRow2=0;		//jRowx points to a row in A (Outer loop), three rows selected at one time
    int iRow0=0, iRow1=0, iRow2=0;		//iRowx points to a row in B (Second inner loop), three rows selected at one time
    float8_t s[9];				//Stores 9 dot product results
    #pragma omp for schedule(dynamic)
    for (j = 0; j < ny; j+=3){			//Outer loop: Three rows in parallel --> increment of three
      outRow0 = j * ny;				//First row of result
      outRow1 = (j+1) * ny;			//Second row of result
      outRow2 = (j+2) * ny;			//Third row of result
      jRow0 = j * vecPerRow;			//First row poinetd by outer loop
      jRow1 = (j+1) * vecPerRow;		//Second row pointed by outer loop
      jRow2 = (j+2) * vecPerRow;		//Third row poinetd by  outer loop
      for (i = j; i < ny; i+=3){		//2nd inner loop: Three rows in parallel --> increment of three
        iRow0 = i * vecPerRow;			//First row pointed by  2nd inner loop
        iRow1 = (i+1) * vecPerRow;		//Second row pointed by  2nd inner loop
        iRow2 = (i+2) * vecPerRow;		//Third row pointed by 2nd inner loop
        for (k = 0; k < 9; ++k){
          for (kk = 0; kk < 8; ++kk){
            s[k][kk] = 0;
          }
        }
	asm("#foo");
        for (k = 0; k < vecPerRow; ++k){			//Inner most loop: Moves throufg entire row
	  s[0] += matVec[jRow0 + k] * matVec[iRow0 + k];	//A0 * B0
	  //If this is the only row left in B to be multiplied, but A has more than 1 row
          if (((ny-i) == 1) && ((ny-j) != 1)){
            s[3] += matVec[jRow1 + k] * matVec[iRow0 + k];	//A1 * B0
            s[6] += matVec[jRow2 + k] * matVec[iRow0 + k];	//A2 * B0
	    asm("#dummy1");
          }
	  //If B has 2 rows left to be multiplied and A has two rows too
          if (((ny-i) > 1) && ((ny-j) == 2)){
            s[1] += matVec[jRow0 + k] * matVec[iRow1 + k];	//A0 * B1
            s[3] += matVec[jRow1 + k] * matVec[iRow0 + k];	//A1 * B0
            s[4] += matVec[jRow1 + k] * matVec[iRow1 + k];	//A1 * B1
	    asm("#dummy2");
          }
	  //If B has two rows left to be multiplied and A has atleast 3
          if (((ny-i) > 1) && ((ny-j) > 2)){
            s[1] += matVec[jRow0 + k] * matVec[iRow1 + k];	//A0 * B1
            s[3] += matVec[jRow1 + k] * matVec[iRow0 + k];	//A1 * B0
            s[4] += matVec[jRow1 + k] * matVec[iRow1 + k];	//A1 * B1
            s[6] += matVec[jRow2 + k] * matVec[iRow0 + k];	//A2 * B0
            s[7] += matVec[jRow2 + k] * matVec[iRow1 + k];	//A2 * B1
	    asm("#dummy3");
          }
	  //If both A and B have atleast 3 rows each to be multiplied
          if ((ny-i) > 2){
            s[2] += matVec[jRow0 + k] * matVec[iRow2 + k];	//A0 * B2
            s[5] += matVec[jRow1 + k] * matVec[iRow2 + k];	//A1 * B2
            s[8] += matVec[jRow2 + k] * matVec[iRow2 + k];	//A2 * B2
	    asm("#dummy4");
          }
        }
        asm("#bar");
        result[outRow0 + i] = s[0][0] + s[0][1] + s[0][2] + s[0][3] + s[0][4] + s[0][5] + s[0][6] + s[0][7];
        if ((ny-i) == 1 && ((ny-j) != 1)){
          result[outRow1 + i] = s[3][0] + s[3][1] + s[3][2] + s[3][3] + s[3][4] + s[3][5] + s[3][6] + s[3][7];
          result[outRow2 + i] = s[6][0] + s[6][1] + s[6][2] + s[6][3] + s[6][4] + s[6][5] + s[6][6] + s[6][7];
	  asm("#dummy5");
        }
        if (((ny-i) > 1) && ((ny-j) == 2)){
          result[outRow0 + i+1] = s[1][0] + s[1][1] + s[1][2] + s[1][3] + s[1][4] + s[1][5] + s[1][6] + s[1][7];
          result[outRow1 + i]   = s[3][0] + s[3][1] + s[3][2] + s[3][3] + s[3][4] + s[3][5] + s[3][6] + s[3][7];
          result[outRow1 + i+1] = s[4][0] + s[4][1] + s[4][2] + s[4][3] + s[4][4] + s[4][5] + s[4][6] + s[4][7];
	  asm("#dummy6");
        }
        if (((ny-i) > 1) && ((ny-j) > 2)){
          result[outRow0 + i+1] = s[1][0] + s[1][1] + s[1][2] + s[1][3] + s[1][4] + s[1][5] + s[1][6] + s[1][7];
          result[outRow1 + i]   = s[3][0] + s[3][1] + s[3][2] + s[3][3] + s[3][4] + s[3][5] + s[3][6] + s[3][7];
          result[outRow1 + i+1] = s[4][0] + s[4][1] + s[4][2] + s[4][3] + s[4][4] + s[4][5] + s[4][6] + s[4][7];
          result[outRow2 + i]   = s[6][0] + s[6][1] + s[6][2] + s[6][3] + s[6][4] + s[6][5] + s[6][6] + s[6][7];
          result[outRow2 + i+1] = s[7][0] + s[7][1] + s[7][2] + s[7][3] + s[7][4] + s[7][5] + s[7][6] + s[7][7];
	  asm("#dummy7");
        }
        if ((ny-i) > 2){
          result[outRow0 + i+2] = s[2][0] + s[2][1] + s[2][2] + s[2][3] + s[2][4] + s[2][5] + s[2][6] + s[2][7];
          result[outRow1 + i+2] = s[5][0] + s[5][1] + s[5][2] + s[5][3] + s[5][4] + s[5][5] + s[5][6] + s[5][7];
          result[outRow2 + i+2] = s[8][0] + s[8][1] + s[8][2] + s[8][3] + s[8][4] + s[8][5] + s[8][6] + s[8][7];
	  asm("#dummy8");
        }
      }
    }
  }
  free(matVec);
}
