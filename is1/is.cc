#include "is.h"
#include "vector.h"

Result segment(int ny, int nx, const float* data){
  Result result { ny/3, nx/3, 2*ny/3, 2*nx/3, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f} };
  int totalPixels = ny * nx;
  double4_t* pixelVec = double4_alloc((nx+1)*(ny+1));
  double4_t* sumVec = double4_alloc((nx+1)*(ny+1));
  double4_t totalSum;
  double maxVal=-1;
  int rowLen = nx + 1;
  for (int i = 0; i <= ny; ++i){
    for (int j = 0; j <= nx; ++j){
      pixelVec[(i*rowLen)+j] = double4_0;
      sumVec[(i*rowLen)+j] = double4_0;
      if (i > 0 && j > 0){
        for (int k = 0; k < 3; ++k){
          pixelVec[(i*rowLen)+j][k] = data[(3*(i-1)*nx) + 3*(j-1) + k];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int i = 1; i <= ny; ++i){
    for (int j = 1; j <= nx; ++j){
      sumVec[(i*rowLen)+j] = pixelVec[(i*rowLen)+j] + sumVec[(i*rowLen)+(j-1)] + sumVec[((i-1)*rowLen)+j] - sumVec[((i-1)*rowLen)+(j-1)];
    }
  }
  totalSum = sumVec[rowLen*(ny+1)-1];
  #pragma omp parallel
  {
    int tempk = 0, templ = 0;
    int x = 0, y = 0;
    double xInv = 0, yInv = 0, maxTemp1=0, maxTemp2=-1;
    double4_t tempVxc=double4_0, tempVyc=double4_0;
    double4_t Vxc=double4_0, Vyc=double4_0, Hxy=double4_0;
    #pragma omp for schedule(dynamic)
    for (int i = 1; i <= ny; ++i){
      for (int j = 1; j <= nx; ++j){
        x = i*j;
        xInv = 1.0/x;
        y = totalPixels - x;
        yInv = 1.0/y;
        for (int k = 0; k <= ny-i; ++k){
          for (int l = 0; l <= nx-j; ++l){
            tempVxc = sumVec[(k+i)*rowLen+(l+j)] - sumVec[(k+i)*rowLen+l] - sumVec[k*rowLen+(l+j)] + sumVec[k*rowLen+l];
            tempVyc = totalSum - tempVxc;
            Hxy = (tempVxc * tempVxc * xInv) + (tempVyc * tempVyc * yInv);
            maxTemp1 = Hxy[0] + Hxy[1] + Hxy[2];
            if (maxTemp1 > maxTemp2){
              maxTemp2 = maxTemp1;
              tempk = k;
              templ = l;
              Vxc = tempVxc;
              Vyc = tempVyc;
            }
          }
        }
        #pragma omp critical
        if (maxTemp2 > maxVal){
          maxVal = maxTemp2;
          result.x0 = templ;
          result.y0 = tempk;
          result.x1 = templ+j;
          result.y1 = tempk+i;
          result.inner[0] = Vxc[0]*xInv;
          result.inner[1] = Vxc[1]*xInv;
          result.inner[2] = Vxc[2]*xInv;
          result.outer[0] = Vyc[0]*yInv;
          result.outer[1] = Vyc[1]*yInv;
          result.outer[2] = Vyc[2]*yInv;
        }
      }
    } 
  }
  return result;
}
