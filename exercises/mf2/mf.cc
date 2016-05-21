#include "mf.h"
#include <vector>       // std::vector
#include <algorithm>    // std::max, std::min, std::nth_element

inline float median(std::vector<float> &v){
  float median;
  std::size_t n = v.size() / 2;
  std::nth_element(v.begin(), (v.begin()+n), v.end());
  median = v[n];
  if((v.size() % 2) == 0) {     //If even size
    auto max_it = max_element(v.begin(), v.begin()+n);
    median = (*max_it + median) / 2;
  }
  return median;
}

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {
  int yUp, yDown, xLeft, xRight;
  int x, y, i, j, k;
  int nxLimit = nx-1;
  int nyLimit = ny-1;
  int loopLength = ny*nx;
  int rowNumber;
  std::vector<float> v(hx*hy);
  #pragma omp parallel for schedule(static,1) private(y,x,i,j,v,xLeft,xRight,yUp,yDown,rowNumber)
  for(k = 0; k < loopLength; ++k){
    y = k/nx;
    x = k%nx;
    yUp = std::max((y-hy), 0);
    yDown = std::min((y+hy), nyLimit);
    xLeft = std::max((x-hx), 0);
    xRight = std::min((x+hx), nxLimit);
    v.clear();
    for(j=yUp; j<=yDown; ++j){
      rowNumber = j*nx;
      for(i=xLeft; i<=xRight; ++i){
        v.push_back(in[i + rowNumber]);
      }
    }
    out[x + (y*nx)] = median(v);
  }
}
