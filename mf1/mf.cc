#include "mf.h"
#include <vector>
#include <algorithm>    // std::max, std::min, std::nth_element

inline float median(std::vector<float> &v){
  float median;
  std::size_t n;
  if(v.empty()) {
    return 0.0;
  }
  n = v.size() / 2;
  std::nth_element(v.begin(), (v.begin()+n), v.end());
  median = v[n];
  if((v.size() % 2) == 0) {	//If even size
    auto max_it = max_element(v.begin(), v.begin()+n);
    median = (*max_it + median) / 2;
  }
  return median;
}

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {
  int yUp, yDown, xLeft, xRight;
  int x, y, i, j;
  std::vector<float> v;
  v.reserve(hx*hy);
  for (y = 0; y < ny; y++){
      yUp = std::max( (y-hy), 0 );
      yDown = std::min( (y+hy), (ny-1) );
      for (x = 0; x < nx; x++){
        xLeft = std::max( (x-hx), 0 );
        xRight = std::min( (x+hx), (nx-1) );
        v.clear();
        for (j=yUp; j<=yDown; j++){
          for (i=xLeft; i<=xRight; i++){
            v.push_back(in[i + (j*nx)]);
          }
        }
        out[x + (y*nx)] = median(v);
      }
    }
}


