#include "so.h"
#include <algorithm>	//std::sort
#include <cmath>	//std::ceil
#include <omp.h>
#include <iostream>

void psort(int n, data_t* data){
  int maxThreads = omp_get_max_threads();
  int parts = std::ceil(n/maxThreads);
  #pragma omp parallel for
  for (int i=0; i<maxThreads; ++i){
    if (n > ((i+1)*parts)){
      std::sort(data+(i*parts), data+((i+1)*parts));
    }
    else{
      std::sort(data+(i*parts), data+n);
      asm("#comment 1");
    }
  }
  while (maxThreads > 1){
    #pragma omp parallel for
    for (int j=0; j<maxThreads; j+=2){
      if (n > ((j+2)*parts)){
        std::inplace_merge(data+(j*parts), data+((j+1)*parts), data+((j+2)*parts));
      }
      else{
        std::inplace_merge(data+(j*parts), data+((j+1)*parts), data+n);
	asm("#comment 2");
      }
    }
    maxThreads /= 2;
    parts *= 2;
  }
}
