#include "EdgeDataCUDA.cuh"

#ifdef USE_CUDA


template class EdgeDataCUDA<false>;
template class EdgeDataCUDA<true>;
#endif
