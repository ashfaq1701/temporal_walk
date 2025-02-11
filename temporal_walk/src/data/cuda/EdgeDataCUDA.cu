#include "EdgeDataCUDA.cuh"

#ifdef HAS_CUDA


template class EdgeDataCUDA<false>;
template class EdgeDataCUDA<true>;
#endif
