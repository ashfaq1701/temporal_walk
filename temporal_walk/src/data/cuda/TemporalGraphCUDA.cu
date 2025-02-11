#include "TemporalGraphCUDA.cuh"

#ifdef USE_CUDA


template class TemporalGraphCUDA<false>;
template class TemporalGraphCUDA<true>;
#endif
