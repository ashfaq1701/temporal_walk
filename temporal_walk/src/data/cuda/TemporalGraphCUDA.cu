#include "TemporalGraphCUDA.cuh"

#ifdef HAS_CUDA


template class TemporalGraphCUDA<false>;
template class TemporalGraphCUDA<true>;
#endif
