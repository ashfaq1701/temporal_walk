#include "NodeMappingCUDA.cuh"

#ifdef USE_CUDA


template class NodeMappingCUDA<false>;
template class NodeMappingCUDA<true>;
#endif
