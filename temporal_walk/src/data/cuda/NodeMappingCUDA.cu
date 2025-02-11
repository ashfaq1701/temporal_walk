#include "NodeMappingCUDA.cuh"

#ifdef HAS_CUDA


template class NodeMappingCUDA<false>;
template class NodeMappingCUDA<true>;
#endif
