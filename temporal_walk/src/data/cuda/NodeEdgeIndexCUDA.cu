#include "NodeEdgeIndexCUDA.cuh"

#ifdef HAS_CUDA


template class NodeEdgeIndexCUDA<false>;
template class NodeEdgeIndexCUDA<true>;
#endif
