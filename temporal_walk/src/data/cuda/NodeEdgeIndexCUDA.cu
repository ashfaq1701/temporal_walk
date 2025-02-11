#include "NodeEdgeIndexCUDA.cuh"

#ifdef USE_CUDA


template class NodeEdgeIndexCUDA<false>;
template class NodeEdgeIndexCUDA<true>;
#endif
