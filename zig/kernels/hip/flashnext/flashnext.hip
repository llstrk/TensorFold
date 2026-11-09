// Flash Next's own kernels as one code object: residual streams, n-gram hashes, the indexer and sparse attention.

#include "streams.hpp"
#include "ngram.hpp"
#include "indexer.hpp"
#include "sparse.hpp"
#include "sparse_lanes.hpp"
#include "sparse_wmma.hpp"
#include "indexer_wmma.hpp"
