#include "sssp/build_info.hpp"

#ifndef SSSP_GIT_SHA
#define SSSP_GIT_SHA "unknown"
#endif
#ifndef SSSP_GIT_DIRTY
#define SSSP_GIT_DIRTY 1
#endif
#ifndef SSSP_DIFF_HASH
#define SSSP_DIFF_HASH ""
#endif
#ifndef SSSP_BUILD_FLAGS
#define SSSP_BUILD_FLAGS ""
#endif

namespace sssp {

const char* build_git_sha() { return SSSP_GIT_SHA; }
bool build_git_dirty() { return SSSP_GIT_DIRTY != 0; }
const char* build_diff_hash() { return SSSP_DIFF_HASH; }
const char* build_flags() { return SSSP_BUILD_FLAGS; }

}  // namespace sssp
