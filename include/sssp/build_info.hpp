#pragma once

namespace sssp {

// Set by the Makefile when build_info.o is compiled.
const char* build_git_sha();
bool build_git_dirty();
const char* build_diff_hash();
const char* build_flags();

}  // namespace sssp
