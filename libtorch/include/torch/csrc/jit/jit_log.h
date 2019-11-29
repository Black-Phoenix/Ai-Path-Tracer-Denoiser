#pragma once
#include <memory>
#include <string>
#include <torch/csrc/WindowsTorchApiMacro.h>

// `TorchScript` offers a simple logging facility that can enabled by setting an
// environment variable `PYTORCH_JIT_LOG_LEVEL`.

// Logging is enabled on a per file basis. To enable logging in
// `dead_code_elimination.cpp`, `PYTORCH_JIT_LOG_LEVEL` should be
// set to `dead_code_elimination.cpp` or, simply, to `dead_code_elimination`
// (i.e. `PYTORCH_JIT_LOG_LEVEL=dead_code_elimination`).

// Multiple files can be logged by separating each file name with a colon `:` as
// in the following example,
// `PYTORCH_JIT_LOG_LEVEL=dead_code_elimination:guard_elimination`

// There are 3 logging levels available for your use ordered by the detail level
// from lowest to highest.

// * `GRAPH_DUMP` should be used for printing entire graphs after optimization
// passes
// * `GRAPH_UPDATE` should be used for reporting graph transformations (i.e.
// node deletion, constant folding, etc)
// * `GRAPH_DEBUG` should be used for providing information useful for debugging
//   the internals of a particular optimization pass or analysis

// The current logging level is `GRAPH_UPDATE` meaning that both `GRAPH_DUMP`
// and `GRAPH_UPDATE` will be enabled when
// one specifies a file(s) in `PYTORCH_JIT_LOG_LEVEL`.

// `GRAPH_DEBUG` can be enabled by prefixing a file name with an `>` as in
// `>alias_analysis`.
// `>>` and `>>>` are also valid and **currently** are equivalent to
// `GRAPH_DEBUG` as there is no logging level that is
// higher than `GRAPH_DEBUG`.

namespace torch {
namespace jit {

struct Node;
struct Graph;

enum class JitLoggingLevels {
  GRAPH_DUMP = 0,
  GRAPH_UPDATE,
  GRAPH_DEBUG,
};

std::string TORCH_API getHeader(const Node *node);

std::string TORCH_API log_function(const std::shared_ptr<Graph> &graph);

TORCH_API JitLoggingLevels jit_log_level();

// Prefix every line in a multiline string \p IN_STR with \p PREFIX.
TORCH_API std::string jit_log_prefix(
    const std::string& prefix,
    const std::string& in_str);

TORCH_API std::string jit_log_prefix(
    JitLoggingLevels level,
    const char* fn,
    int l,
    const std::string& in_str);

TORCH_API bool is_enabled(const char *cfname, JitLoggingLevels level);

TORCH_API std::ostream& operator<<(std::ostream& out, JitLoggingLevels level);

#define JIT_LOG(level, ...)                                                    \
  if (is_enabled(__FILE__, level)) {                                           \
    std::cerr << jit_log_prefix(level, __FILE__, __LINE__,                     \
                                ::c10::str(__VA_ARGS__));                      \
  }

// tries to reconstruct original python source
#define SOURCE_DUMP(MSG, G)                                                    \
  JIT_LOG(JitLoggingLevels::GRAPH_DUMP, MSG, "\n", log_function(G));
// use GRAPH_DUMP for dumping graphs after optimization passes
#define GRAPH_DUMP(MSG, G) \
  JIT_LOG(JitLoggingLevels::GRAPH_DUMP, MSG, "\n", (G)->toString());
// use GRAPH_UPDATE for reporting graph transformations (i.e. node deletion,
// constant folding, CSE)
#define GRAPH_UPDATE(...) JIT_LOG(JitLoggingLevels::GRAPH_UPDATE, __VA_ARGS__);
// use GRAPH_DEBUG to provide information useful for debugging a particular opt
// pass
#define GRAPH_DEBUG(...) JIT_LOG(JitLoggingLevels::GRAPH_DEBUG, __VA_ARGS__);
} // namespace jit
} // namespace torch
