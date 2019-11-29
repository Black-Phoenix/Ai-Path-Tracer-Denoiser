#pragma once
#include <torch/csrc/jit/graph_executor.h>

#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/autodiff.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/profiling_record.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/tracer.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/logging.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

void packGradient(const Gradient& gradient, Node* dnode);
bool needsGradient(const std::shared_ptr<const Graph>& graph);
void runOptimization(std::shared_ptr<Graph>& graph);
void runNondiffOptimization(std::shared_ptr<Graph>& graph);
void debugSetAutodiffSubgraphInlining(bool state);
bool getAutodiffSubgraphInlining();

// Tunable parameters for deciding when to create/keep subgraphs of
// differentiable code
const size_t autodiffSubgraphNodeThreshold = 2;
const size_t autodiffSubgraphInlineThreshold = 5;

// a Graph can be created via tracing, or via a language-based frontend
// GraphExecutor runs it. It can run the same graph on many different sizes
// and different requires_grad states, and handles specializations for each
// situation. GraphExecutor is completely unaware of tracing or module
// parameters to keep the tracing concerns separated.
struct GraphExecutorImplBase {
  static std::shared_ptr<Graph> prepareGraph(
      const std::shared_ptr<Graph>& graph) {
    auto copy = graph->copy();
    EraseShapeInformation(copy);
    return copy;
  }

  GraphExecutorImplBase(const std::shared_ptr<Graph>& graph)
      : graph(prepareGraph(graph)),
        num_inputs(this->graph->inputs().size()),
        num_outputs(this->graph->outputs().size()) {}

  // entry point where execution begins
  void run(Stack& stack);

  virtual ExecutionPlan getPlanFor(Stack& stack) = 0;
  virtual GraphExecutorState getDebugState() = 0;
  virtual ~GraphExecutorImplBase() = default;

 protected:
  friend struct GraphExecutor;

  // The unoptimized starting graph. This field is effectively const, but we
  // can't make it so because Graph::copy() is not const (and making it const is
  // not that easy at this point).
  std::shared_ptr<Graph> graph;

  // If false, we'll run the graph as we get it, without any optimizations.
  // Useful for debugging.
  const size_t num_inputs;
  const size_t num_outputs;

  // GraphExecutors can be accessed from multiple threads, so this thread needs
  // to be held every time we access the fallback or plan_cache.
  std::mutex compile_mutex;
};

} // namespace jit
} // namespace torch
