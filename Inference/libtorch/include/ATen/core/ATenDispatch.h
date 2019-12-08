#pragma once

#include <c10/core/TensorTypeSet.h>
#include <c10/core/Backend.h>
#include <c10/core/impl/LocalTensorTypeSet.h>
#include <unordered_map>
#include <unordered_set>
#include <ATen/core/OpsAlreadyMovedToC10.h>
#include <ATen/core/Variadic.h>
#include <ATen/core/TensorBody.h>
#include <c10/util/C++17.h>
#include <memory>
#include <mutex>
#include <ATen/core/interned_strings.h>
#include <ATen/core/stack.h>

// TODO: Rewrite this comment
//
// This dispatch class serves as a replacement for our previous dispatch
// mechanism, in which all functions were members of a Type class. A derived
// class existed for each backend (and Variable), and the vtable was used to
// dispatch to the correct implementation. This class is to be replaced by
// the c10 dispatcher when it supports all argument and return types.
// This implementation opts to store implementations in a table of void*.

namespace at {

namespace impl {

// Take a TensorTypeSet for a Tensor, and combine it with the current thread
// local valid (implemented) and enabled (not implemented) TensorTypeSets
// to determine what the actual dispatch TensorTypeId should be.  Unlike
// Tensor::type_set(), the value of this on a tensor can change depending
// on TLS.
//
// NB: I didn't make this take a Tensor to avoid header include shenanigans.
//
// TODO: I'm not sure if this should live in this header or not; the operant
// question is whether or not we have access to all the relevant TLS at this
// point.
static inline TensorTypeId dispatchTypeId(TensorTypeSet ts) {
  return (ts - c10::impl::tls_excluded_tensor_type_set()).highestPriorityTypeId();
}

}

namespace detail {
  struct MultiDispatchTensorTypeSet : IterArgs<MultiDispatchTensorTypeSet> {
    TensorTypeSet ts;
    void operator()(const at::Tensor& x) {
      ts = ts | x.type_set();
    }
    void operator()(TensorOptions x) {
      ts = ts | x.type_set();
    }
    void operator()(at::ArrayRef<at::Tensor> xs) {
      for (const auto& x : xs) {
        ts = ts | x.type_set();
      }
    }
    template <typename T>
    void operator()(const T& x) {
      // do nothing
    }
  };

  template <typename... Args>
  TensorTypeSet multi_dispatch_tensor_type_set(const Args&... args) {
    return MultiDispatchTensorTypeSet().apply(args...).ts;
  }
}

// ATenOpTable stores the implementations for each backend, in addition to
// an implementation for variables.
class CAFFE2_API ATenOpTable {
 public:
  ATenOpTable(std::string schema)
    : schema_(std::move(schema)) {}

  template<class Result, class... Args>
  Result callUnboxed(Args... args) const {
    using FuncType = Result(Args...);
    TensorTypeSet ts = detail::multi_dispatch_tensor_type_set(args...);
    TensorTypeId tid = impl::dispatchTypeId(ts);

    // You might think we can eliminate the second branch by maintaining a
    // bitmask of registered operator keys, so we don't select dispatch ids
    // which don't have implementations here.  But the net effect is that if you
    // get a Variable CPUTensor, if there is no variable registration, you'll
    // fall back to the CPU implementation.  Is this what you want?  Unlikely...

    auto* unboxed_fn = reinterpret_cast<FuncType*>(function_table_[static_cast<int64_t>(tid)]);
    if (C10_LIKELY(unboxed_fn != nullptr)) {
      return (*unboxed_fn)(args...);
    }

    auto* unboxed_fallback_fn = reinterpret_cast<FuncType*>(function_table_[static_cast<int64_t>(TensorTypeId::UndefinedTensorId)]);
    if (C10_LIKELY(unboxed_fallback_fn != nullptr)) {
      return (*unboxed_fallback_fn)(args...);
    }

    reportError(tid);
    TORCH_INTERNAL_ASSERT(0);
  }

 private:
  void registerOp(TensorTypeId tid, void* fn) {
    TORCH_CHECK(function_table_[static_cast<int64_t>(tid)] == nullptr,
        "Attempting to register function for schema ", schema_,
        " and tensor type ", toString(tid),
        " but there is already a function registered");
    function_table_[static_cast<int64_t>(tid)] = fn;
  }

  C10_NORETURN void reportError(TensorTypeId tid) const;

  friend class ATenDispatch;

  std::string schema_;
  void* function_table_[static_cast<int64_t>(TensorTypeId::NumTensorIds)] = {nullptr};
};

class CAFFE2_API ATenDispatch {
 public:
  template<class FuncType>
  ATenDispatch& registerOp(TensorTypeId id, const char* schema, FuncType* fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (op_tables_.find(schema) == op_tables_.end()) {
      op_tables_.insert(std::make_pair(schema, ATenOpTable(schema)));
    }
    op_tables_.at(schema).registerOp(id, reinterpret_cast<void*>(fn));
    return *this;
  }

  const ATenOpTable* getOpTable(const char* schema) const {
    auto iter = op_tables_.find(schema);
    TORCH_CHECK(iter != op_tables_.end(),
        "No functions are registered for schema ", schema);
    return &iter->second;
  }

 private:
  std::unordered_map<std::string, ATenOpTable> op_tables_;
  std::mutex mutex_;
};

CAFFE2_API ATenDispatch& globalATenDispatch();

} // namespace at
