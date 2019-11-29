#ifndef C10_UTIL_EXCEPTION_H_
#define C10_UTIL_EXCEPTION_H_

#include "c10/macros/Macros.h"
#include "c10/util/StringUtil.h"
#include "c10/util/Deprecated.h"

#include <cstddef>
#include <exception>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

namespace c10 {

/// The primary ATen error class.
/// Provides a complete error message with source location information via
/// `what()`, and a more concise message via `what_without_backtrace()`.
/// Don't throw this directly; use TORCH_CHECK/TORCH_INTERNAL_ASSERT instead.
///
/// NB: c10::Error is handled specially by the default torch to suppress the
/// backtrace, see torch/csrc/Exceptions.h
class C10_API Error : public std::exception {
  std::vector<std::string> msg_stack_;
  std::string backtrace_;

  // These two are derived fields from msg_stack_ and backtrace_, but we need
  // fields for the strings so that we can return a const char* (as the
  // signature of std::exception requires).
  std::string msg_;
  std::string msg_without_backtrace_;

  // This is a little debugging trick: you can stash a relevant pointer
  // in caller, and then when you catch the exception, you can compare
  // against pointers you have on hand to get more information about
  // where the exception came from.  In Caffe2, this is used to figure
  // out which operator raised an exception.
  const void* caller_;

 public:
  Error(
      const std::string& msg,
      const std::string& backtrace,
      const void* caller = nullptr);
  Error(SourceLocation source_location, const std::string& msg);
  Error(
      const char* file,
      const uint32_t line,
      const char* condition,
      const std::string& msg,
      const std::string& backtrace,
      const void* caller = nullptr);

  void AppendMessage(const std::string& msg);

  // Compute the full message from msg_ and msg_without_backtrace_
  // TODO: Maybe this should be private
  std::string msg() const;
  std::string msg_without_backtrace() const;

  const std::vector<std::string>& msg_stack() const {
    return msg_stack_;
  }

  /// Returns the complete error message, including the source location.
  const char* what() const noexcept override {
    return msg_.c_str();
  }

  const void* caller() const noexcept {
    return caller_;
  }

  /// Returns only the error message string, without source location.
  const char* what_without_backtrace() const noexcept {
    return msg_without_backtrace_.c_str();
  }
};

class C10_API Warning {
  using handler_t =
      void (*)(const SourceLocation& source_location, const char* msg);

 public:
  /// Issue a warning with a given message. Dispatched to the current
  /// warning handler.
  static void warn(SourceLocation source_location, std::string msg);
  /// Sets the global warning handler. This is not thread-safe, so it should
  /// generally be called once during initialization.
  static void set_warning_handler(handler_t handler);
  /// The default warning handler. Prints the message to stderr.
  static void print_warning(
      const SourceLocation& source_location,
      const char* msg);

 private:
  static handler_t warning_handler_;
};

// Used in ATen for out-of-bound indices that can reasonably only be detected
// lazily inside a kernel (See: advanced indexing).  These turn into
// IndexError when they cross to Python.
class C10_API IndexError : public Error {
  using Error::Error;
};


// A utility function to return an exception std::string by prepending its
// exception type before its what() content
C10_API std::string GetExceptionString(const std::exception& e);

namespace detail {

// Return x if it is non-empty; otherwise return y.
inline std::string if_empty_then(std::string x, std::string y) {
  if (x.empty()) {
    return y;
  } else {
    return x;
  }
}

}


} // namespace c10

// Private helper macro for implementing TORCH_INTERNAL_ASSERT and TORCH_CHECK
//
// Note: In the debug build With MSVC, __LINE__ might be of long type (a.k.a int32_t),
// which is different from the definition of `SourceLocation` that requires
// unsigned int (a.k.a uint32_t) and may cause a compile error with the message:
// error C2397: conversion from 'long' to 'uint32_t' requires a narrowing conversion
// Here the static cast is used to pass the build.
// if this is used inside a lambda the __func__ macro expands to operator(),
// which isn't very useful, but hard to fix in a macro so suppressing the warning.
#define C10_THROW_ERROR(err_type, msg) \
  throw ::c10::err_type({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)

// Private helper macro for workaround MSVC misexpansion of nested macro
// invocations involving __VA_ARGS__.  See
// https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
#define C10_EXPAND_MSVC_WORKAROUND(x) x

// On nvcc, C10_UNLIKELY thwarts missing return statement analysis.  In cases
// where the unlikely expression may be a constant, use this macro to ensure
// return statement analysis keeps working (at the cost of not getting the
// likely/unlikely annotation on nvcc). https://github.com/pytorch/pytorch/issues/21418
//
// Currently, this is only used in the error reporting macros below.  If you
// want to use it more generally, move me to Macros.h
//
// TODO: Brian Vaughan observed that we might be able to get this to work on nvcc
// by writing some sort of C++ overload that distinguishes constexpr inputs
// from non-constexpr.  Since there isn't any evidence that losing C10_UNLIKELY
// in nvcc is causing us perf problems, this is not yet implemented, but this
// might be an interesting piece of C++ code for an intrepid bootcamper to
// write.
#if defined(__CUDACC__)
#define C10_UNLIKELY_OR_CONST(e) e
#else
#define C10_UNLIKELY_OR_CONST(e) C10_UNLIKELY(e)
#endif


// ----------------------------------------------------------------------------
// Error reporting macros
// ----------------------------------------------------------------------------

// A utility macro to provide assert()-like functionality; that is, enforcement
// of internal invariants in code.  It supports an arbitrary number of extra
// arguments (evaluated only on failure), which will be printed in the assert
// failure message using operator<< (this is useful to print some variables
// which may be useful for debugging.)
//
// Usage:
//    TORCH_INTERNAL_ASSERT(should_be_true);
//    TORCH_INTERNAL_ASSERT(x == 0, "x = ", x);
//
// Assuming no bugs in PyTorch, the conditions tested by this macro should
// always be true; e.g., it should be possible to disable all of these
// conditions without changing observable user behavior.  If you would like to
// do error reporting for user input, please use TORCH_CHECK instead.
//
// NOTE: It is SAFE to use this macro in production code; on failure, this
// simply raises an exception, it does NOT unceremoniously quit the process
// (unlike assert()).
//
#ifdef C10_MOBILE
#define TORCH_INTERNAL_ASSERT(cond, ...)      \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {       \
    C10_THROW_ERROR(Error,                    \
        #cond " INTERNAL ASSERT FAILED at"    \
        __FILE__                              \
    );                                        \
  }
#else
#define TORCH_INTERNAL_ASSERT(cond, ...)      \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {       \
    C10_THROW_ERROR(Error, ::c10::str(        \
        #cond " INTERNAL ASSERT FAILED at ",  \
        __FILE__,                             \
        ":",                                  \
        __LINE__,                             \
        ", please report a bug to PyTorch. ", \
        ::c10::str(__VA_ARGS__)               \
    ));                                       \
  }
#endif

// A utility macro to make it easier to test for error conditions from user
// input.  Like TORCH_INTERNAL_ASSERT, it supports an arbitrary number of extra
// arguments (evaluated only on failure), which will be printed in the error
// message using operator<< (e.g., you can pass any object which has
// operator<< defined.  Most objects in PyTorch have these definitions!)
//
// Usage:
//    TORCH_CHECK(should_be_true); // A default error message will be provided
//                                 // in this case; but we recommend writing an
//                                 // explicit error message, as it is more
//                                 // user friendly.
//    TORCH_CHECK(x == 0, "Expected x to be 0, but got ", x);
//
// On failure, this macro will raise an exception.  If this exception propagates
// to Python, it will convert into a Python RuntimeError.
//
// NOTE: It is SAFE to use this macro in production code; on failure, this
// simply raises an exception, it does NOT unceremoniously quit the process
// (unlike CHECK() from glog.)
//
#ifdef C10_MOBILE
#define TORCH_CHECK(cond, ...)                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {       \
    C10_THROW_ERROR(Error,                    \
        #cond " CHECK FAILED at "             \
        __FILE__                              \
    );                                        \
  }
#else
#define TORCH_CHECK(cond, ...)                              \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                     \
    C10_THROW_ERROR(Error,                                  \
      ::c10::detail::if_empty_then(                         \
        ::c10::str(__VA_ARGS__),                            \
        "Expected " #cond " to be true, but got false.  "   \
        "(Could this error message be improved?  If so, "   \
        "please report an enhancement request to PyTorch.)" \
      )                                                     \
    );                                                      \
  }
#endif
// TODO: We're going to get a lot of similar looking string literals
// this way; check if this actually affects binary size.

// Like TORCH_CHECK, but raises IndexErrors instead of Errors.
#ifdef C10_MOBILE
#define TORCH_CHECK_INDEX(cond, ...)          \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {       \
    C10_THROW_ERROR(Error,                    \
        #cond " INDEX CHECK FAILED at "       \
        __FILE__                              \
    );                                        \
  }
#else
#define TORCH_CHECK_INDEX(cond, ...)                        \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                     \
    C10_THROW_ERROR(IndexError,                             \
      ::c10::detail::if_empty_then(                         \
        ::c10::str(__VA_ARGS__),                            \
        "Expected " #cond " to be true, but got false.  "   \
        "(Could this error message be improved?  If so, "   \
        "please report an enhancement request to PyTorch.)" \
      )                                                     \
    );                                                      \
  }
#endif


// Report a warning to the user.  Accepts an arbitrary number of extra
// arguments which are concatenated into the warning message using operator<<
//
#define TORCH_WARN(...) \
  ::c10::Warning::warn({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, ::c10::str(__VA_ARGS__))

// Report a warning to the user only once.  Accepts an arbitrary number of extra
// arguments which are concatenated into the warning message using operator<<
//
#define TORCH_WARN_ONCE(...) \
  C10_UNUSED static const auto C10_ANONYMOUS_VARIABLE(torch_warn_once_) = [&] { \
    ::c10::Warning::warn({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, ::c10::str(__VA_ARGS__)); \
    return true; \
  }()


// ----------------------------------------------------------------------------
// Deprecated macros
// ----------------------------------------------------------------------------

namespace c10 { namespace detail {

/*
// Deprecation disabled until we fix sites in our codebase
C10_DEPRECATED_MESSAGE("AT_ERROR(msg) is deprecated, use TORCH_CHECK(false, msg) instead.")
*/
inline void deprecated_AT_ERROR() {}

/*
// Deprecation disabled until we fix sites in our codebase
C10_DEPRECATED_MESSAGE("AT_INDEX_ERROR(msg) is deprecated, use TORCH_CHECK_INDEX(false, msg) instead.")
*/
inline void deprecated_AT_INDEX_ERROR() {}

/*
// Deprecation disabled until we fix sites in our codebase
C10_DEPRECATED_MESSAGE("AT_WARN is deprecated, use TORCH_WARN instead.")
*/
inline void deprecated_AT_WARN() {}

C10_DEPRECATED_MESSAGE("AT_CHECK is deprecated, use TORCH_CHECK instead.")
inline void deprecated_AT_CHECK() {}

/*
// Deprecation disabled until we fix sites in our codebase
C10_DEPRECATED_MESSAGE("AT_ASSERT is deprecated, if you mean to indicate an internal invariant failure, use " \
                       "TORCH_INTERNAL_ASSERT instead; if you mean to do user error checking, use " \
                       "TORCH_CHECK.  See https://github.com/pytorch/pytorch/issues/20287 for more details.")
*/
inline void deprecated_AT_ASSERT() {}

/*
// Deprecation disabled until we fix sites in our codebase
C10_DEPRECATED_MESSAGE("AT_ASSERTM is deprecated, if you mean to indicate an internal invariant failure, use " \
                       "TORCH_INTERNAL_ASSERT instead; if you mean to do user error checking, use " \
                       "TORCH_CHECK.  See https://github.com/pytorch/pytorch/issues/20287 for more details.")
*/
inline void deprecated_AT_ASSERTM() {}

}} // namespace c10::detail

// Deprecated alias; this alias was deprecated because it wasn't clear to
// people that you should use a macro with AT_ prefix inside the torch/csrc
// directory.  Use TORCH_CHECK instead.
#define AT_CHECK(...)                                     \
  do {                                                    \
    ::c10::detail::deprecated_AT_CHECK();                 \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_CHECK(__VA_ARGS__)); \
  } while (false)

// Deprecated alias; this alias was deprecated because people kept mistakenly
// using it for user error checking.  Use TORCH_INTERNAL_ASSERT or TORCH_CHECK
// instead. See https://github.com/pytorch/pytorch/issues/20287 for more details.
#define AT_ASSERT(...)                                              \
  do {                                                              \
    ::c10::detail::deprecated_AT_ASSERT();                          \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(__VA_ARGS__)); \
  } while (false)

// Deprecated alias, like AT_ASSERT.  The new TORCH_INTERNAL_ASSERT macro supports
// both 0-ary and variadic calls, so having a separate message-accepting macro
// is not necessary.
//
// NB: we MUST include cond explicitly here, as MSVC will miscompile the macro
// expansion, shunting all of __VA_ARGS__ to cond.  An alternate workaround
// can be seen at
// https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
#define AT_ASSERTM(cond, ...)                                                 \
  do {                                                                        \
    ::c10::detail::deprecated_AT_ASSERTM();                                   \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(cond, __VA_ARGS__));     \
  } while (false)

// Deprecated alias; this alias was deprecated because it represents extra API
// surface that makes it hard for people to understand what macro to use.
// Use TORCH_CHECK(false, ...) or TORCH_INTERNAL_ASSERT(false, ...) to
// unconditionally fail at a line of code.
#define AT_ERROR(...)                                                         \
  do {                                                                        \
    ::c10::detail::deprecated_AT_ERROR();                                     \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_CHECK(false, ::c10::str(__VA_ARGS__)));  \
  } while (false)

// Deprecated alias; this alias was deprecated for consistency with TORCH_CHECK.
#define AT_INDEX_ERROR(...)                                                         \
  do {                                                                              \
    ::c10::detail::deprecated_AT_INDEX_ERROR();                                     \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_CHECK_INDEX(false, ::c10::str(__VA_ARGS__)));  \
  } while (false)

// Deprecated alias; this alias was deprecated because it wasn't clear to
// people that you should use a macro with AT_ prefix inside the torch/csrc
// directory.  Use TORCH_WARN instead.
#define AT_WARN(...)                                      \
  do {                                                    \
    ::c10::detail::deprecated_AT_WARN();                  \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_WARN(__VA_ARGS__));  \
  } while (false)


#endif // C10_UTIL_EXCEPTION_H_
