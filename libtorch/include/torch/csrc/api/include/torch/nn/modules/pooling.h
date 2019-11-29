#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/options/pooling.h>
#include <torch/nn/functional/pooling.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) avgpool modules.
template <size_t D, typename Derived>
class TORCH_API AvgPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  AvgPoolImpl(ExpandingArray<D> kernel_size)
      : AvgPoolImpl(AvgPoolOptions<D>(kernel_size)) {}
  explicit AvgPoolImpl(const AvgPoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `AvgPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  AvgPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AvgPool1d to learn
/// about the exact behavior of this module.
class TORCH_API AvgPool1dImpl : public AvgPoolImpl<1, AvgPool1dImpl> {
 public:
  using AvgPoolImpl<1, AvgPool1dImpl>::AvgPoolImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AvgPool1dImpl`.
/// See the documentation for `AvgPool1dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AvgPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AvgPool2d to learn
/// about the exact behavior of this module.
class TORCH_API AvgPool2dImpl : public AvgPoolImpl<2, AvgPool2dImpl> {
 public:
  using AvgPoolImpl<2, AvgPool2dImpl>::AvgPoolImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AvgPool2dImpl`.
/// See the documentation for `AvgPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AvgPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AvgPool3d to learn
/// about the exact behavior of this module.
class TORCH_API AvgPool3dImpl : public AvgPoolImpl<3, AvgPool3dImpl> {
 public:
  using AvgPoolImpl<3, AvgPool3dImpl>::AvgPoolImpl;
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AvgPool2dImpl`.
/// See the documentation for `AvgPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AvgPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) maxpool modules.
template <size_t D, typename Derived>
class TORCH_API MaxPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  MaxPoolImpl(ExpandingArray<D> kernel_size)
      : MaxPoolImpl(MaxPoolOptions<D>(kernel_size)) {}
  explicit MaxPoolImpl(const MaxPoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `MaxPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  MaxPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MaxPool1d to learn
/// about the exact behavior of this module.
class TORCH_API MaxPool1dImpl : public MaxPoolImpl<1, MaxPool1dImpl> {
 public:
  using MaxPoolImpl<1, MaxPool1dImpl>::MaxPoolImpl;
  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `torch::nn::MaxUnpool1d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `MaxPool1dImpl`.
/// See the documentation for `MaxPool1dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MaxPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MaxPool2d to learn
/// about the exact behavior of this module.
class TORCH_API MaxPool2dImpl : public MaxPoolImpl<2, MaxPool2dImpl> {
 public:
  using MaxPoolImpl<2, MaxPool2dImpl>::MaxPoolImpl;
  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `torch::nn::MaxUnpool2d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `MaxPool2dImpl`.
/// See the documentation for `MaxPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MaxPool3d to learn
/// about the exact behavior of this module.
class TORCH_API MaxPool3dImpl : public MaxPoolImpl<3, MaxPool3dImpl> {
 public:
  using MaxPoolImpl<3, MaxPool3dImpl>::MaxPoolImpl;
  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `torch::nn::MaxUnpool3d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `MaxPool3dImpl`.
/// See the documentation for `MaxPool3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MaxPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) adaptive maxpool modules.
template <size_t D, typename Derived>
class TORCH_API AdaptiveMaxPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  AdaptiveMaxPoolImpl(ExpandingArray<D> output_size)
      : AdaptiveMaxPoolImpl(AdaptiveMaxPoolOptions<D>(output_size)) {}
  explicit AdaptiveMaxPoolImpl(const AdaptiveMaxPoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `AdaptiveMaxPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  AdaptiveMaxPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive maxpool over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveMaxPool1d
/// to learn about the exact behavior of this module.
class TORCH_API AdaptiveMaxPool1dImpl :
  public AdaptiveMaxPoolImpl<1, AdaptiveMaxPool1dImpl> {
 public:
  using AdaptiveMaxPoolImpl<1, AdaptiveMaxPool1dImpl>::AdaptiveMaxPoolImpl;

  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool1d.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveMaxPool1dImpl`.
/// See the documentation for `AdaptiveMaxPool1dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AdaptiveMaxPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive maxpool over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveMaxPool2d
/// to learn about the exact behavior of this module.
class TORCH_API AdaptiveMaxPool2dImpl :
  public AdaptiveMaxPoolImpl<2, AdaptiveMaxPool2dImpl> {
 public:
  using AdaptiveMaxPoolImpl<2, AdaptiveMaxPool2dImpl>::AdaptiveMaxPoolImpl;

  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool2d.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveMaxPool2dImpl`.
/// See the documentation for `AdaptiveMaxPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AdaptiveMaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive maxpool over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveMaxPool3d
/// to learn about the exact behavior of this module.
class TORCH_API AdaptiveMaxPool3dImpl :
  public AdaptiveMaxPoolImpl<3, AdaptiveMaxPool3dImpl> {
 public:
  using AdaptiveMaxPoolImpl<3, AdaptiveMaxPool3dImpl>::AdaptiveMaxPoolImpl;

  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool3d.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveMaxPool3dImpl`.
/// See the documentation for `AdaptiveMaxPool3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AdaptiveMaxPool3d);

} // namespace nn
} // namespace torch
