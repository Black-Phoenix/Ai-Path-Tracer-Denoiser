#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/onnx/onnx.h>

#include <ostream>

namespace torch {
namespace jit {

// This map is used to keep track of parameters that should be exported
// externally. When `defer_weight_export` is true, the returned map contains
// kv pairs that map {external reference name} -> {at::Tensor to be exported}.
// It is the responsibility of the caller to export these appropriately.
//
// For example, when exporting to a zip archive, the caller may write out files
// for each entry in the export map, with the filename being the key and the
// file contents being the raw tensor data.
using RawDataExportMap = std::unordered_map<std::string, at::Tensor>;

constexpr size_t CURRENT_OP_VERSION_SET = 1;

TORCH_API std::tuple<std::string, RawDataExportMap> export_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    const std::unordered_map<std::string, std::unordered_map<int64_t, std::string>>& dynamic_axes,
    bool defer_weight_export = false,
    ::torch::onnx::OperatorExportTypes operator_export_type =
        ::torch::onnx::OperatorExportTypes::ONNX,
    bool strip_doc_string = true,
    bool keep_initializers_as_inputs = true);

// For testing purposes
TORCH_API std::string pretty_print_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    ::torch::onnx::OperatorExportTypes operator_export_type =
        ::torch::onnx::OperatorExportTypes::ONNX,
    bool google_printer = false,
    bool keep_initializers_as_inputs = true);

TORCH_API void ExportModule(
    const script::Module& module,
    std::ostream& out,
    const script::ExtraFilesMap& metadata = script::ExtraFilesMap(),
    bool bytecode_format = false);

TORCH_API void ExportModule(
    const script::Module& module,
    const std::string& filename,
    const script::ExtraFilesMap& metadata = script::ExtraFilesMap(),
    bool bytecode_format = false);

// Surrounding system can install an additional hook to produce extra files
// with metadata based on environment every time a module is serialized.
using ExportModuleExtraFilesHook =
    std::function<script::ExtraFilesMap(const script::Module&)>;
TORCH_API void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook);

} // namespace jit
} // namespace torch
