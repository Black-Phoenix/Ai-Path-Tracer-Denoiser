#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace script {

struct SourceImporterImpl;

// Given a directory of serialized TorchScript sources,
// This class allows the loading of individual named types in source.
// Resolves the dependencies between source files and parses
// the source files as necessary.
using SourceLoader = std::function<std::shared_ptr<Source>(const std::string&)>;

struct TORCH_API SourceImporter {
  SourceImporter(
      // The compilation unit that will own the imported source
      std::shared_ptr<CompilationUnit> cu,
      const std::vector<at::Tensor>* tensor_table,
      SourceLoader loader);

  TypePtr loadNamedType(const QualifiedName& name) const;

  // Add the methods defined in `src` to the module `mod`, using SourceImporter
  // to resolve any classes via loadNamedType
  void LEGACY_import_methods(
      const script::Module& mod,
      const std::shared_ptr<Source>& src);
  ~SourceImporter();

 private:
  std::shared_ptr<SourceImporterImpl> pImpl;
};

} // namespace script
} // namespace jit
} // namespace torch
