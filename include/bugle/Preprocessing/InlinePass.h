#ifndef BUGLE_PREPROCESSING_INLINEPASS_H
#define BUGLE_PREPROCESSING_INLINEPASS_H

#include "bugle/Translator/TranslateModule.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/CallGraph.h"
#include <set>

namespace bugle {

// This should be a ModulePass as inlining affects multiple functions.

class InlinePass : public llvm::ModulePass {
private:
  llvm::Module *M;
  TranslateModule::SourceLanguage SL;
  std::set<std::string> GPUEntryPoints;

  bool doInline(llvm::Instruction *I, llvm::Function *OF);
  void doInline(llvm::BasicBlock *B, llvm::Function *OF);
  void doInline(llvm::Function *F);

public:
  static char ID;

  InlinePass(TranslateModule::SourceLanguage SL, std::set<std::string> &EP)
      : ModulePass(ID), M(0), SL(SL), GPUEntryPoints(EP) {}

  llvm::StringRef getPassName() const override { return "Function inlining"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<llvm::CallGraphWrapperPass>();
  }

  bool runOnModule(llvm::Module &M) override;
};
}

#endif
