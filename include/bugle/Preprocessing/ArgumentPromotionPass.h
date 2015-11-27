#ifndef BUGLE_PREPROCESSING_ARGUMENTPROMOTIONPASS_H
#define BUGLE_PREPROCESSING_ARGUMENTPROMOTIONPASS_H

#include "bugle/Translator/TranslateModule.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include <set>

namespace bugle {

// Compiler pass to promote call-by-value function arguments to registers.
// The implementation of this pass is based on LLVM's ArgumentPromotion.cpp.
// We do not update call graph or alias information, but we invalidate this
// information instead.

class ArgumentPromotionPass : public llvm::ModulePass {
private:
  llvm::Module *M;
  TranslateModule::SourceLanguage SL;
  std::set<std::string> GPUEntryPoints;

  bool needsPromotion(llvm::Function *F);
  bool canPromote(llvm::Function *F);
  bool usesFunctionPointers(llvm::Function *F);
  llvm::Function *createNewFunction(llvm::Function *F);
  void updateCallSite(llvm::CallSite *CS, llvm::Function *F,
                      llvm::Function *NF);
  void replaceMetaData(llvm::Function *F, llvm::Function *NF, llvm::MDNode *MD,
                       std::set<llvm::MDNode *> &doneMD);
  void spliceBody(llvm::Function *F, llvm::Function *NF);
  void promote(llvm::Function *F);

public:
  static char ID;

  ArgumentPromotionPass(TranslateModule::SourceLanguage SL,
                        std::set<std::string> &EP)
      : ModulePass(ID), M(nullptr), SL(SL), GPUEntryPoints(EP) {}

  virtual const char *getPassName() const { return "Argument promotion"; }

  virtual bool runOnModule(llvm::Module &M);
};
}

#endif
