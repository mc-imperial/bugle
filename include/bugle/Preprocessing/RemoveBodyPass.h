#ifndef BUGLE_PREPROCESSING_REMOVEBODYPASS_H
#define BUGLE_PREPROCESSING_REMOVEBODYPASS_H

#include "bugle/Preprocessing/InlinePass.h"
#include "bugle/Translator/TranslateModule.h"
#include "llvm/Pass.h"

namespace bugle {

class RemoveBodyPass : public llvm::FunctionPass {
private:
  llvm::Module *M;
  TranslateModule::SourceLanguage SL;
  std::set<std::string> GPUEntryPoints;

public:
  static char ID;

 RemoveBodyPass(llvm::Module *M, TranslateModule::SourceLanguage SL,
                std::set<std::string> &EP) :
  FunctionPass(ID), M(M), SL(SL), GPUEntryPoints(EP) {}

  virtual const char *getPassName() const {
    return "Remove Function Bodies After Inlining";
  }

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const {
    AU.addRequired<InlinePass>();
  }

  virtual bool runOnFunction(llvm::Function &M);
};

}

#endif
