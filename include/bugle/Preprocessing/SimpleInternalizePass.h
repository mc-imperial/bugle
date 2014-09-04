#ifndef BUGLE_PREPROCESSING_SIMPLEINTERNALIZEPASS_H
#define BUGLE_PREPROCESSING_SIMPLEINTERNALIZEPASS_H

#include "bugle/Translator/TranslateModule.h"
#include "llvm/Pass.h"

namespace bugle {

class SimpleInternalizePass : public llvm::ModulePass {
private:
  llvm::Module *M;
  TranslateModule::SourceLanguage SL;
  std::set<std::string> GPUEntryPoints;
  bool OnlyExplicitEntryPoints;

  bool isEntryPoint(llvm::Function *F);
  bool doInternalize(llvm::Function *F);

public:
  static char ID;

  SimpleInternalizePass(TranslateModule::SourceLanguage SL,
                        std::set<std::string> &EP, bool EEP)
      : ModulePass(ID), SL(SL), GPUEntryPoints(EP),
        OnlyExplicitEntryPoints(EEP) {}

  virtual const char *getPassName() const {
    return "Internalize all normal functions that are not entry points";
  }

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const {
    AU.setPreservesCFG();
  }

  virtual bool runOnModule(llvm::Module &M);
};
}

#endif
