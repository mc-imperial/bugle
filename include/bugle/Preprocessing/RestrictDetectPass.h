#ifndef BUGLE_PREPROCESSING_RESTRICTDETECTPASS_H
#define BUGLE_PREPROCESSING_RESTRICTDETECTPASS_H

#include "bugle/Translator/TranslateModule.h"
#include "llvm/Pass.h"
#include "llvm/IR/DebugInfo.h"

namespace bugle {

class RestrictDetectPass : public llvm::FunctionPass {
private:
  llvm::Module *M;
  llvm::DebugInfoFinder DIF;
  TranslateModule::SourceLanguage SL;
  std::set<std::string> GPUEntryPoints;

  std::string getFunctionLocation(llvm::Function *F);
  void doRestrictCheck(llvm::Function &F);
public:
  static char ID;

  RestrictDetectPass(llvm::Module *M, TranslateModule::SourceLanguage SL,
                     std::set<std::string> &EP) :
    FunctionPass(ID), M(M), SL(SL), GPUEntryPoints(EP) {
      DIF.processModule(*M);
    }

  virtual const char *getPassName() const {
    return "Detect restrict usage on global pointers";
  }

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  virtual bool runOnFunction(llvm::Function &M);
};

}

#endif
