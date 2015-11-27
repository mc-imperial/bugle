#include "bugle/Preprocessing/SimpleInternalizePass.h"
#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Translator/TranslateModule.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace bugle;

bool SimpleInternalizePass::isEntryPoint(llvm::Function *F) {
  if (OnlyExplicitEntryPoints)
    return GPUEntryPoints.find(F->getName()) != GPUEntryPoints.end();
  else
    return TranslateModule::isGPUEntryPoint(F, M, SL, GPUEntryPoints) ||
           TranslateFunction::isStandardEntryPoint(SL, F->getName());
}

bool SimpleInternalizePass::doInternalize(llvm::Function *F) {
  if (!TranslateFunction::isNormalFunction(SL, F) || isEntryPoint(F))
    return false;

  F->setVisibility(GlobalValue::DefaultVisibility);
  F->setLinkage(GlobalValue::InternalLinkage);
  return true;
}

bool SimpleInternalizePass::runOnModule(llvm::Module &M) {
  bool changed = false;
  this->M = &M;

  for (auto i = M.begin(), e = M.end(); i != e; ++i)
    changed &= doInternalize(&*i);

  return changed;
}

char SimpleInternalizePass::ID = 0;
