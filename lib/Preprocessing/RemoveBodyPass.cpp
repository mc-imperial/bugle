#include "bugle/Module.h"
#include "bugle/Preprocessing/RemoveBodyPass.h"
#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Translator/TranslateModule.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace bugle;

bool RemoveBodyPass::runOnFunction(llvm::Function &F) {
  if (!TranslateFunction::isNormalFunction(SL, &F))
    return false;

  if (TranslateModule::isGPUEntryPoint(&F, M, GPUEntryPoints) ||
      TranslateFunction::isStandardEntryPoint(SL, F.getName()))
    return false;

  F.deleteBody();
  return true;
}

char RemoveBodyPass::ID = 0;
