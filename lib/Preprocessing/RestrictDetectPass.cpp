#include "bugle/Preprocessing/RestrictDetectPass.h"
#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Translator/TranslateModule.h"
#include "bugle/util/ErrorReporter.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"

using namespace llvm;
using namespace bugle;

void RestrictDetectPass::doRestrictCheck(llvm::Function &F) {
  std::vector<Argument *> AL;
  for (auto i = F.arg_begin(), e = F.arg_end(); i != e; ++i) {
    if (i->hasNoAliasAttr() || !i->getType()->isPointerTy())
      continue;

    switch (i->getType()->getPointerAddressSpace()) {
    case TranslateModule::AddressSpaces::standard:
      if (SL == TranslateModule::SL_CUDA)
        AL.push_back(i);
      break;
    case TranslateModule::AddressSpaces::global:
      AL.push_back(i);
      break;
    default:
      break;
    }
  }

  if (AL.size() <= 1)
    return;

  std::string msg = "Assuming the arguments ";

  auto i = AL.begin(), e = AL.end();
  do {
    msg += "'" + (*i)->getName().str() + "'";
    ++i;
    if (i != e)
      msg += ", ";
  } while (i != e);

  msg += " of '" + F.getName().str() + "' to be non-aliased; ";
  msg += "please consider adding a restrict qualifier to these arguments";
  ErrorReporter::emitWarning(msg);
}

bool RestrictDetectPass::runOnFunction(llvm::Function &F) {
  if (!(SL == TranslateModule::SL_OpenCL || SL == TranslateModule::SL_CUDA))
    return false;
  if (!TranslateFunction::isNormalFunction(SL, &F))
    return false;
  if (!TranslateModule::isGPUEntryPoint(&F, M, GPUEntryPoints))
    return false;

  doRestrictCheck(F);

  return false;
}

char RestrictDetectPass::ID = 0;
