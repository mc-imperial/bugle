#include "bugle/Preprocessing/InlinePass.h"
#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Translator/TranslateModule.h"
#include "bugle/util/ErrorReporter.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace bugle;

bool InlinePass::doInline(llvm::Instruction *I, llvm::Function *OF) {
  auto CI = dyn_cast<CallInst>(I);

  if (!CI)
    return false;

  auto F = CI->getCalledFunction();

  if (!F)
    ErrorReporter::reportImplementationLimitation(
        "Function pointers not compatible with inlining");

  if (!(TranslateModule::isGPUEntryPoint(OF, M, SL, GPUEntryPoints) ||
        TranslateFunction::isStandardEntryPoint(SL, OF->getName()))) {
    if (TranslateFunction::isPreOrPostCondition(F->getName())) {
      ErrorReporter::reportFatalError(
          "Cannot inline, detected function with pre- or post-condition");
    } else { // Do not perform inlining on non-entry point functions.
      return false;
    }
  }

  // Do not inline functions that are special.
  if (!TranslateFunction::isNormalFunction(SL, F))
    return false;

  // Do not inline entry points in entry points, they may have pre- and
  // post-conditions.
  if (TranslateModule::isGPUEntryPoint(F, M, SL, GPUEntryPoints) ||
      TranslateFunction::isStandardEntryPoint(SL, F->getName()))
    return false;

  CallSite CS(CI);
  CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  InlineFunctionInfo IFI(&CG);
  if (InlineFunction(CI, IFI))
    return true;
  else
    return false;
}

void InlinePass::doInline(llvm::BasicBlock *BB, llvm::Function *OF) {
  // Re-process block as long as we did some inlining.
  bool AppliedInlining = true;
  while (AppliedInlining) {
    for (auto &I : *BB) {
      AppliedInlining = doInline(&I, OF);
      if (AppliedInlining)
        break;
    }
  }
}

void InlinePass::doInline(llvm::Function *F) {
  // Only apply inlining to normal functions.
  if (!TranslateFunction::isNormalFunction(SL, F))
    return;

  for (auto &BB : *F)
    doInline(&BB, F);
}

bool InlinePass::runOnModule(llvm::Module &M) {
  this->M = &M;

  for (auto &F : M)
    doInline(&F);

  return true;
}

char InlinePass::ID = 0;
