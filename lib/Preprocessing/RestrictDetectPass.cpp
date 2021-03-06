#include "bugle/Preprocessing/RestrictDetectPass.h"
#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Translator/TranslateModule.h"
#include "bugle/util/ErrorReporter.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace bugle;

bool RestrictDetectPass::doInitialization(llvm::Module &M) {
  this->M = &M;
  DIF.processModule(M);
  return false;
}

const DISubprogram *RestrictDetectPass::getDebugInfo(llvm::Function *F) {
  for (auto *S : DIF.subprograms()) {
    if (S->describes(F))
      return S;
  }

  return nullptr;
}

std::string RestrictDetectPass::getFunctionLocation(llvm::Function *F) {
  auto *MDS = getDebugInfo(F);
  if (MDS) {
    std::string S;
    llvm::raw_string_ostream SS(S);
    SS << "'" << MDS->getName() << "' on line " << MDS->getLine()
       << " of " << MDS->getFilename();
    return SS.str();
  } else
    return "'" + F->getName().str() + "'";
}

bool RestrictDetectPass::ignoreArgument(unsigned i, const DISubprogram *DIS) {
  if (!DIS || SL != TranslateModule::SL_OpenCL)
    return false;

  auto Ty = DIS->getType()->getTypeArray()[i + 1].resolve()->getName();
  return (Ty == "__bugle_image2d_t" || Ty == "__bugle_image3d_t");
}

void RestrictDetectPass::doRestrictCheck(llvm::Function &F) {
  auto *DIS = getDebugInfo(&F);
  std::vector<Argument *> AL;
  for (auto &Arg : F.args()) {
    if (!Arg.getType()->isPointerTy())
      continue;
    if (Arg.hasNoAliasAttr())
      continue;
    if (Arg.getType()->getPointerElementType()->isFunctionTy())
      continue;
    if (ignoreArgument(Arg.getArgNo(), DIS))
      continue;

    unsigned addressSpace = Arg.getType()->getPointerAddressSpace();
    if (addressSpace == AddressSpaces.generic &&
        SL == TranslateModule::SL_CUDA)
      AL.push_back(&Arg);

    if (addressSpace == AddressSpaces.global)
      AL.push_back(&Arg);
  }

  if (AL.size() <= 1)
    return;

  std::string msg = "Assuming the arguments ";

  auto i = AL.begin(), e = AL.end();
  do {
    msg += "'" + TranslateModule::getSourceName(*i, &F) + "'";
    ++i;
    if (i != e)
      msg += ", ";
  } while (i != e);

  std::string name = getFunctionLocation(&F);
  msg += " of " + name + " to be non-aliased; ";
  msg += "please consider adding a restrict qualifier to these arguments";
  ErrorReporter::emitWarning(msg);
}

bool RestrictDetectPass::runOnFunction(llvm::Function &F) {
  if (SL != TranslateModule::SL_OpenCL && SL != TranslateModule::SL_CUDA)
    return false;
  if (!TranslateFunction::isNormalFunction(SL, &F))
    return false;
  if (!TranslateModule::isGPUEntryPoint(&F, M, SL, GPUEntryPoints))
    return false;

  doRestrictCheck(F);

  return false;
}

char RestrictDetectPass::ID = 0;
