#include "bugle/Translator/TranslateModule.h"
#include "bugle/Expr.h"
#include "llvm/Constant.h"

using namespace bugle;

ref<Expr> TranslateModule::translateConstant(llvm::Constant *C) {
  return ref<Expr>();
}

void TranslateModule::translate() {
}
