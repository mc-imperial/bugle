#ifndef BUGLE_TRANSLATOR_TRANSLATEFUNCTION_H
#define BUGLE_TRANSLATOR_TRANSLATEFUNCTION_H

namespace llvm {

class BasicBlock;
class Function;

}

namespace bugle {

class BasicBlock;
class TranslateModule;

class TranslateFunction {
  TranslateModule *TM;
  llvm::Function *F;

  void translateBasicBlock(BasicBlock *BBB, llvm::BasicBlock *BB);

public:
  TranslateFunction(TranslateModule *TM, llvm::Function *F) : TM(TM), F(F) {}
  void translate();
};

}

#endif
