#ifndef BUGLE_BPLMODULEWRITER_H
#define BUGLE_BPLMODULEWRITER_H

namespace llvm {

class raw_ostream;

}

namespace bugle {

class BPLModuleWriter {
  llvm::raw_ostream &OS;

public:
  BPLModuleWriter(llvm::raw_ostream &OS) : OS(OS) {}

  friend class BPLFunctionWriter;
};

}

#endif
