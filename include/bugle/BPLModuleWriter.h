#ifndef BUGLE_BPLMODULEWRITER_H
#define BUGLE_BPLMODULEWRITER_H

namespace llvm {

class raw_ostream;

}

namespace bugle {

class Module;
class Type;

class BPLModuleWriter {
  llvm::raw_ostream &OS;
  bugle::Module *M;

  void writeType(llvm::raw_ostream &OS, const bugle::Type &t);

public:
  BPLModuleWriter(llvm::raw_ostream &OS, bugle::Module *M) : OS(OS), M(M) {}

  void write();

  friend class BPLFunctionWriter;
};

}

#endif
