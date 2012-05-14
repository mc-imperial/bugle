#ifndef BUGLE_BPLMODULEWRITER_H
#define BUGLE_BPLMODULEWRITER_H

#include <functional>
#include <set>
#include <string>

namespace llvm {

class raw_ostream;

}

namespace bugle {

class Module;
struct Type;

class BPLModuleWriter {
  llvm::raw_ostream &OS;
  bugle::Module *M;
  std::set<std::string> IntrinsicSet;

  void writeType(llvm::raw_ostream &OS, const bugle::Type &t);
  void writeIntrinsic(std::function<void(llvm::raw_ostream &)> F);

public:
  BPLModuleWriter(llvm::raw_ostream &OS, bugle::Module *M) : OS(OS), M(M) {}

  void write();

  friend class BPLFunctionWriter;
};

}

#endif
