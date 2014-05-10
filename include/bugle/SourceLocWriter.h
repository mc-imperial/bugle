#ifndef BUGLE_SOURCELOCWRITER_H
#define BUGLE_SOURCELOCWRITER_H

#include "bugle/SourceLoc.h"

namespace llvm {

class tool_output_file;
}

namespace bugle {

class SourceLocWriter {
  llvm::tool_output_file *L;
  unsigned SourceLocCounter;

public:
  SourceLocWriter(llvm::tool_output_file *L) : L(L), SourceLocCounter(0) {}
  unsigned writeSourceLocs(const SourceLocsRef &sourcelocs);
};
}

#endif
