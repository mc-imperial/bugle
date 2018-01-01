#include "bugle/SourceLocWriter.h"
#include "bugle/SourceLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace bugle;

unsigned SourceLocWriter::writeSourceLocs(const SourceLocsRef &sourcelocs) {
  ++SourceLocCounter;
  if (L == nullptr)
    return SourceLocCounter - 1;

  llvm::raw_ostream &OS = L->os();

  for (auto i = sourcelocs->begin(), e = sourcelocs->end(); i != e; ++i) {
    OS << i->getLineNo() << "\x1F";   // unit separator
    OS << i->getColNo() << "\x1F";    // unit separator
    OS << i->getFileName() << "\x1F"; // unit separator
    OS << i->getPath() << "\x1F";     // unit separator
    OS << "\x1E";                     // record separator
  }
  OS << "\x1D"; // group separator

  return SourceLocCounter - 1;
}
