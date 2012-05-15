#include "bugle/util/UniqueNameSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace bugle;
using namespace llvm;

std::string UniqueNameSet::makeName(StringRef OrigName) {
  if (!OrigName.empty() && Names.insert(OrigName))
    return OrigName;

  unsigned i = 0;
  while (true) {
    std::string S = OrigName;
    llvm::raw_string_ostream SS(S);
    SS << i;
    if (Names.insert(SS.str()))
      return S;
    ++i;
  }
}
