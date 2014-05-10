#include "bugle/Ident.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <iterator>

using namespace bugle;

namespace {

bool isBoogieIdentCharOrAt(char c) {
  static char IdentChars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                             "abcdefghijklmnopqrstuvwxyz"
                             "'~#$^_.?`0123456789@";
  return strchr(IdentChars, c);
}
}

std::string bugle::makeBoogieIdent(llvm::StringRef S) {
  std::string Ident;
  std::copy_if(S.begin(), S.end(), std::back_inserter(Ident),
               isBoogieIdentCharOrAt);
  std::replace(Ident.begin(), Ident.end(), '@', '~');
  return Ident;
}
