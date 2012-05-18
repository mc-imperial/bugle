#include "bugle/Ident.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>

using namespace bugle;

bool bugle::isBoogieIdentChar(char c) {
  static char IdentChars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                             "abcdefghijklmnopqrstuvwxyz"
                             "'~#$^_.?`0123456789";
  return strchr(IdentChars, c);
}

std::string bugle::makeBoogieIdent(llvm::StringRef S) {
  std::string Ident;
  std::copy_if(S.begin(), S.end(), std::back_inserter(Ident),
               isBoogieIdentChar);
  return Ident;
}
