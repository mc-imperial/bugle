#include <string>

#ifndef BUGLE_GLOBALARRAY_H
#define BUGLE_GLOBALARRAY_H

namespace bugle {

class GlobalArray {
  std::string name;

public:
  GlobalArray(const std::string &name) : name(name) {}
  const std::string &getName() { return name; }
};

}

#endif
