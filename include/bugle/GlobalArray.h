#ifndef BUGLE_GLOBALARRAY_H
#define BUGLE_GLOBALARRAY_H

#include <set>
#include <string>

namespace bugle {

class GlobalArray {
  std::string name;
  std::set<std::string> attributes;

public:
  GlobalArray(const std::string &name) : name(name) {}
  const std::string &getName() { return name; }
  void addAttribute(const std::string &attrib) {
    attributes.insert(attrib);
  }

  std::set<std::string>::const_iterator attrib_begin() const {
    return attributes.begin();
  }
  std::set<std::string>::const_iterator attrib_end() const {
    return attributes.end();
  }
};

}

#endif
