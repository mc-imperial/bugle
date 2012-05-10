#include <algorithm>
#include <vector>

#ifndef BUGLE_OWNINGPTRVECTOR_H
#define BUGLE_OWNINGPTRVECTOR_H

namespace bugle {

template <typename T>
class OwningPtrVector : public std::vector<T *> {
  ~OwningPtrVector() {
    std::for_each(this->begin(), this->end(), [](T *p){ delete p; });
  }
};

}

#endif
