#include <algorithm>
#include <vector>

#ifndef BUGLE_OWNINGPTRVECTOR_H
#define BUGLE_OWNINGPTRVECTOR_H

namespace bugle {

template <typename T> class OwningPtrVector : public std::vector<T *> {
  OwningPtrVector(const OwningPtrVector &);            // DO NOT IMPLEMENT
  OwningPtrVector &operator=(const OwningPtrVector &); // DO NOT IMPLEMENT

public:
  OwningPtrVector() {}
  ~OwningPtrVector() {
    std::for_each(this->rbegin(), this->rend(), [](T *p) { delete p; });
  }
};
}

#endif
