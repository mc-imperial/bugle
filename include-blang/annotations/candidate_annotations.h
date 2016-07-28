#ifndef CANDIDATE_ANNOTATIONS_H
#define CANDIDATE_ANNOTATIONS_H

#ifndef ANNOTATIONS_H
#error candidate_annotations.h must be included after annotations.h
#endif

#undef    __invariant
#undef    __global_invariant
#define   __invariant(X)        __candidate_invariant(X)
#define   __global_invariant(X) __candidate_global_invariant(X)
#define   __function_wide_invariant(X) __function_wide_candidate_invariant(X)

#endif
