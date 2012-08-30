#ifndef BUGLE_H
#define BUGLE_H

#include <device_qualifier.h>

void _DEVICE_QUALIFIER bugle_assert(int);
void _DEVICE_QUALIFIER bugle_assume(int);
void _DEVICE_QUALIFIER bugle_requires(int);
void _DEVICE_QUALIFIER bugle_ensures(int);
void _DEVICE_QUALIFIER bugle_barrier(bool local_flags, bool global_flags);

#endif
