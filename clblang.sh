#!/bin/sh

BASE=`basename $1 .cl`

@LLVM_BINDIR@/clang -I@CMAKE_SOURCE_DIR@/include -I@GPUVERIFY_BENCHMARKING_DIR@/include_cl -emit-llvm -c -o $BASE.bc "$@"
@LLVM_BINDIR@/opt -mem2reg -o $BASE.opt.bc $BASE.bc
@CMAKE_BINARY_DIR@/bugle -l cl -o $BASE.bpl $BASE.opt.bc
