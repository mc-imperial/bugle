#!/bin/sh

BASE=`basename $1 .cu`

@LLVM_BINDIR@/clang -I@CMAKE_SOURCE_DIR@/include -I@GPUVERIFY_BENCHMARKING_DIR@/include_cl -emit-llvm -Xclang -fcuda-is-device -c -o $BASE.bc "$@"
@LLVM_BINDIR@/opt -mem2reg -o $BASE.opt.bc $BASE.bc
@CMAKE_BINARY_DIR@/bugle -l cu -o $BASE.bpl $BASE.opt.bc
