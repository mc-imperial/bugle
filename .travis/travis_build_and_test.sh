#!/bin/bash
set -ev

export GPUVERIFY_DIR=${BUILD_ROOT}/gpuverify
export BUGLE_DIR=${BUILD_ROOT}
export DOWNLOADS_DIR=${BUILD_ROOT}/downloads

git clone --depth=${CLONE_DEPTH} \
  --branch=master https://github.com/mc-imperial/gpuverify.git

${GPUVERIFY_DIR}/.travis/build_and_test.sh
