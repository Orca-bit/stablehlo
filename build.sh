#!/bin/bash

set -e

cur_dir=$(readlink -f $(dirname $0))

[[ "$(uname)" != "Darwin" ]] && LLVM_ENABLE_LLD="ON" || LLVM_ENABLE_LLD="OFF"

# 解析命令行参数
build_llvm=true
if [ "$1" == "--skip-llvm" ]; then
  build_llvm=false
fi

cd $cur_dir

# 只有在需要时才构建 LLVM
if [ "$build_llvm" = true ]; then
  if [ ! -d "llvm-project" ]; then
    git clone git@github.com:llvm/llvm-project.git
  fi
  (cd llvm-project && git fetch && git checkout $(cat ../build_tools/llvm_version.txt))

  cd $cur_dir
  MLIR_ENABLE_BINDINGS_PYTHON=OFF
  bash build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build
fi

# 构建 StableHLO
mkdir -p build && cd build

cmake .. -GNinja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_ENABLE_LLD="$LLVM_ENABLE_LLD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
  -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir

cmake --build .
