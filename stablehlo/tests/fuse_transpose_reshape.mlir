// RUN: stablehlo-ext-opt --fuse-transpose-reshape-transpose %s | FileCheck %s
// 该命令使用 stablehlo-ext-opt 工具运行 --fuse-transpose-reshape-transpose pass，
// 并使用 FileCheck 检查输出是否符合预期。

// 测试用例 1: 可融合的 transpose -> reshape -> transpose 序列
// 这个例子基于 FuseTransposeReshapeTranspose.cpp 中的注释
func.func @test_fuse_transpose_reshape_transpose(%arg0: tensor<128x30x16x32xf16>) -> tensor<16x128x960xf16> {
  // CHECK-LABEL: @test_fuse_transpose_reshape_transpose
  // CHECK-SAME:    %[[ARG0:.+]]: tensor<128x30x16x32xf16>

  // CHECK: %[[T_NEW:.+]] = stablehlo.transpose %[[ARG0]], dims = [2, 0, 1, 3] : (tensor<128x30x16x32xf16>) -> tensor<16x128x30x32xf16> // Updated attribute name
  // CHECK: %[[R_NEW:.+]] = stablehlo.reshape %[[T_NEW]] : (tensor<16x128x30x32xf16>) -> tensor<16x128x960xf16>
  // CHECK: return %[[R_NEW]] : tensor<16x128x960xf16>

  // 原始序列
  // Note: Changed permutation to dims based on error message
  %t1 = stablehlo.transpose %arg0, dims = [0, 2, 1, 3] : (tensor<128x30x16x32xf16>) -> tensor<128x16x30x32xf16>
  %r1 = stablehlo.reshape %t1 : (tensor<128x16x30x32xf16>) -> tensor<128x16x960xf16>
  // Note: Changed permutation to dims based on error message
  %t2 = stablehlo.transpose %r1, dims = [1, 0, 2] : (tensor<128x16x960xf16>) -> tensor<16x128x960xf16>
  return %t2 : tensor<16x128x960xf16>
}

// 测试用例 2: Reshape 操作执行维度拆分，不应融合
// transpose (128x960x16 -> 128x16x960) -> reshape (128x16x960 -> 128x16x30x32) -> transpose (128x16x30x32 -> 128x30x16x32)
func.func @test_no_fuse_dimension_split(%arg0: tensor<128x960x16xf16>) -> tensor<128x30x16x32xf16> {
  // CHECK-LABEL: @test_no_fuse_dimension_split
  // CHECK: stablehlo.transpose %arg0, dims = [0, 2, 1]
  // CHECK: stablehlo.reshape
  // CHECK: stablehlo.transpose {{.*}}, dims = [0, 2, 1, 3]
  // CHECK: return

  %t1 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<128x960x16xf16>) -> tensor<128x16x960xf16>
  // Reshape 拆分了最后一个维度 (960 -> 30x32)
  %r1 = stablehlo.reshape %t1 : (tensor<128x16x960xf16>) -> tensor<128x16x30x32xf16>
  %t2 = stablehlo.transpose %r1, dims = [0, 2, 1, 3] : (tensor<128x16x30x32xf16>) -> tensor<128x30x16x32xf16>
  return %t2 : tensor<128x30x16x32xf16>
}

// 测试用例 3: Reshape 操作合并非尾部维度，可以融合
// transpose (128x30x16x32 -> 128x30x32x16) -> reshape (128x30x32x16 -> 128x960x16) -> transpose (128x960x16 -> 16x128x960)
func.func @test_fuse_non_tail_merge_v2(%arg0: tensor<128x30x16x32xf16>) -> tensor<16x128x960xf16> {
  // CHECK-LABEL: @test_no_fuse_non_tail_merge_v2
  // CHECK-SAME:    %[[ARG0:.+]]: tensor<128x30x16x32xf16>
  // CHECK:         %[[T_NEW:.+]] = stablehlo.transpose %[[ARG0]], dims = [2, 0, 1, 3] : (tensor<128x30x16x32xf16>) -> tensor<16x128x30x32xf16>
  // CHECK:         %[[R_NEW:.+]] = stablehlo.reshape %[[T_NEW]] : (tensor<16x128x30x32xf16>) -> tensor<16x128x960xf16>
  // CHECK:         return %[[R_NEW]] : tensor<16x128x960xf16>

  %t1 = stablehlo.transpose %arg0, dims = [0, 1, 3, 2] : (tensor<128x30x16x32xf16>) -> tensor<128x30x32x16xf16>
  // Reshape 合并了中间维度 (30x32 -> 960)，不是尾部维度
  %r1 = stablehlo.reshape %t1 : (tensor<128x30x32x16xf16>) -> tensor<128x960x16xf16>
  %t2 = stablehlo.transpose %r1, dims = [2, 0, 1] : (tensor<128x960x16xf16>) -> tensor<16x128x960xf16>
  return %t2 : tensor<16x128x960xf16>
}
