// RUN: stablehlo-ext-opt --remove-unused-broadcast %s | FileCheck %s
// 该命令会使用 stablehlo-ext-opt 工具运行 --remove-unused-broadcast pass，
// 并使用 FileCheck 检查输出是否符合预期。

// 测试用例 1: 冗余的 broadcast，应该被移除
func.func @test_remove_redundant_broadcast(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK-LABEL: @test_remove_redundant_broadcast
  // CHECK-SAME:    %[[ARG0:.+]]: tensor<2x3xf32>
  // CHECK-NEXT:  return %[[ARG0]] : tensor<2x3xf32>
  // 预期结果：broadcast_in_dim 操作被移除，直接返回输入 %arg0
  // Note: Changed broadcast_dimensions to dims based on error message
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// 测试用例 2: 输入输出形状不同，不应该被移除
func.func @test_keep_broadcast_different_shape(%arg0: tensor<2x3xf32>) -> tensor<2x3x1xf32> {
  // CHECK-LABEL: @test_keep_broadcast_different_shape
  // CHECK:       %[[RES:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x1xf32> // CHECK attribute might need update if pass runs
  // CHECK-NEXT:  return %[[RES]] : tensor<2x3x1xf32>
  // 预期结果：broadcast_in_dim 操作保留
  // Note: Changed broadcast_dimensions to dims based on error message
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
  return %0 : tensor<2x3x1xf32>
}
