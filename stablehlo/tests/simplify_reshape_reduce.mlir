// RUN: stablehlo-ext-opt %s --simplify-reshape-reduce | FileCheck %s

func.func @test_reshape_reduce_add_valid(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // 测试有效的 reshape + reduce (加法，初始值为0) 简化
  // CHECK-LABEL: @test_reshape_reduce_add_valid
  // CHECK: return %arg0 : tensor<2x3xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<2x1x3xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<f32>
  %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.add across dimensions = [1] : (tensor<2x1x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}

func.func @test_reshape_reduce_mul_valid(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  // 测试有效的 reshape + reduce (乘法，初始值为1) 简化
  // CHECK-LABEL: @test_reshape_reduce_mul_valid
  // CHECK: return %arg0 : tensor<4x5xi32>
  %0 = stablehlo.reshape %arg0 : (tensor<4x5xi32>) -> tensor<4x1x5xi32>
  %1 = stablehlo.constant dense<1> : tensor<i32>
  %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.multiply across dimensions = [1] : (tensor<4x1x5xi32>, tensor<i32>) -> tensor<4x5xi32>
  return %2 : tensor<4x5xi32>
}

func.func @test_reshape_reduce_add_invalid_init(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // 测试无效的初始值（加法操作但初始值不为0）
  // CHECK-LABEL: @test_reshape_reduce_add_invalid_init
  // CHECK: stablehlo.reshape
  // CHECK: stablehlo.reduce
  %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<2x1x3xf32>
  %1 = stablehlo.constant dense<1.0> : tensor<f32>  // 错误的初始值
  %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.add across dimensions = [1] : (tensor<2x1x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}

func.func @test_reshape_reduce_mul_invalid_init(%arg0: tensor<4x5xi32>) -> tensor<4x5xi32> {
  // 测试无效的初始值（乘法操作但初始值不为1）
  // CHECK-LABEL: @test_reshape_reduce_mul_invalid_init
  // CHECK: stablehlo.reshape
  // CHECK: stablehlo.reduce
  %0 = stablehlo.reshape %arg0 : (tensor<4x5xi32>) -> tensor<4x1x5xi32>
  %1 = stablehlo.constant dense<2> : tensor<i32>  // 错误的初始值
  %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.multiply across dimensions = [1] : (tensor<4x1x5xi32>, tensor<i32>) -> tensor<4x5xi32>
  return %2 : tensor<4x5xi32>
}

func.func @test_reshape_reduce_multi_dim(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // 测试多维度 reduce（应该不被简化）
  // CHECK-LABEL: @test_reshape_reduce_multi_dim
  // CHECK: stablehlo.reshape
  // CHECK: stablehlo.reduce
  %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<2x1x1x3xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<f32>
  %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.add across dimensions = [1, 2] : (tensor<2x1x1x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}

func.func @test_reshape_reduce_wrong_dim(%arg0: tensor<2x3xf32>) -> tensor<1x3xf32> {
  // 测试在非新增维度上进行 reduce（应该不被简化）
  // CHECK-LABEL: @test_reshape_reduce_wrong_dim
  // CHECK: stablehlo.reshape
  // CHECK: stablehlo.reduce
  %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<2x1x3xf32>
  %1 = stablehlo.constant dense<0.0> : tensor<f32>
  %2 = stablehlo.reduce(%0 init: %1) applies stablehlo.add across dimensions = [0] : (tensor<2x1x3xf32>, tensor<f32>) -> tensor<1x3xf32>
  return %2 : tensor<1x3xf32>
}