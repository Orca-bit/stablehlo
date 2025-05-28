// RUN: stablehlo-ext-opt --outline-main-function %s | FileCheck %s

// CHECK-LABEL: func.func private @__outlined_func_0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<4xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:    %2 = stablehlo.reduce(%1 init: %arg2) applies stablehlo.add across dimensions = [0, 1] : (tensor<4x1xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    return %2 : tensor<f32>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @main() -> tensor<1xf32> {
func.func @main() -> tensor<1xf32> {
  // CHECK: %[[CST0:.+]] = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %cst0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  // CHECK: %[[CST1:.+]] = stablehlo.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  %cst1 = stablehlo.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  // CHECK: %[[CST_INIT:.+]] = stablehlo.constant dense<0.0> : tensor<f32>
  %cst_init = stablehlo.constant dense<0.0> : tensor<f32>

  // Fusible sequence starts here
  %add = stablehlo.add %cst0, %cst1 : tensor<4xf32>
  %broadcast = stablehlo.broadcast_in_dim %add, dims = [0] : (tensor<4xf32>) -> tensor<4x1xf32>
  %reduce = stablehlo.reduce(%broadcast init: %cst_init) applies stablehlo.add across dimensions = [0, 1] : (tensor<4x1xf32>, tensor<f32>) -> tensor<f32>
  // Fusible sequence ends here

  // CHECK: %[[CALL_RESULT:.+]] = func.call @__outlined_func_0(%[[CST0]], %[[CST1]], %[[CST_INIT]]) : (tensor<4xf32>, tensor<4xf32>, tensor<f32>) -> tensor<f32>

  // Non-fusible operation
  // CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %[[CALL_RESULT]] : (tensor<f32>) -> tensor<1xf32>
  %reshape = stablehlo.reshape %reduce : (tensor<f32>) -> tensor<1xf32>

  // CHECK: return %[[RESHAPE]] : tensor<1xf32>
  return %reshape : tensor<1xf32>
}