// RUN: stablehlo-opt %s --convert-iota-to-constant | FileCheck %s

// -----
// CHECK-LABEL: @test_iota_1d
// CHECK-SAME: () -> tensor<5xi32>
func.func @test_iota_1d() -> tensor<5xi32> {
  // CHECK: stablehlo.constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %0 = stablehlo.iota dim = 0 : tensor<5xi32> // Renamed attribute
  return %0 : tensor<5xi32>
}

// -----
// CHECK-LABEL: @test_iota_2d_dim0
// CHECK-SAME: () -> tensor<2x3xi32>
func.func @test_iota_2d_dim0() -> tensor<2x3xi32> {
  // CHECK: stablehlo.constant dense<[[0, 0, 0], [1, 1, 1]]> : tensor<2x3xi32>
  %0 = stablehlo.iota dim = 0 : tensor<2x3xi32> // Renamed attribute
  return %0 : tensor<2x3xi32>
}

// -----
// CHECK-LABEL: @test_iota_2d_dim1
// CHECK-SAME: () -> tensor<2x3xi32>
func.func @test_iota_2d_dim1() -> tensor<2x3xi32> {
  // CHECK: stablehlo.constant dense<[[0, 1, 2], [0, 1, 2]]> : tensor<2x3xi32>
  %0 = stablehlo.iota dim = 1 : tensor<2x3xi32> // Renamed attribute
  return %0 : tensor<2x3xi32>
}

// -----
// CHECK-LABEL: @test_iota_3d_dim1
// CHECK-SAME: () -> tensor<2x3x2xi32>
func.func @test_iota_3d_dim1() -> tensor<2x3x2xi32> {
  // CHECK: stablehlo.constant dense<[[[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]]]> : tensor<2x3x2xi32>
  %0 = stablehlo.iota dim = 1 : tensor<2x3x2xi32> // Renamed attribute
  return %0 : tensor<2x3x2xi32>
}

// -----
// CHECK-LABEL: @test_iota_3d_dim0
// CHECK-SAME: () -> tensor<2x3x2xi32>
func.func @test_iota_3d_dim0() -> tensor<2x3x2xi32> {
  // CHECK: stablehlo.constant dense<[[[0, 0], [0, 0], [0, 0]], [[1, 1], [1, 1], [1, 1]]]> : tensor<2x3x2xi32>
  %0 = stablehlo.iota dim = 0 : tensor<2x3x2xi32>
  return %0 : tensor<2x3x2xi32>
}

// -----
// CHECK-LABEL: @test_iota_3d_dim2
// CHECK-SAME: () -> tensor<2x3x2xi32>
func.func @test_iota_3d_dim2() -> tensor<2x3x2xi32> {
  // CHECK: stablehlo.constant dense<[[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]> : tensor<2x3x2xi32>
  %0 = stablehlo.iota dim = 2 : tensor<2x3x2xi32> // Renamed attribute
  return %0 : tensor<2x3x2xi32>
}

// -----
// CHECK-LABEL: @test_iota_f32
// CHECK-SAME: () -> tensor<2x2xf32>
func.func @test_iota_f32() -> tensor<2x2xf32> {
  // CHECK: stablehlo.constant dense<[[0.000000e+00, 1.000000e+00], [0.000000e+00, 1.000000e+00]]> : tensor<2x2xf32>
  %0 = stablehlo.iota dim = 1 : tensor<2x2xf32> // Renamed attribute
  return %0 : tensor<2x2xf32>
}
