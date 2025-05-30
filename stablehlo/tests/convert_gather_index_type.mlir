// RUN: stablehlo-ext-opt %s --convert-gather-index-type | FileCheck %s

// Test uint32 to int32 conversion with constant indices
// CHECK-LABEL: func.func @gather_uint32_constant_indices
func.func @gather_uint32_constant_indices(%arg0: tensor<4x2xf32>) -> tensor<2xf32> {
  %indices = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xui32>
  // CHECK: %[[INDICES:.*]] = stablehlo.constant dense<{{.*}}> : tensor<2x1xi32>
  // CHECK: %[[RESULT:.*]] = "stablehlo.gather"(%arg0, %[[INDICES]])
  %result = "stablehlo.gather"(%arg0, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    indices_are_sorted = false
  } : (tensor<4x2xf32>, tensor<2x1xui32>) -> tensor<2x2xf32>
  %slice = "stablehlo.slice"(%result) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 2, 1>,
    strides = array<i64: 1, 1>
  } : (tensor<2x2xf32>) -> tensor<2x1xf32>
  %reshape = "stablehlo.reshape"(%slice) : (tensor<2x1xf32>) -> tensor<2xf32>
  return %reshape : tensor<2xf32>
}

// Test uint64 to int64 conversion with constant indices
// CHECK-LABEL: func.func @gather_uint64_constant_indices
func.func @gather_uint64_constant_indices(%arg0: tensor<8x3xi32>) -> tensor<3x3xi32> {
  %indices = stablehlo.constant dense<[[1], [3], [5]]> : tensor<3x1xui64>
  // CHECK: %[[INDICES:.*]] = stablehlo.constant dense<{{.*}}> : tensor<3x1xi64>
  // CHECK: %[[RESULT:.*]] = "stablehlo.gather"(%arg0, %[[INDICES]])
  %result = "stablehlo.gather"(%arg0, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 3>,
    indices_are_sorted = false
  } : (tensor<8x3xi32>, tensor<3x1xui64>) -> tensor<3x3xi32>
  return %result : tensor<3x3xi32>
}

// Test uint32 to int32 conversion with non-constant indices
// CHECK-LABEL: func.func @gather_uint32_non_constant_indices
func.func @gather_uint32_non_constant_indices(%arg0: tensor<6x4xf16>, %arg1: tensor<2x1xui32>) -> tensor<2x4xf16> {
  // CHECK: %[[CONVERTED:.*]] = stablehlo.convert %arg1 : (tensor<2x1xui32>) -> tensor<2x1xi32>
  // CHECK: %[[RESULT:.*]] = "stablehlo.gather"(%arg0, %[[CONVERTED]])
  %result = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 4>,
    indices_are_sorted = false
  } : (tensor<6x4xf16>, tensor<2x1xui32>) -> tensor<2x4xf16>
  return %result : tensor<2x4xf16>
}

// Test uint64 to int64 conversion with non-constant indices
// CHECK-LABEL: func.func @gather_uint64_non_constant_indices
func.func @gather_uint64_non_constant_indices(%arg0: tensor<10x5xi64>, %arg1: tensor<3x1xui64>) -> tensor<3x5xi64> {
  // CHECK: %[[CONVERTED:.*]] = stablehlo.convert %arg1 : (tensor<3x1xui64>) -> tensor<3x1xi64>
  // CHECK: %[[RESULT:.*]] = "stablehlo.gather"(%arg0, %[[CONVERTED]])
  %result = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 5>,
    indices_are_sorted = false
  } : (tensor<10x5xi64>, tensor<3x1xui64>) -> tensor<3x5xi64>
  return %result : tensor<3x5xi64>
}

// Test that int32 indices are not converted (negative test)
// CHECK-LABEL: func.func @gather_int32_no_conversion
func.func @gather_int32_no_conversion(%arg0: tensor<4x3xf32>, %arg1: tensor<2x1xi32>) -> tensor<2x3xf32> {
  // CHECK-NOT: stablehlo.convert
  // CHECK: %[[RESULT:.*]] = "stablehlo.gather"(%arg0, %arg1)
  %result = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 3>,
    indices_are_sorted = false
  } : (tensor<4x3xf32>, tensor<2x1xi32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// Test that int64 indices are not converted (negative test)
// CHECK-LABEL: func.func @gather_int64_no_conversion
func.func @gather_int64_no_conversion(%arg0: tensor<5x2xi32>, %arg1: tensor<3x1xi64>) -> tensor<3x2xi32> {
  // CHECK-NOT: stablehlo.convert
  // CHECK: %[[RESULT:.*]] = "stablehlo.gather"(%arg0, %arg1)
  %result = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    indices_are_sorted = false
  } : (tensor<5x2xi32>, tensor<3x1xi64>) -> tensor<3x2xi32>
  return %result : tensor<3x2xi32>
}

// Test multi-dimensional indices with uint32 to int32 conversion
// CHECK-LABEL: func.func @gather_multi_dim_uint32_indices
func.func @gather_multi_dim_uint32_indices(%arg0: tensor<4x6x8xf32>) -> tensor<2x3x8xf32> {
  %indices = stablehlo.constant dense<[[[0, 1], [1, 2], [2, 3]], [[1, 0], [2, 1], [3, 2]]]> : tensor<2x3x2xui32>
  // CHECK: %[[INDICES:.*]] = stablehlo.constant dense<{{.*}}> : tensor<2x3x2xi32>
  // CHECK: %[[RESULT:.*]] = "stablehlo.gather"(%arg0, %[[INDICES]])
  %result = "stablehlo.gather"(%arg0, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<4x6x8xf32>, tensor<2x3x2xui32>) -> tensor<2x3x8xf32>
  return %result : tensor<2x3x8xf32>
}
