include "mlir/Pass/PassBase.td"

def RemoveUnusedBroadcastPass : Pass<"remove-unused-broadcast", "mlir::func::FuncOp"> {
  let summary = "Remove unnecessary broadcast_in_dim operations";
  let description = [{
    This pass identifies and removes broadcast_in_dim operations where the input and output 
    shapes are identical and the dimension order is not changed.
    
    For example, when a broadcast_in_dim operation has the same input and output tensor shape, 
    and the broadcast_dimensions attribute is [0, 1, 2, ..., n-1], the operation doesn't 
    actually change the tensor shape or dimension order, so it can be safely removed.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def FuseTransposeReshapeTransposePass : Pass<"fuse-transpose-reshape-transpose", "mlir::func::FuncOp"> {
  let summary = "Simplify TransposeReshapeTranspose to TransposeReshape";
  let description = [{
    This pass identifies sequences of transpose -> reshape -> transpose operations
    and attempts to simplify them into a single transpose -> reshape sequence
    where possible. This optimization can reduce computational overhead and
    improve performance by eliminating redundant data movement.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def ConvertIotaToConstantPass : Pass<"convert-iota-to-constant", "mlir::func::FuncOp"> {
  let summary = "Convert iota to constant when shape is known";
  let description = [{
    This pass identifies stablehlo.iota operations with a statically known output shape
    and replaces them with an equivalent stablehlo.constant operation.
    This materializes the iota sequence into a constant tensor.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def OutlineMainFunctionPass : Pass<"outline-main-function", "mlir::func::FuncOp"> {
  let summary = "Outline the main function into smaller callable functions";
  let description = [{
    This pass identifies the main function (typically 'main') and outlines its body
    into smaller, potentially device-specific functions. Each outlined function
    can then be targeted for kernel generation by the compiler backend.
    This facilitates modular compilation and potential parallel execution.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect", "mlir::func::FuncDialect"];
}

def SimplifyReshapeReducePass : Pass<"simplify-reshape-reduce", "mlir::func::FuncOp"> {
  let summary = "Simplify ineffective reshape+reduce patterns";
  let description = [{
    This pass identifies and simplifies ineffective reshape+reduce operation patterns.
    It looks for sequences where a reshape operation is immediately followed by a reduce
    operation, and determines if the reshape can be eliminated without affecting the
    final result.
    
    For example:
    - reshape(tensor<100x64xf32>) -> tensor<100x1x64xf32> followed by 
      reduce(dims=[1]) can be simplified by removing the reshape entirely
    - reshape operations that only add dimensions of size 1 that are immediately
      reduced can be eliminated
    
    This optimization reduces memory usage and computational overhead by eliminating
    unnecessary tensor shape transformations before reduction operations.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def ConvertGatherIndexTypePass : Pass<"convert-gather-index-type", "mlir::func::FuncOp"> {
  let summary = "Convert gather index types from unsigned to signed integers";
  let description = [{
    This pass converts the index input types of stablehlo.gather operations from
    unsigned integer types to their corresponding signed integer types:
    - uint32 -> int32
    - uint64 -> int64
    
    This conversion is useful for compatibility with backends or frameworks that
    expect signed integer indices for gather operations. The pass ensures that
    the semantic meaning of the indices is preserved during the type conversion.
    
    For example:
    - gather(operand, indices: tensor<4xui32>) -> gather(operand, indices: tensor<4xi32>)
    - gather(operand, indices: tensor<2x3xui64>) -> gather(operand, indices: tensor<2x3xi64>)
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}
