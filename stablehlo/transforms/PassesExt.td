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