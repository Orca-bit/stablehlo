#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassesExt.h"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_REMOVEUNUSEDBROADCASTPASS
#include "stablehlo/transforms/PassesExt.h.inc"

namespace {

// Define a rewrite pattern to eliminate broadcast_in_dim operations where input
// and output shapes are identical
struct RemoveRedundantBroadcast final
    : OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  
  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Get input and output types
    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand().getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());

    // If not RankedTensorType, don't rewrite
    if (!inputType || !resultType)
      return mlir::failure();

    // Check if input and output shapes are identical
    if (inputType.getShape() == resultType.getShape()) {
      // Check if dims attribute would cause dimension reordering
      auto dimsAttr = op.getBroadcastDimensions();

      // Check if dims is [0, 1, 2, ..., n-1]
      bool isSequential = true;
      for (size_t i = 0; i < dimsAttr.size(); ++i) {
        if (dimsAttr[i] != static_cast<int64_t>(i)) {
          isSequential = false;
          break;
        }
      }

      // Only replace the operation if dims is sequential and shapes are
      // identical
      if (isSequential) {
        // Shapes are identical and dimension order is unchanged, directly
        // replace with input
        rewriter.replaceOp(op, op.getOperand());
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

struct RemoveUnusedBroadcastPass final
    : impl::RemoveUnusedBroadcastPassBase<RemoveUnusedBroadcastPass> {
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    populateRemoveRedundantBroadcastPatterns(context, &owningPatterns);
    patterns = std::move(owningPatterns);
    return success();
  }

  void runOnOperation() override {
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patterns))) {
      return signalPassFailure();
    }
  }

private:
  FrozenRewritePatternSet patterns;
};
} // end anonymous namespace

void populateRemoveRedundantBroadcastPatterns(MLIRContext *context,
                                              RewritePatternSet *patterns) {
  patterns->add<RemoveRedundantBroadcast>(context);
}
} // namespace stablehlo_ext
} // namespace mlir
