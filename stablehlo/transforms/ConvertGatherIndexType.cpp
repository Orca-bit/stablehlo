#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassesExt.h"

namespace mlir {
namespace stablehlo_ext {
#define GEN_PASS_DEF_CONVERTGATHERINDEXTYPEPASS
#include "stablehlo/transforms/PassesExt.h.inc"

namespace {

struct ConvertGatherIndexTypePattern
    : public OpRewritePattern<stablehlo::GatherOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto startIndices = op.getStartIndices();
    auto startIndicesType = startIndices.getType();
    auto elementType = startIndicesType.getElementType();

    // 检查是否需要转换类型
    Type targetType;
    if (elementType.isUnsignedInteger(32)) {
      targetType = rewriter.getI32Type();
    } else if (elementType.isUnsignedInteger(64)) {
      targetType = rewriter.getI64Type();
    } else {
      // 不需要转换
      return failure();
    }

    // 创建新的张量类型
    auto newStartIndicesType =
        RankedTensorType::get(startIndicesType.getShape(), targetType);

    Value newStartIndices;

    // 检查输入是否为常量
    if (auto constantOp = startIndices.getDefiningOp<stablehlo::ConstantOp>()) {
      // 直接修改常量的类型
      auto denseAttr = cast<DenseIntElementsAttr>(constantOp.getValue());

      // 提取原始值并转换为新类型
      SmallVector<APInt> newValues;
      for (auto value : denseAttr.getValues<APInt>()) {
        // 创建新的APInt，保持相同的数值但使用新的位宽
        if (targetType.isSignedInteger()) {
          newValues.push_back(APInt(targetType.getIntOrFloatBitWidth(),
                                    value.getSExtValue(), /*isSigned=*/true));
        } else {
          newValues.push_back(APInt(targetType.getIntOrFloatBitWidth(),
                                    value.getZExtValue(), /*isSigned=*/false));
        }
      }

      // 创建新的DenseIntElementsAttr
      auto newDenseAttr =
          DenseIntElementsAttr::get(newStartIndicesType, newValues);

      // 创建新的常量操作
      newStartIndices = rewriter.create<stablehlo::ConstantOp>(
          constantOp.getLoc(), newStartIndicesType, newDenseAttr);
    } else {
      // 插入类型转换操作
      newStartIndices = rewriter.create<stablehlo::ConvertOp>(
          op.getLoc(), newStartIndicesType, startIndices);
    }

    // 创建新的gather操作
    auto newGatherOp = rewriter.create<stablehlo::GatherOp>(
        op.getLoc(), op.getType(), op.getOperand(), newStartIndices,
        op.getDimensionNumbers(), op.getSliceSizes(), op.getIndicesAreSorted());

    rewriter.replaceOp(op, newGatherOp.getResult());
    return success();
  }
};

struct ConvertGatherIndexTypePass
    : public impl::ConvertGatherIndexTypePassBase<ConvertGatherIndexTypePass> {
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    populateConvertGatherIndexTypePatterns(context, &owningPatterns);
    patterns = std::move(owningPatterns);
    return success();
  }

  void runOnOperation() override {
    if (failed(applyPatternsGreedily(getOperation(), patterns))) {
      return signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
};
}  // namespace

void populateConvertGatherIndexTypePatterns(MLIRContext *context,
                                            RewritePatternSet *patterns) {
  patterns->add<ConvertGatherIndexTypePattern>(context);
}
}  // namespace stablehlo_ext
}  // namespace mlir