#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassesExt.h"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_SIMPLIFYRESHAPEREDUCEPASS
#include "stablehlo/transforms/PassesExt.h.inc"

namespace {
struct SimplifyReshapeReducePattern
    : public OpRewritePattern<stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    // 检查reduce操作的输入是否来自reshape操作
    auto reshapeOp =
        reduceOp.getInputs()[0].getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshapeOp) {
      return failure();
    }

    // 获取reduce的维度，目前仅考虑单维度的情况
    auto reduceDimensions = reduceOp.getDimensions();
    if (reduceDimensions.size() != 1) {
      return failure();
    }

    // 检查reduce操作的初始值
    auto initValues = reduceOp.getInitValues();
    if (initValues.size() != 1) {
      return failure();
    }

    auto initValue = initValues[0];
    auto initConstOp = initValue.getDefiningOp<stablehlo::ConstantOp>();
    if (!initConstOp) {
      return failure();
    }

    // 检查reduce操作的body，确定是加法还是乘法
    auto &bodyRegion = reduceOp.getBody();
    if (bodyRegion.getBlocks().size() != 1) {
      return failure();
    }

    auto &bodyBlock = bodyRegion.front();
    if (bodyBlock.getOperations().size() !=
        2) {  // 应该只有一个操作和一个return
      return failure();
    }

    auto &bodyOp = bodyBlock.front();
    bool isAddition = false;
    bool isMultiplication = false;

    if (auto addOp = dyn_cast<stablehlo::AddOp>(bodyOp)) {
      isAddition = true;
    } else if (auto mulOp = dyn_cast<stablehlo::MulOp>(bodyOp)) {
      isMultiplication = true;
    } else {
      return failure();  // 不支持的reduce操作
    }

    // 验证初始值的正确性
    auto initAttr = initConstOp.getValue();
    if (isAddition) {
      // 加法操作，初始值应该为0
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(initAttr)) {
        // 处理张量常量
        if (denseAttr.isSplat()) {
          if (auto elementType =
                  dyn_cast<FloatType>(denseAttr.getElementType())) {
            if (!denseAttr.getSplatValue<APFloat>().isZero()) {
              return failure();
            }
          } else if (auto elementType =
                         dyn_cast<IntegerType>(denseAttr.getElementType())) {
            if (!denseAttr.getSplatValue<APInt>().isZero()) {
              return failure();
            }
          } else {
            return failure();
          }
        } else {
          return failure();
        }
      } else if (auto floatAttr = dyn_cast<FloatAttr>(initAttr)) {
        // 处理标量浮点常量
        if (!floatAttr.getValue().isZero()) {
          return failure();
        }
      } else if (auto intAttr = dyn_cast<IntegerAttr>(initAttr)) {
        // 处理标量整数常量
        if (!intAttr.getValue().isZero()) {
          return failure();
        }
      } else {
        return failure();
      }
    } else if (isMultiplication) {
      // 乘法操作，初始值应该为1
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(initAttr)) {
        // 处理张量常量
        if (denseAttr.isSplat()) {
          if (auto elementType =
                  dyn_cast<FloatType>(denseAttr.getElementType())) {
            APFloat splatValue = denseAttr.getSplatValue<APFloat>();
            APFloat one(splatValue.getSemantics(), 1);
            if (!splatValue.bitwiseIsEqual(one)) {
              return failure();
            }
          } else if (auto elementType =
                         dyn_cast<IntegerType>(denseAttr.getElementType())) {
            APInt splatValue = denseAttr.getSplatValue<APInt>();
            if (!splatValue.isOne()) {
              return failure();
            }
          } else {
            return failure();
          }
        } else {
          return failure();
        }
      } else if (auto floatAttr = dyn_cast<FloatAttr>(initAttr)) {
        // 处理标量浮点常量
        APFloat value = floatAttr.getValue();
        APFloat one(value.getSemantics(), 1);
        if (!value.bitwiseIsEqual(one)) {
          return failure();
        }
      } else if (auto intAttr = dyn_cast<IntegerAttr>(initAttr)) {
        // 处理标量整数常量
        if (!intAttr.getValue().isOne()) {
          return failure();
        }
      } else {
        return failure();
      }
    }

    // 获取reshape的输入和输出类型
    auto reshapeInputType = reshapeOp.getOperand().getType();
    auto reshapeOutputType = reshapeOp.getResult().getType();
    if (reduceOp->getNumResults() != 1) {
      return failure();
    }
    auto reduceOutputType = reduceOp.getResult(0).getType();

    if (!reshapeInputType || !reshapeOutputType || !reduceOutputType) {
      return failure();
    }

    // 检查是否可以简化：
    // 1. reshape只是添加了大小为1的维度
    // 2. reduce操作正好在这些添加的维度上进行归约

    auto inputShape = reshapeInputType.getShape();
    auto reshapeShape = reshapeOutputType.getShape();

    // 检查reshape是否只是插入了大小为1的维度
    llvm::SmallVector<int64_t> addedDims;
    size_t inputIdx = 0;

    for (size_t i = 0; i < reshapeShape.size(); ++i) {
      if (inputIdx < inputShape.size() &&
          reshapeShape[i] == inputShape[inputIdx]) {
        // 这个维度来自原始输入
        inputIdx++;
      } else if (reshapeShape[i] == 1) {
        // 这是一个新添加的大小为1的维度
        addedDims.push_back(i);
      } else {
        // reshape做了其他变换，不是简单的添加维度
        return failure();
      }
    }

    // 检查所有的reduce维度是否都是新添加的大小为1的维度
    if (addedDims.size() != 1 || addedDims[0] != reduceDimensions[0]) {
      // reduce的维度不完全是新添加的维度，无法简化
      return failure();
    }

    // 验证简化后的形状是否正确
    // 移除reduce维度后的形状应该等于原始输入形状
    llvm::SmallVector<int64_t> expectedShape;
    for (size_t i = 0; i < reshapeShape.size(); ++i) {
      if (reduceDimensions[0] != static_cast<int64_t>(i)) {
        expectedShape.push_back(reshapeShape[i]);
      }
    }

    if (expectedShape.size() != inputShape.size()) {
      return failure();
    }

    for (size_t i = 0; i < expectedShape.size(); ++i) {
      if (expectedShape[i] != inputShape[i]) {
        return failure();
      }
    }

    // 执行简化：直接用reshape的输入替换reduce的结果
    // 因为reshape只是添加了大小为1的维度，然后reduce又把这些维度归约掉
    // 所以最终结果就是原始的reshape输入
    rewriter.replaceOp(reduceOp, reshapeOp.getOperand());

    return success();
  }
};

struct SimplifyReshapeReducePass
    : public impl::SimplifyReshapeReducePassBase<SimplifyReshapeReducePass> {
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    populateSimplifyReshapeReducePatterns(context, &owningPatterns);
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

void populateSimplifyReshapeReducePatterns(MLIRContext *context,
                                           RewritePatternSet *patterns) {
  patterns->add<SimplifyReshapeReducePattern>(context);
}

}  // namespace stablehlo_ext
}  // namespace mlir