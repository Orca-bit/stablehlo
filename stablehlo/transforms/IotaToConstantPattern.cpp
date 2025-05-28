#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassesExt.h"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_CONVERTIOTATOCONSTANTPASS
#include "stablehlo/transforms/PassesExt.h.inc"

namespace {
// Helper function to calculate iota values
template <typename T>
void computeIotaValues(ArrayRef<int64_t> shape, size_t iotaDimension,
                       SmallVectorImpl<T> &values) {
  size_t rank = shape.size();
  size_t numElements = 1;
  for (size_t dim : shape) {
    // Use saturating multiply to prevent overflow
    numElements = llvm::SaturatingMultiply(numElements, dim);
  }
  values.resize(numElements);

  if (numElements == 0) return;

  // Calculate stride for the iota dimension
  size_t stride = 1;
  for (size_t i = iotaDimension + 1; i < rank; ++i) {
    stride = llvm::SaturatingMultiply(stride, static_cast<size_t>(shape[i]));
  }

  for (size_t i = 0; i < numElements; ++i) {
    size_t indexValue = (i / stride) % shape[iotaDimension];
    if constexpr (std::is_floating_point_v<T>) {
      values[i] = static_cast<T>(indexValue);
    } else if constexpr (std::is_integral_v<T>) {
      // Ensure the value fits within the target integer type if necessary
      // For standard types like int32_t, int64_t, direct assignment is fine
      // as indexValue is int64_t. For smaller types, check bounds if needed.
      values[i] = static_cast<T>(indexValue);
    }
  }
}

struct ConvertIotaToConstantPattern final
    : OpRewritePattern<stablehlo::IotaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::IotaOp op,
                                PatternRewriter &rewriter) const override {
    // 1. 检查输出类型和静态形状
    auto resultType = op.getType();
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires static shape");
    }

    ArrayRef<int64_t> shape = resultType.getShape();
    Type elementType = resultType.getElementType();
    size_t iotaDimension = op.getIotaDimension();
    size_t rank = shape.size();

    // 2. 检查 iota_dimension 是否有效
    if (iotaDimension < 0 || iotaDimension >= rank) {
      return rewriter.notifyMatchFailure(op, "invalid iota_dimension");
    }

    // 3. 计算数值并创建 DenseElementsAttr
    DenseElementsAttr valueAttr;
    if (elementType.isInteger(32)) {
      SmallVector<int32_t> values;
      computeIotaValues<int32_t>(shape, iotaDimension, values);
      valueAttr = DenseElementsAttr::get(resultType, ArrayRef(values));
    } else if (elementType.isInteger(64)) {
      SmallVector<int64_t> values;
      computeIotaValues<int64_t>(shape, iotaDimension, values);
      valueAttr = DenseElementsAttr::get(resultType, ArrayRef(values));
    } else if (elementType.isF32()) {
      SmallVector<float> values;
      computeIotaValues<float>(shape, iotaDimension, values);
      valueAttr = DenseElementsAttr::get(resultType, ArrayRef(values));
    } else if (elementType.isF64()) {
      SmallVector<double> values;
      computeIotaValues<double>(shape, iotaDimension, values);
      valueAttr = DenseElementsAttr::get(resultType, ArrayRef(values));
    }
    // TODO: 可以添加对其他类型 (如 bf16, i8, i16 等) 的支持
    else {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    if (!valueAttr) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to create DenseElementsAttr");
    }

    // 4. 创建 ConstantOp 并替换
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, valueAttr);
    return success();
  }
};

struct ConvertIotaToConstantPass final
    : impl::ConvertIotaToConstantPassBase<ConvertIotaToConstantPass> {
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    populateIotaToConstantPatterns(context, &owningPatterns);
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

void populateIotaToConstantPatterns(MLIRContext *context,
                                    RewritePatternSet *patterns) {
  patterns->add<ConvertIotaToConstantPattern>(context);
}

}  // namespace stablehlo_ext
}  // namespace mlir