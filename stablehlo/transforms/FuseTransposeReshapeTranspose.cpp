#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassesExt.h"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_FUSETRANSPOSERESHAPETRANSPOSEPASS
#include "stablehlo/transforms/PassesExt.h.inc"

namespace {
using stablehlo::ReshapeOp;
using stablehlo::TransposeOp;

// Helper function to check if dimensions are merged contiguously
bool isContiguousMerge(ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape,
                       int &mergePos, int &mergeStart, int &mergeEnd) {
  if (inShape.size() <= outShape.size()) return false;

  mergePos = -1;
  mergeStart = -1;
  int currentInDim = 0;
  for (unsigned i = 0; i < outShape.size(); ++i) {
    if (currentInDim >= static_cast<int>(inShape.size()))
      return false;  // Output has more dims than input processed

    if (inShape[currentInDim] == outShape[i]) {
      currentInDim++;
      continue;
    }

    // Found a potential merge point
    if (mergePos != -1) return false;  // Only one merge allowed for now

    mergePos = i;
    mergeStart = currentInDim;
    int64_t mergedSize = 1;
    while (currentInDim < static_cast<int>(inShape.size())) {
      mergedSize *= inShape[currentInDim];
      currentInDim++;
      if (mergedSize == outShape[i]) break;  // Found the end of the merge
      if (mergedSize > outShape[i])
        return false;  // Overshot, not a simple merge
    }
    if (mergedSize != outShape[i]) return false;  // Didn't find a valid merge
    mergeEnd = currentInDim - 1;                  // Inclusive end index
  }
  // Ensure all input dimensions were consumed
  if (currentInDim != static_cast<int>(inShape.size())) return false;

  return mergePos != -1;  // A merge must have happened
}

// clang-format off
// before pass:
// %2153 = stablehlo.transpose %2152, dims = [0, 2, 1, 3] : (tensor<128x30x16x32xf16>) -> tensor<128x16x30x32xf16> 
// %2154 = stablehlo.reshape %2153 : (tensor<128x16x30x32xf16>) -> tensor<128x16x960xf16>
// %2155 = stablehlo.transpose %2154, dims = [1, 0, 2] : (tensor<128x16x960xf16>) -> tensor<16x128x960xf16>
//
// after pass:
// %2153 = stablehlo.transpose %2152, dims = [2, 0, 1, 3] : (tensor<128x30x16x32xf16>) -> tensor<16x128x30x32xf16>
// %2154 = stablehlo.reshape %2153 : (tensor<16x128x30x32xf16>) -> tensor<16x128x960xf16>
// clang-format on
// NOTE 支持 reshape 合并任意连续维度
struct FuseTransposeReshapeTransposePattern final
    : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp transpose2,
                                PatternRewriter &rewriter) const override {
    // 匹配 transpose(reshape(transpose)) 模式
    auto reshapeOp = transpose2.getOperand().getDefiningOp<ReshapeOp>();
    if (!reshapeOp) return failure();

    auto transpose1 = reshapeOp.getOperand().getDefiningOp<TransposeOp>();
    if (!transpose1) return failure();

    // 验证reshape是合并连续维度 (可以是任意位置)
    auto inShape = transpose1.getType().getShape();
    auto outShape = reshapeOp.getType().getShape();

    int mergePos = -1, mergeStart = -1, mergeEnd = -1;
    if (!isContiguousMerge(inShape, outShape, mergePos, mergeStart, mergeEnd)) {
      return failure();
    }

    // 获取permutation参数
    auto p1 = transpose1.getPermutation();
    auto p2 = transpose2.getPermutation();

    // Find which output dimension of transpose2 corresponds to the merged
    // dimension
    int mergedDimInT2Output = -1;
    for (size_t i = 0; i < p2.size(); ++i) {
      if (p2[i] == mergePos) {
        mergedDimInT2Output = i;
        break;
      }
    }
    if (mergedDimInT2Output == -1) {
      // This case might be complex or invalid, fail for now.
      // It means the merged dimension is dropped by the second transpose?
      return failure();
    }

    // 构建新permutation：将最终输出维度映射回原始输入维度
    SmallVector<int64_t> newPerm;
    SmallVector<int64_t> t1OutIdxToOrgInIdx(p1.size());
    for (size_t i = 0; i < p1.size(); ++i) {
      t1OutIdxToOrgInIdx[i] = p1[i];
    }

    for (int64_t t2OutDimIdx :
         p2) {  // Iterate through dims of reshape output referenced by p2
      if (t2OutDimIdx < mergePos) {
        // Dimension before the merge
        newPerm.push_back(t1OutIdxToOrgInIdx[t2OutDimIdx]);
      } else if (t2OutDimIdx == mergePos) {
        // The merged dimension: expand back to original dims via p1
        for (int k = mergeStart; k <= mergeEnd; ++k) {
          newPerm.push_back(t1OutIdxToOrgInIdx[k]);
        }
      } else {  // t2OutDimIdx > mergePos
        // Dimension after the merge
        // Calculate corresponding index in transpose1's output shape
        int t1OutDimIdx = mergeEnd + 1 + (t2OutDimIdx - (mergePos + 1));
        if (t1OutDimIdx >= static_cast<int>(t1OutIdxToOrgInIdx.size()))
          return failure();  // Index out of bounds
        newPerm.push_back(t1OutIdxToOrgInIdx[t1OutDimIdx]);
      }
    }

    // 计算新的转置操作后的形状
    auto inputShape = transpose1.getOperand().getType().getShape();
    SmallVector<int64_t> newTransposeShape;
    // for (auto dim : newPerm) {
    //   newTransposeShape.push_back(inputShape[dim]);
    // }
    for (auto dim : newPerm) {
      if (dim < 0 || dim >= static_cast<int64_t>(inputShape.size()))
        return failure();  // Invalid dim index
      newTransposeShape.push_back(inputShape[dim]);
    }

    // 创建新的返回类型
    auto newTransposeType = RankedTensorType::get(
        newTransposeShape, transpose1.getOperand().getType().getElementType());

    // 创建优化后的操作
    auto newTranspose = rewriter.create<TransposeOp>(
        transpose1.getLoc(), newTransposeType, transpose1.getOperand(),
        DenseI64ArrayAttr::get(rewriter.getContext(), newPerm));

    // 构建新reshape形状（保持合并后的维度）
    auto newReshapeType = RankedTensorType::get(
        transpose2.getType().getShape(),
        transpose1.getOperand().getType().getElementType());
    auto newReshape = rewriter.create<ReshapeOp>(reshapeOp.getLoc(),
                                                 newReshapeType, newTranspose);

    rewriter.replaceOp(transpose2, newReshape.getResult());
    return success();
  }
};

struct FuseTransposeReshapeTransposePass final
    : impl::FuseTransposeReshapeTransposePassBase<
          FuseTransposeReshapeTransposePass> {
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    populateFuseTransposeReshapeTransposePatterns(context, &owningPatterns);
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

}  // namespace

void populateFuseTransposeReshapeTransposePatterns(
    MLIRContext *context, RewritePatternSet *patterns) {
  patterns->add<FuseTransposeReshapeTransposePattern>(context);
}

}  // namespace stablehlo_ext
}  // namespace mlir