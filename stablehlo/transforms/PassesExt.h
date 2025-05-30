#ifndef STABLEHLO_TRANSFORMS_PASSES_EXT_H
#define STABLEHLO_TRANSFORMS_PASSES_EXT_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "stablehlo/transforms/PassesExt.h.inc"

void populateRemoveRedundantBroadcastPatterns(MLIRContext *context,
                                              RewritePatternSet *patterns);

void populateFuseTransposeReshapeTransposePatterns(MLIRContext *context,
                                                   RewritePatternSet *patterns);

void populateIotaToConstantPatterns(MLIRContext *context,
                                    RewritePatternSet *patterns);

void populateSimplifyReshapeReducePatterns(MLIRContext *context,
                                           RewritePatternSet *patterns);

void populateConvertGatherIndexTypePatterns(MLIRContext *context,
                                            RewritePatternSet *patterns);

}  // namespace stablehlo_ext
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_PASSES_EXT_H