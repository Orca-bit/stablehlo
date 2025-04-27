#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/reference/InterpreterOps.h"
#include "stablehlo/tests/CheckOps.h"
#include "stablehlo/tests/TestUtils.h"
#include "stablehlo/transforms/PassesExt.h"

int main(int argc, char **argv) {
  mlir::stablehlo_ext::registerPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "StableHLO Extensions Optimizer Driver\n", registry));
}