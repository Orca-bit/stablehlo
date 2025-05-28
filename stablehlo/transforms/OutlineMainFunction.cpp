#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"        // 需要 StringRef
#include "llvm/Support/raw_ostream.h"  // 用于调试打印 (可选)
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"   // 需要包含 Builders
#include "mlir/IR/IRMapping.h"  // 可能需要 IRMapping
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"                   // 需要 SymbolTable
#include "mlir/Interfaces/SideEffectInterfaces.h"  // 需要 SideEffectInterfaces
#include "mlir/Pass/Pass.h"                        // 需要包含 Pass.h
#include "mlir/Support/LogicalResult.h"            // 需要 LogicalResult
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_OUTLINEMAINFUNCTIONPASS
#include "stablehlo/transforms/PassesExt.h.inc"

namespace {

// 辅助函数：检查操作是否是可融合的类型
bool isFusibleOp(Operation *op) {
  // 检查是否是 Elementwise 操作
  if (op->hasTrait<OpTrait::Elementwise>()) {
    return true;
  }
  // 检查是否是 Broadcast 操作
  if (isa<stablehlo::BroadcastInDimOp>(op)) {
    return true;
  }
  // 检查是否是 Reduce 操作 (注意：可能有多种 Reduce 操作)
  if (isa<stablehlo::ReduceOp, stablehlo::ReduceWindowOp>(op)) {
    // 注意：Reduce 操作的融合可能更复杂，需要考虑其 body
    // 这里暂时简单地将其视为可融合
    return true;
  }
  // 其他需要融合的操作可以加在这里
  return false;
}

// 用于封装 Outline 逻辑的结构体
struct MainFunctionOutliner {
  func::FuncOp mainFunc;
  MLIRContext *context;
  ModuleOp module;
  SymbolTable symbolTable;      // 用于管理新函数的符号
  int outlinedFuncCounter = 0;  // 用于生成新函数名

  MainFunctionOutliner(func::FuncOp func)
      : mainFunc(func),
        context(func.getContext()),
        module(func->getParentOfType<ModuleOp>()),
        symbolTable(module) {}

  // 执行 Outline 的主要逻辑
  LogicalResult run() {
    llvm::outs() << "Running Outline Logic for: " << mainFunc.getName() << "\n";

    if (mainFunc.getBlocks().size() > 1) {
      llvm::errs() << "Main function has multiple blocks, skipping.\n";
      return failure();
    }

    Block &entryBlock = mainFunc.front();
    bool changed = false;

    // 使用迭代器遍历，因为我们可能会修改 Block
    for (auto it = entryBlock.begin(); it != entryBlock.end();
         /* no increment here */) {
      Operation *startOp = &(*it);

      // 检查当前操作是否是可融合序列的开始
      if (!isFusibleOp(startOp)) {
        ++it;
        continue;
      }

      // 找到连续的可融合操作序列
      SmallVector<Operation *> fusibleSequence;
      Operation *currentOp = startOp;
      while (currentOp != nullptr && isFusibleOp(currentOp)) {
        // 检查操作是否安全移动 (没有副作用，或者副作用可控)
        // 对于 elementwise/broadcast/reduce 通常是安全的
        if (!isMemoryEffectFree(currentOp)) {
          // 如果遇到有副作用的操作，停止当前序列
          // 或者根据具体策略决定是否包含
          break;
        }
        fusibleSequence.push_back(currentOp);

        // 检查下一个操作是否由当前操作产生且只有一个用户
        // 这是一个简单的连续性检查，可能需要更复杂的图遍历
        if (currentOp->getNumResults() == 1 &&
            currentOp->getResult(0).hasOneUse()) {
          Operation *nextUser = *currentOp->getResult(0).getUsers().begin();
          // 确保下一个用户在同一个 Block 中
          if (nextUser->getBlock() == &entryBlock) {
            currentOp = nextUser;
          } else {
            currentOp = nullptr;  // 跨 Block 了，停止
          }
        } else {
          currentOp = nullptr;  // 结果不用或多用，或无结果，停止
        }
      }

      // 如果序列只有一个操作，可能不值得 Outline，跳过
      if (fusibleSequence.size() <= 1) {
        it = std::next(
            Block::iterator(fusibleSequence.back()));  // 移动迭代器到序列之后
        continue;
      }

      llvm::outs() << "Found fusible sequence of size: "
                   << fusibleSequence.size() << " starting at " << *startOp
                   << "\n";

      // --- 实际的 Outline 逻辑 ---
      FailureOr<func::CallOp> outlineResult = outlineSequence(fusibleSequence);
      if (failed(outlineResult)) {
        startOp->emitError("Failed to outline fusible sequence");
        return failure();  // Propagate failure
      }

      changed = true;
      func::CallOp newCallOp = *outlineResult;
      // Outline 成功后，原序列的操作已被移动或替换
      // 迭代器需要重置或小心地更新。最简单的方式是重新开始遍历。
      // 但为了效率，理想情况下应该更新迭代器指向 CallOp 之后。
      // 这里为了简单，我们假设 outlineSequence 内部处理了迭代器或替换
      // 或者我们可以在 outlineSequence 返回新 CallOp 的迭代器位置。
      // 暂时先移动到原序列最后一个操作之后的位置
      it = std::next(Block::iterator(newCallOp));

      // 注意：如果 outlineSequence 删除了
      // fusibleSequence.back()，上面的迭代器会失效。 更安全的方式是在
      // outlineSequence 中返回下一个有效的迭代器位置。 或者，在 outlineSequence
      // 替换操作后，让迭代器指向新插入的 CallOp，然后 ++it。
      // **这里需要根据 outlineSequence 的具体实现来调整迭代器管理**
    }

    return success(changed);  // 返回是否修改了 IR
  }

  // 将识别出的操作序列提取到新函数中
  FailureOr<func::CallOp> outlineSequence(
      const SmallVector<Operation *> &sequence) {
    if (sequence.empty()) return failure();

    OpBuilder builder(context);
    Location loc = sequence.front()->getLoc();  // 使用第一个操作的位置

    // 1. 确定新函数的输入和输出
    llvm::SetVector<Value> inputs;  // 使用 SetVector 保持顺序并去重
    llvm::SetVector<Value> outputs;
    llvm::SmallVector<Type> inputTypes;
    llvm::SmallVector<Type> outputTypes;
    llvm::DenseMap<Value, Value>
        externalToInternalMapping;  // 外部值到新函数参数的映射

    // 遍历序列，找出外部依赖（输入）和外部使用（输出）
    for (Operation *op : sequence) {
      for (Value operand : op->getOperands()) {
        Operation *definingOp = operand.getDefiningOp();
        // 如果操作数定义在序列之外，或者是一个 Block 参数，则为输入
        if (!definingOp || std::find(sequence.begin(), sequence.end(),
                                     definingOp) == sequence.end()) {
          if (inputs.insert(operand)) {  // 首次遇到该输入
            inputTypes.push_back(operand.getType());
          }
        }
      }

      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          // 如果结果被序列之外的操作使用，则为输出
          if (std::find(sequence.begin(), sequence.end(), user) ==
              sequence.end()) {
            if (outputs.insert(result)) {  // 首次遇到该输出
              outputTypes.push_back(result.getType());
            }
            break;  // 只要有一个外部用户，就是输出
          }
        }
      }
    }

    // 2. 创建新函数
    std::string newFuncName =
        "__outlined_func_" + std::to_string(outlinedFuncCounter++);
    auto funcType = FunctionType::get(context, inputTypes, outputTypes);

    // 将新函数插入到模块中，确保名称唯一
    builder.setInsertionPoint(module.getBody(),
                              module.getBody()->end());  // 插在模块末尾
    auto newFunc = builder.create<func::FuncOp>(loc, newFuncName, funcType);
    newFunc.setPrivate();         // 通常设为私有，除非需要外部调用
    symbolTable.insert(newFunc);  // 注册到符号表

    // 3. 创建新函数的入口 Block，并添加参数
    Block *newFuncBody = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(newFuncBody);

    // 建立外部输入到新函数参数的映射
    for (size_t i = 0; i < inputs.size(); ++i) {
      externalToInternalMapping[inputs[i]] = newFuncBody->getArgument(i);
    }

    // 4. 移动操作到新函数，并更新操作数
    IRMapping valueMapping;  // 用于内部值重映射
    for (Value input : inputs) {
      valueMapping.map(input, externalToInternalMapping[input]);
    }

    for (Operation *opToMove : sequence) {
      // 使用 cloneAndRemap 而不是直接移动，这样更容易处理值的映射
      Operation *clonedOp = builder.clone(*opToMove, valueMapping);
      // valueMapping 会自动更新 clonedOp 结果的映射
      llvm::outs() << "  Cloned op: " << *clonedOp << "\n";
    }

    // 5. 添加返回操作
    SmallVector<Value> returnValues;
    for (Value output : outputs) {
      returnValues.push_back(valueMapping.lookup(output));  // 查找映射后的值
    }
    builder.create<func::ReturnOp>(loc, returnValues);

    // 6. 在 main 函数中插入 CallOp
    builder.setInsertionPoint(sequence.front());  // 插在原序列第一个操作之前
    SmallVector<Value> callOperands;
    for (Value input : inputs) {
      callOperands.push_back(input);  // 使用原始的外部输入值
    }
    auto callOp = builder.create<func::CallOp>(loc, newFunc, callOperands);

    // 7. 替换原序列输出的使用
    if (callOp.getNumResults() != outputs.size()) {
      sequence.front()->emitError(
          "Mismatch between outlined function results and call op results");
      newFunc.erase();  // 清理创建的新函数
      return failure();
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      // 需要替换所有在序列外部对原输出的使用
      Value originalOutput = outputs[i];
      Value newResult = callOp.getResult(i);
      // 使用 replaceUsesWithIf 安全地替换
      originalOutput.replaceUsesWithIf(newResult, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        // 只替换序列外部的使用
        return std::find(sequence.begin(), sequence.end(), user) ==
               sequence.end();
      });
    }

    // 8. 删除原序列的操作 (从后往前删，避免迭代器失效问题)
    for (auto it = sequence.rbegin(); it != sequence.rend(); ++it) {
      Operation *opToDelete = *it;
      if (opToDelete->use_empty()) {  // 确保所有用途已被替换
        llvm::outs() << "  Erasing op: " << *opToDelete << "\n";
        opToDelete->erase();
      } else {
        // 如果还有用途，说明替换逻辑有问题
        opToDelete->emitWarning("Operation still has uses after outlining");
        // 可能需要打印剩余的用户进行调试
        // for(auto& use : opToDelete->getUses()) { llvm::errs() << "  Remaining
        // user: " << *use.getOwner() << "\n"; }
      }
    }

    llvm::outs() << "Successfully outlined sequence into: " << newFuncName
                 << "\n";
    return callOp;
  }
};

// OutlineMainFunctionPass 的实现
struct OutlineMainFunctionPass final
    : impl::OutlineMainFunctionPassBase<OutlineMainFunctionPass> {
  void runOnOperation() override {
    func::FuncOp mainFunc = getOperation();

    // 检查是否是 main 函数 (可以根据需要调整判断逻辑)
    if (mainFunc.getName() != "main") {
      return;
    }

    // 创建 Outliner 实例并执行
    MainFunctionOutliner outliner(mainFunc);
    if (failed(outliner.run())) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

}  // namespace stablehlo_ext
}  // namespace mlir