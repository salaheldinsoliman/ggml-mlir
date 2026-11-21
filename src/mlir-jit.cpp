// MLIR text -> native code on the host CPU via upstream MLIR -> LLVM, cached in memory by node key.

#include "ggml-mlir-impl.h"
#include "ggml-impl.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>

// naive CPU lowering: correctness first, performance later (tiling + vectorization)
// note: no "builtin.module(...)" wrapper - the PassManager is already anchored on the module
static const char * k_cpu_pipeline =
    "eliminate-empty-tensors,"
    "one-shot-bufferize,"
    "buffer-deallocation-pipeline,"
    "convert-linalg-to-loops,"
    "expand-strided-metadata,"
    "lower-affine,"
    "convert-scf-to-cf,"
    "finalize-memref-to-llvm,"
    "convert-arith-to-llvm,"
    "convert-func-to-llvm,"
    "convert-cf-to-llvm,"
    "reconcile-unrealized-casts";

// C layout of a rank-4 memref descriptor (see mlir/ExecutionEngine/CRunnerUtils.h)
struct memref4 {
    void *  allocated;
    void *  aligned;
    int64_t offset;
    int64_t sizes[4];
    int64_t strides[4];
};

typedef void (*kernel3_t)(memref4 *, memref4 *, memref4 *);

struct kernel_entry {
    std::unique_ptr<mlir::ExecutionEngine> engine;
    void * fn = nullptr;
};

struct ggml_mlir_jit {
    mlir::MLIRContext ctx;
    std::unordered_map<std::string, kernel_entry> cache;
    bool dump         = false; // GGML_MLIR_DUMP=1: print emitted MLIR
    bool dump_lowered = false; // GGML_MLIR_DUMP=2: also print the lowered (LLVM dialect) module
    bool stats        = false; // GGML_MLIR_STATS=1: print kernel and call counts on exit
    size_t n_calls    = 0;

    ggml_mlir_jit() : ctx(make_registry(), mlir::MLIRContext::Threading::DISABLED) {
        ctx.loadAllAvailableDialects();
        const char * env = std::getenv("GGML_MLIR_DUMP");
        dump         = env != nullptr;
        dump_lowered = env != nullptr && std::atoi(env) >= 2;
        stats        = std::getenv("GGML_MLIR_STATS") != nullptr;
    }

    static mlir::DialectRegistry make_registry() {
        mlir::DialectRegistry registry;
        mlir::registerAllDialects(registry);
        mlir::registerAllExtensions(registry);
        mlir::registerAllToLLVMIRTranslations(registry);
        return registry;
    }
};

static void jit_global_init() {
    static std::once_flag once;
    std::call_once(once, []() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        mlir::registerAllPasses();
    });
}

ggml_mlir_jit * ggml_mlir_jit_new(void) {
    jit_global_init();
    return new ggml_mlir_jit;
}

void ggml_mlir_jit_free(ggml_mlir_jit * jit) {
    if (jit->stats) {
        fprintf(stderr, "ggml-mlir: %zu kernels compiled, %zu kernel calls\n", jit->cache.size(), jit->n_calls);
    }
    delete jit;
}

static kernel_entry compile(ggml_mlir_jit * jit, const struct ggml_tensor * node) {
    const std::string src = ggml_mlir_emit(node);
    if (jit->dump) {
        fprintf(stderr, "// ggml-mlir: %s\n%s\n", ggml_mlir_node_key(node).c_str(), src.c_str());
    }

    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(src, &jit->ctx);
    if (!module) {
        GGML_ABORT("ggml-mlir: failed to parse emitted MLIR for %s", ggml_mlir_node_key(node).c_str());
    }

    mlir::PassManager pm(&jit->ctx);
    std::string err;
    llvm::raw_string_ostream err_os(err);
    if (mlir::failed(mlir::parsePassPipeline(k_cpu_pipeline, pm, err_os))) {
        GGML_ABORT("ggml-mlir: bad pass pipeline: %s", err.c_str());
    }
    if (mlir::failed(pm.run(*module))) {
        GGML_ABORT("ggml-mlir: lowering failed for %s", ggml_mlir_node_key(node).c_str());
    }
    if (jit->dump_lowered) {
        module->print(llvm::errs());
        llvm::errs() << "\n";
    }

    mlir::ExecutionEngineOptions opts;
    opts.transformer = mlir::makeOptimizingTransformer(/*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
    auto engine = mlir::ExecutionEngine::create(*module, opts);
    if (!engine) {
        GGML_ABORT("ggml-mlir: JIT failed: %s", llvm::toString(engine.takeError()).c_str());
    }

    auto fn = (*engine)->lookup("_mlir_ciface_kernel");
    if (!fn) {
        GGML_ABORT("ggml-mlir: kernel symbol not found: %s", llvm::toString(fn.takeError()).c_str());
    }

    kernel_entry e;
    e.engine = std::move(*engine);
    e.fn     = *fn;
    return e;
}

// describe a ggml tensor as a rank-4 memref: sizes/strides reversed, strides in elements
static memref4 make_memref4(const struct ggml_tensor * t) {
    const size_t es = ggml_type_size(t->type);
    memref4 d;
    d.allocated = t->data;
    d.aligned   = t->data;
    d.offset    = 0;
    for (int i = 0; i < 4; i++) {
        d.sizes[i]   = t->ne[3 - i];
        d.strides[i] = (int64_t) (t->nb[3 - i] / es);
    }
    return d;
}

void ggml_mlir_jit_compute(ggml_mlir_jit * jit, struct ggml_tensor * node) {
    const std::string key = ggml_mlir_node_key(node);

    auto it = jit->cache.find(key);
    if (it == jit->cache.end()) {
        it = jit->cache.emplace(key, compile(jit, node)).first;
    }

    jit->n_calls++;

    memref4 a = make_memref4(node->src[0]);
    memref4 b = make_memref4(node->src[1]);
    memref4 d = make_memref4(node);
    ((kernel3_t) it->second.fn)(&a, &b, &d);
}
