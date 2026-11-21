// ggml-mlir: a ggml backend that compiles ggml ops through MLIR.
// Structure follows ggml/src/ggml-blas/ggml-blas.cpp: tensors live in host (CPU) buffers,
// the backend only claims the ops it can compile.

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstring>

struct ggml_backend_mlir_context {
    int dummy = 0; // placeholder: will hold the compile cache
};

// backend interface

static const char * ggml_backend_mlir_get_name(ggml_backend_t backend) {
    return "MLIR";

    GGML_UNUSED(backend);
}

static void ggml_backend_mlir_free(ggml_backend_t backend) {
    delete (ggml_backend_mlir_context *) backend->context;
    delete backend;
}

static enum ggml_status ggml_backend_mlir_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i mlir_backend_i = {
    /* .get_name                = */ ggml_backend_mlir_get_name,
    /* .free                    = */ ggml_backend_mlir_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_mlir_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_mlir_guid(void) {
    static ggml_guid guid = { 0x6d, 0x6c, 0x69, 0x72, 0x2d, 0x67, 0x67, 0x6d, 0x6c, 0x3a, 0x9e, 0x41, 0xb2, 0x07, 0x5c, 0xe1 };
    return &guid;
}

static ggml_backend_reg_t ggml_backend_mlir_reg(void);

static ggml_backend_t ggml_backend_mlir_init(void) {
    return new ggml_backend {
        /* .guid    = */ ggml_backend_mlir_guid(),
        /* .iface   = */ mlir_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_mlir_reg(), 0),
        /* .context = */ new ggml_backend_mlir_context,
    };
}

// device interface

static const char * ggml_backend_mlir_device_get_name(ggml_backend_dev_t dev) {
    return "MLIR";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_mlir_device_get_description(ggml_backend_dev_t dev) {
    return "MLIR (CPU via LLVM)";

    GGML_UNUSED(dev);
}

static void ggml_backend_mlir_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // uses host memory, nothing to report
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_mlir_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_mlir_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_mlir_device_get_name(dev);
    props->description = ggml_backend_mlir_device_get_description(dev);
    props->type        = ggml_backend_mlir_device_get_type(dev);
    ggml_backend_mlir_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_mlir_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_mlir_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_mlir_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_mlir_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_mlir_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_mlir_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_mlir_device_i = {
    /* .get_name             = */ ggml_backend_mlir_device_get_name,
    /* .get_description      = */ ggml_backend_mlir_device_get_description,
    /* .get_memory           = */ ggml_backend_mlir_device_get_memory,
    /* .get_type             = */ ggml_backend_mlir_device_get_type,
    /* .get_props            = */ ggml_backend_mlir_device_get_props,
    /* .init_backend         = */ ggml_backend_mlir_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_mlir_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_mlir_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_mlir_device_supports_op,
    /* .supports_buft        = */ ggml_backend_mlir_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_mlir_reg_get_name(ggml_backend_reg_t reg) {
    return "MLIR";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_mlir_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_mlir_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_mlir_device = {
        /* .iface   = */ ggml_backend_mlir_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_mlir_device;
}

static const struct ggml_backend_reg_i ggml_backend_mlir_reg_i = {
    /* .get_name         = */ ggml_backend_mlir_reg_get_name,
    /* .get_device_count = */ ggml_backend_mlir_reg_get_device_count,
    /* .get_device       = */ ggml_backend_mlir_reg_get_device,
    /* .get_proc_address = */ NULL,
};

static ggml_backend_reg_t ggml_backend_mlir_reg(void) {
    static struct ggml_backend_reg ggml_backend_mlir_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_mlir_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_mlir_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_mlir_reg)
