cmake_minimum_required(VERSION 3.20)
project(ggml-mlir C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp must be built with BUILD_SHARED_LIBS=ON and GGML_BACKEND_DL=ON
set(LLAMA_CPP_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp" CACHE PATH "llama.cpp source tree")
set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_DIR}/build"                            CACHE PATH "llama.cpp build tree")

find_library(GGML_BASE_LIB ggml-base PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)

# the plugin: loaded at runtime via GGML_BACKEND_PATH or ggml_backend_load()
add_library(ggml-mlir MODULE src/ggml-mlir.cpp)

target_include_directories(ggml-mlir PRIVATE
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/ggml/src)          # ggml-backend-impl.h, ggml-impl.h (internal, pinned to our llama.cpp commit)

# same definitions ggml_add_backend_library() gives in-tree dynamic backends
target_compile_definitions(ggml-mlir PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED GGML_SHARED)

target_link_libraries(ggml-mlir PRIVATE ${GGML_BASE_LIB})
