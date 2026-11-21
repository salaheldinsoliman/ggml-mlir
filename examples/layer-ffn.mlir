// Emitted by ggml-mlir from SmolLM2-135M-Instruct (f16 weights), 128 tokens, with GGML_MLIR_EXPORT.
// One function for a run of 12 llama.cpp operations in a transformer layer:
// attention output projection, residual add, RMS norm, feed-forward block (gate, up, SwiGLU, down),
// residual add, RMS norm, and the first projection of the next layer.
// Arguments are llama.cpp's own buffers: weights (f16), inputs, and one destination per operation.
func.func @kernel(%a0: memref<1x1x576x576xf16>, %a1: memref<1x1x128x576xf32>, %a2: memref<1x1x128x576xf32>, %a3: memref<1x1x128x576xf32>, %a4: memref<1x1x128x576xf32>, %a5: memref<1x1x128x576xf32>, %a6: memref<1x1x1x576xf32>, %a7: memref<1x1x128x576xf32>, %a8: memref<1x1x1536x576xf16>, %a9: memref<1x1x128x1536xf32>, %a10: memref<1x1x1536x576xf16>, %a11: memref<1x1x128x1536xf32>, %a12: memref<1x1x128x1536xf32>, %a13: memref<1x1x576x1536xf16>, %a14: memref<1x1x128x576xf32>, %a15: memref<1x1x128x576xf32>, %a16: memref<1x1x128x576xf32>, %a17: memref<1x1x1x576xf32>, %a18: memref<1x1x128x576xf32>, %a19: memref<1x1x576x576xf16>, %a20: memref<1x1x128x576xf32>) {
  // MUL_MAT
  %v0 = bufferization.to_tensor %a0 restrict : memref<1x1x576x576xf16> to tensor<1x1x576x576xf16>
  %v1 = bufferization.to_tensor %a1 restrict : memref<1x1x128x576xf32> to tensor<1x1x128x576xf32>
  %v3 = tensor.collapse_shape %v0 [[0, 1, 2], [3]] : tensor<1x1x576x576xf16> into tensor<576x576xf16>
  %v4 = tensor.collapse_shape %v1 [[0, 1, 2], [3]] : tensor<1x1x128x576xf32> into tensor<128x576xf32>
  %v6 = tensor.empty() : tensor<128x576xf32>
  %v7 = arith.constant 0.0 : f32
  %v8 = linalg.fill ins(%v7 : f32) outs(%v6 : tensor<128x576xf32>) -> tensor<128x576xf32>
  %v5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v4, %v3 : tensor<128x576xf32>, tensor<576x576xf16>) outs(%v8 : tensor<128x576xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x576xf32>
  %v2 = tensor.expand_shape %v5 [[0, 1, 2], [3]] output_shape [1, 1, 128, 576] : tensor<128x576xf32> into tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v2 in restrict writable %a2 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // ADD
  %v9 = bufferization.to_tensor %a3 restrict : memref<1x1x128x576xf32> to tensor<1x1x128x576xf32>
  %v10 = tensor.empty() : tensor<1x1x128x576xf32>
  %v11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v2, %v9 : tensor<1x1x128x576xf32>, tensor<1x1x128x576xf32>) outs(%v10 : tensor<1x1x128x576xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %y = arith.addf %in0, %in1 : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v11 in restrict writable %a4 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // RMS_NORM
  %v12 = tensor.empty() : tensor<1x1x128xf64>
  %v13 = arith.constant 0.0 : f64
  %v14 = linalg.fill ins(%v13 : f64) outs(%v12 : tensor<1x1x128xf64>) -> tensor<1x1x128xf64>
  %v15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%v11 : tensor<1x1x128x576xf32>) outs(%v14 : tensor<1x1x128xf64>) {
  ^bb0(%in: f32, %acc: f64):
    %sq = arith.mulf %in, %in : f32
    %sqd = arith.extf %sq : f32 to f64
    %s = arith.addf %acc, %sqd : f64
    linalg.yield %s : f64
  } -> tensor<1x1x128xf64>
  %v16 = tensor.empty() : tensor<1x1x128x576xf32>
  %v17 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v11, %v15 : tensor<1x1x128x576xf32>, tensor<1x1x128xf64>) outs(%v16 : tensor<1x1x128x576xf32>) {
  ^bb0(%in: f32, %sum: f64, %out: f32):
    %n = arith.constant 576.0 : f64
    %eps = arith.constant 0x3727C5AC : f32
    %one = arith.constant 1.0 : f32
    %meand = arith.divf %sum, %n : f64
    %mean = arith.truncf %meand : f64 to f32
    %me = arith.addf %mean, %eps : f32
    %rt = math.sqrt %me : f32
    %scale = arith.divf %one, %rt : f32
    %y = arith.mulf %in, %scale : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v17 in restrict writable %a5 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // MUL
  %v18 = bufferization.to_tensor %a6 restrict : memref<1x1x1x576xf32> to tensor<1x1x1x576xf32>
  %v19 = tensor.collapse_shape %v18 [[0], [1], [2, 3]] : tensor<1x1x1x576xf32> into tensor<1x1x576xf32>
  %v20 = tensor.empty() : tensor<1x1x128x576xf32>
  %v21 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v17, %v19 : tensor<1x1x128x576xf32>, tensor<1x1x576xf32>) outs(%v20 : tensor<1x1x128x576xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %y = arith.mulf %in0, %in1 : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v21 in restrict writable %a7 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // MUL_MAT
  %v22 = bufferization.to_tensor %a8 restrict : memref<1x1x1536x576xf16> to tensor<1x1x1536x576xf16>
  %v24 = tensor.collapse_shape %v22 [[0, 1, 2], [3]] : tensor<1x1x1536x576xf16> into tensor<1536x576xf16>
  %v25 = tensor.collapse_shape %v21 [[0, 1, 2], [3]] : tensor<1x1x128x576xf32> into tensor<128x576xf32>
  %v27 = tensor.empty() : tensor<128x1536xf32>
  %v28 = arith.constant 0.0 : f32
  %v29 = linalg.fill ins(%v28 : f32) outs(%v27 : tensor<128x1536xf32>) -> tensor<128x1536xf32>
  %v26 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v25, %v24 : tensor<128x576xf32>, tensor<1536x576xf16>) outs(%v29 : tensor<128x1536xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x1536xf32>
  %v23 = tensor.expand_shape %v26 [[0, 1, 2], [3]] output_shape [1, 1, 128, 1536] : tensor<128x1536xf32> into tensor<1x1x128x1536xf32>
  bufferization.materialize_in_destination %v23 in restrict writable %a9 : (tensor<1x1x128x1536xf32>, memref<1x1x128x1536xf32>) -> ()
  // MUL_MAT
  %v30 = bufferization.to_tensor %a10 restrict : memref<1x1x1536x576xf16> to tensor<1x1x1536x576xf16>
  %v32 = tensor.collapse_shape %v30 [[0, 1, 2], [3]] : tensor<1x1x1536x576xf16> into tensor<1536x576xf16>
  %v33 = tensor.collapse_shape %v21 [[0, 1, 2], [3]] : tensor<1x1x128x576xf32> into tensor<128x576xf32>
  %v35 = tensor.empty() : tensor<128x1536xf32>
  %v36 = arith.constant 0.0 : f32
  %v37 = linalg.fill ins(%v36 : f32) outs(%v35 : tensor<128x1536xf32>) -> tensor<128x1536xf32>
  %v34 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v33, %v32 : tensor<128x576xf32>, tensor<1536x576xf16>) outs(%v37 : tensor<128x1536xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x1536xf32>
  %v31 = tensor.expand_shape %v34 [[0, 1, 2], [3]] output_shape [1, 1, 128, 1536] : tensor<128x1536xf32> into tensor<1x1x128x1536xf32>
  bufferization.materialize_in_destination %v31 in restrict writable %a11 : (tensor<1x1x128x1536xf32>, memref<1x1x128x1536xf32>) -> ()
  // SWIGLU
  %v38 = tensor.empty() : tensor<1x1x128x1536xf32>
  %v39 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v23, %v31 : tensor<1x1x128x1536xf32>, tensor<1x1x128x1536xf32>) outs(%v38 : tensor<1x1x128x1536xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %one = arith.constant 1.0 : f32
    %neg = arith.negf %in0 : f32
    %exp = math.exp %neg : f32
    %den = arith.addf %one, %exp : f32
    %act = arith.divf %in0, %den : f32
    %y = arith.mulf %act, %in1 : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x1536xf32>
  bufferization.materialize_in_destination %v39 in restrict writable %a12 : (tensor<1x1x128x1536xf32>, memref<1x1x128x1536xf32>) -> ()
  // MUL_MAT
  %v40 = bufferization.to_tensor %a13 restrict : memref<1x1x576x1536xf16> to tensor<1x1x576x1536xf16>
  %v42 = tensor.collapse_shape %v40 [[0, 1, 2], [3]] : tensor<1x1x576x1536xf16> into tensor<576x1536xf16>
  %v43 = tensor.collapse_shape %v39 [[0, 1, 2], [3]] : tensor<1x1x128x1536xf32> into tensor<128x1536xf32>
  %v45 = tensor.empty() : tensor<128x576xf32>
  %v46 = arith.constant 0.0 : f32
  %v47 = linalg.fill ins(%v46 : f32) outs(%v45 : tensor<128x576xf32>) -> tensor<128x576xf32>
  %v44 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v43, %v42 : tensor<128x1536xf32>, tensor<576x1536xf16>) outs(%v47 : tensor<128x576xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x576xf32>
  %v41 = tensor.expand_shape %v44 [[0, 1, 2], [3]] output_shape [1, 1, 128, 576] : tensor<128x576xf32> into tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v41 in restrict writable %a14 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // ADD
  %v48 = tensor.empty() : tensor<1x1x128x576xf32>
  %v49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v41, %v11 : tensor<1x1x128x576xf32>, tensor<1x1x128x576xf32>) outs(%v48 : tensor<1x1x128x576xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %y = arith.addf %in0, %in1 : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v49 in restrict writable %a15 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // RMS_NORM
  %v50 = tensor.empty() : tensor<1x1x128xf64>
  %v51 = arith.constant 0.0 : f64
  %v52 = linalg.fill ins(%v51 : f64) outs(%v50 : tensor<1x1x128xf64>) -> tensor<1x1x128xf64>
  %v53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%v49 : tensor<1x1x128x576xf32>) outs(%v52 : tensor<1x1x128xf64>) {
  ^bb0(%in: f32, %acc: f64):
    %sq = arith.mulf %in, %in : f32
    %sqd = arith.extf %sq : f32 to f64
    %s = arith.addf %acc, %sqd : f64
    linalg.yield %s : f64
  } -> tensor<1x1x128xf64>
  %v54 = tensor.empty() : tensor<1x1x128x576xf32>
  %v55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v49, %v53 : tensor<1x1x128x576xf32>, tensor<1x1x128xf64>) outs(%v54 : tensor<1x1x128x576xf32>) {
  ^bb0(%in: f32, %sum: f64, %out: f32):
    %n = arith.constant 576.0 : f64
    %eps = arith.constant 0x3727C5AC : f32
    %one = arith.constant 1.0 : f32
    %meand = arith.divf %sum, %n : f64
    %mean = arith.truncf %meand : f64 to f32
    %me = arith.addf %mean, %eps : f32
    %rt = math.sqrt %me : f32
    %scale = arith.divf %one, %rt : f32
    %y = arith.mulf %in, %scale : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v55 in restrict writable %a16 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // MUL
  %v56 = bufferization.to_tensor %a17 restrict : memref<1x1x1x576xf32> to tensor<1x1x1x576xf32>
  %v57 = tensor.collapse_shape %v56 [[0], [1], [2, 3]] : tensor<1x1x1x576xf32> into tensor<1x1x576xf32>
  %v58 = tensor.empty() : tensor<1x1x128x576xf32>
  %v59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v55, %v57 : tensor<1x1x128x576xf32>, tensor<1x1x576xf32>) outs(%v58 : tensor<1x1x128x576xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %y = arith.mulf %in0, %in1 : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v59 in restrict writable %a18 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // MUL_MAT
  %v60 = bufferization.to_tensor %a19 restrict : memref<1x1x576x576xf16> to tensor<1x1x576x576xf16>
  %v62 = tensor.collapse_shape %v60 [[0, 1, 2], [3]] : tensor<1x1x576x576xf16> into tensor<576x576xf16>
  %v63 = tensor.collapse_shape %v59 [[0, 1, 2], [3]] : tensor<1x1x128x576xf32> into tensor<128x576xf32>
  %v65 = tensor.empty() : tensor<128x576xf32>
  %v66 = arith.constant 0.0 : f32
  %v67 = linalg.fill ins(%v66 : f32) outs(%v65 : tensor<128x576xf32>) -> tensor<128x576xf32>
  %v64 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v63, %v62 : tensor<128x576xf32>, tensor<576x576xf16>) outs(%v67 : tensor<128x576xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x576xf32>
  %v61 = tensor.expand_shape %v64 [[0, 1, 2], [3]] output_shape [1, 1, 128, 576] : tensor<128x576xf32> into tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v61 in restrict writable %a20 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  %v68 = tensor.collapse_shape %v61 [[0, 1, 2, 3]] : tensor<1x1x128x576xf32> into tensor<73728xf32>
  %v69 = tensor.expand_shape %v68 [[0, 1, 2, 3]] output_shape [1, 128, 9, 64] : tensor<73728xf32> into tensor<1x128x9x64xf32>
  return
}
