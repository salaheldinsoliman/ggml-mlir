// Emitted by ggml-mlir from SmolLM2-135M-Instruct (f16 weights), 128 tokens, with GGML_MLIR_EXPORT
// (llama.cpp run with -fa off, so attention is built from separate operations).
// One function for a run of 20 llama.cpp operations, one whole transformer layer:
// attention scores, masked softmax, attention output and its projection, residual add, RMS norm,
// feed-forward block (gate, up, SwiGLU, down), residual add, RMS norm, and the next layer's
// query/key/value projections with RoPE.
// Arguments are llama.cpp's own buffers: weights (f16), inputs such as the KV cache, the mask and
// the token positions, and one destination per operation.
func.func @kernel(%a0: memref<1x3x256x64xf16, strided<[49152, 64, 192, 1], offset: 0>>, %a1: memref<1x9x128x64xf32, strided<[73728, 64, 576, 1], offset: 0>>, %a2: memref<1x9x128x256xf32>, %a3: memref<1x1x128x256xf32>, %a4: memref<1x9x128x256xf32>, %a5: memref<1x3x64x256xf16>, %a6: memref<1x9x128x64xf32>, %a7: memref<1x1x128x576xf32>, %a8: memref<1x1x576x576xf16>, %a9: memref<1x1x128x576xf32>, %a10: memref<1x1x128x576xf32>, %a11: memref<1x1x128x576xf32>, %a12: memref<1x1x128x576xf32>, %a13: memref<1x1x1x576xf32>, %a14: memref<1x1x128x576xf32>, %a15: memref<1x1x1536x576xf16>, %a16: memref<1x1x128x1536xf32>, %a17: memref<1x1x1536x576xf16>, %a18: memref<1x1x128x1536xf32>, %a19: memref<1x1x128x1536xf32>, %a20: memref<1x1x576x1536xf16>, %a21: memref<1x1x128x576xf32>, %a22: memref<1x1x128x576xf32>, %a23: memref<1x1x128x576xf32>, %a24: memref<1x1x1x576xf32>, %a25: memref<1x1x128x576xf32>, %a26: memref<1x1x576x576xf16>, %a27: memref<1x1x128x576xf32>, %a28: memref<1x1x1x128xi32>, %a29: memref<1x128x9x64xf32>, %a30: memref<1x1x192x576xf16>, %a31: memref<1x1x128x192xf32>, %a32: memref<1x1x192x576xf16>, %a33: memref<1x1x128x192xf32>, %a34: memref<1x128x3x64xf32>) {
  // MUL_MAT
  %v0 = bufferization.to_tensor %a0 restrict : memref<1x3x256x64xf16, strided<[49152, 64, 192, 1], offset: 0>> to tensor<1x3x256x64xf16>
  %v1 = bufferization.to_tensor %a1 restrict : memref<1x9x128x64xf32, strided<[73728, 64, 576, 1], offset: 0>> to tensor<1x9x128x64xf32>
  %v3 = tensor.expand_shape %v1 [[0, 1], [2, 3], [4], [5]] output_shape [1, 1, 3, 3, 128, 64] : tensor<1x9x128x64xf32> into tensor<1x1x3x3x128x64xf32>
  %v5 = tensor.empty() : tensor<1x1x3x3x128x256xf32>
  %v6 = arith.constant 0.0 : f32
  %v7 = linalg.fill ins(%v6 : f32) outs(%v5 : tensor<1x1x3x3x128x256xf32>) -> tensor<1x1x3x3x128x256xf32>
  %v4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d6)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d5, d6)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%v3, %v0 : tensor<1x1x3x3x128x64xf32>, tensor<1x3x256x64xf16>) outs(%v7 : tensor<1x1x3x3x128x256xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<1x1x3x3x128x256xf32>
  %v2 = tensor.collapse_shape %v4 [[0, 1], [2, 3], [4], [5]] : tensor<1x1x3x3x128x256xf32> into tensor<1x9x128x256xf32>
  bufferization.materialize_in_destination %v2 in restrict writable %a2 : (tensor<1x9x128x256xf32>, memref<1x9x128x256xf32>) -> ()
  // SOFT_MAX
  %v8 = bufferization.to_tensor %a3 restrict : memref<1x1x128x256xf32> to tensor<1x1x128x256xf32>
  %v9 = tensor.collapse_shape %v8 [[0], [1, 2], [3]] : tensor<1x1x128x256xf32> into tensor<1x128x256xf32>
  %v10 = tensor.empty() : tensor<1x9x128x256xf32>
  %v11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v2, %v9 : tensor<1x9x128x256xf32>, tensor<1x128x256xf32>) outs(%v10 : tensor<1x9x128x256xf32>) {
  ^bb0(%in: f32, %m: f32, %out: f32):
    %scale = arith.constant 0x3E000000 : f32
    %sc = arith.mulf %in, %scale : f32
    %y = arith.addf %sc, %m : f32
    linalg.yield %y : f32
  } -> tensor<1x9x128x256xf32>
  %v12 = tensor.empty() : tensor<1x9x128xf32>
  %v13 = arith.constant 0xFF800000 : f32
  %v14 = linalg.fill ins(%v13 : f32) outs(%v12 : tensor<1x9x128xf32>) -> tensor<1x9x128xf32>
  %v15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%v11 : tensor<1x9x128x256xf32>) outs(%v14 : tensor<1x9x128xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %gt = arith.cmpf ogt, %in, %acc : f32
    %m = arith.select %gt, %in, %acc : f32
    linalg.yield %m : f32
  } -> tensor<1x9x128xf32>
  %v16 = tensor.empty() : tensor<1x9x128xf64>
  %v17 = arith.constant 0.0 : f64
  %v18 = linalg.fill ins(%v17 : f64) outs(%v16 : tensor<1x9x128xf64>) -> tensor<1x9x128xf64>
  %v19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%v11, %v15 : tensor<1x9x128x256xf32>, tensor<1x9x128xf32>) outs(%v18 : tensor<1x9x128xf64>) {
  ^bb0(%in: f32, %max: f32, %acc: f64):
    %d = arith.subf %in, %max : f32
    %e = math.exp %d : f32
    %ed = arith.extf %e : f32 to f64
    %s = arith.addf %acc, %ed : f64
    linalg.yield %s : f64
  } -> tensor<1x9x128xf64>
  %v20 = tensor.empty() : tensor<1x9x128x256xf32>
  %v21 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v11, %v15, %v19 : tensor<1x9x128x256xf32>, tensor<1x9x128xf32>, tensor<1x9x128xf64>) outs(%v20 : tensor<1x9x128x256xf32>) {
  ^bb0(%in: f32, %max: f32, %sum: f64, %out: f32):
    %d = arith.subf %in, %max : f32
    %e = math.exp %d : f32
    %oned = arith.constant 1.0 : f64
    %invd = arith.divf %oned, %sum : f64
    %inv = arith.truncf %invd : f64 to f32
    %y = arith.mulf %e, %inv : f32
    linalg.yield %y : f32
  } -> tensor<1x9x128x256xf32>
  bufferization.materialize_in_destination %v21 in restrict writable %a4 : (tensor<1x9x128x256xf32>, memref<1x9x128x256xf32>) -> ()
  // MUL_MAT
  %v22 = bufferization.to_tensor %a5 restrict : memref<1x3x64x256xf16> to tensor<1x3x64x256xf16>
  %v24 = tensor.expand_shape %v21 [[0, 1], [2, 3], [4], [5]] output_shape [1, 1, 3, 3, 128, 256] : tensor<1x9x128x256xf32> into tensor<1x1x3x3x128x256xf32>
  %v26 = tensor.empty() : tensor<1x1x3x3x128x64xf32>
  %v27 = arith.constant 0.0 : f32
  %v28 = linalg.fill ins(%v27 : f32) outs(%v26 : tensor<1x1x3x3x128x64xf32>) -> tensor<1x1x3x3x128x64xf32>
  %v25 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d6)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d5, d6)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%v24, %v22 : tensor<1x1x3x3x128x256xf32>, tensor<1x3x64x256xf16>) outs(%v28 : tensor<1x1x3x3x128x64xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<1x1x3x3x128x64xf32>
  %v23 = tensor.collapse_shape %v25 [[0, 1], [2, 3], [4], [5]] : tensor<1x1x3x3x128x64xf32> into tensor<1x9x128x64xf32>
  bufferization.materialize_in_destination %v23 in restrict writable %a6 : (tensor<1x9x128x64xf32>, memref<1x9x128x64xf32>) -> ()
  %v29 = tensor.empty() : tensor<1x128x9x64xf32>
  %v30 = linalg.transpose ins(%v23 : tensor<1x9x128x64xf32>) outs(%v29 : tensor<1x128x9x64xf32>) permutation = [0, 2, 1, 3]
  // CONT
  %v31 = tensor.collapse_shape %v30 [[0, 1, 2, 3]] : tensor<1x128x9x64xf32> into tensor<73728xf32>
  %v32 = tensor.expand_shape %v31 [[0, 1, 2, 3]] output_shape [1, 1, 128, 576] : tensor<73728xf32> into tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v32 in restrict writable %a7 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // MUL_MAT
  %v33 = bufferization.to_tensor %a8 restrict : memref<1x1x576x576xf16> to tensor<1x1x576x576xf16>
  %v35 = tensor.collapse_shape %v33 [[0, 1, 2], [3]] : tensor<1x1x576x576xf16> into tensor<576x576xf16>
  %v36 = tensor.collapse_shape %v32 [[0, 1, 2], [3]] : tensor<1x1x128x576xf32> into tensor<128x576xf32>
  %v38 = tensor.empty() : tensor<128x576xf32>
  %v39 = arith.constant 0.0 : f32
  %v40 = linalg.fill ins(%v39 : f32) outs(%v38 : tensor<128x576xf32>) -> tensor<128x576xf32>
  %v37 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v36, %v35 : tensor<128x576xf32>, tensor<576x576xf16>) outs(%v40 : tensor<128x576xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x576xf32>
  %v34 = tensor.expand_shape %v37 [[0, 1, 2], [3]] output_shape [1, 1, 128, 576] : tensor<128x576xf32> into tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v34 in restrict writable %a9 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // ADD
  %v41 = bufferization.to_tensor %a10 restrict : memref<1x1x128x576xf32> to tensor<1x1x128x576xf32>
  %v42 = tensor.empty() : tensor<1x1x128x576xf32>
  %v43 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v34, %v41 : tensor<1x1x128x576xf32>, tensor<1x1x128x576xf32>) outs(%v42 : tensor<1x1x128x576xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %y = arith.addf %in0, %in1 : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v43 in restrict writable %a11 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // RMS_NORM
  %v44 = tensor.empty() : tensor<1x1x128xf64>
  %v45 = arith.constant 0.0 : f64
  %v46 = linalg.fill ins(%v45 : f64) outs(%v44 : tensor<1x1x128xf64>) -> tensor<1x1x128xf64>
  %v47 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%v43 : tensor<1x1x128x576xf32>) outs(%v46 : tensor<1x1x128xf64>) {
  ^bb0(%in: f32, %acc: f64):
    %sq = arith.mulf %in, %in : f32
    %sqd = arith.extf %sq : f32 to f64
    %s = arith.addf %acc, %sqd : f64
    linalg.yield %s : f64
  } -> tensor<1x1x128xf64>
  %v48 = tensor.empty() : tensor<1x1x128x576xf32>
  %v49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v43, %v47 : tensor<1x1x128x576xf32>, tensor<1x1x128xf64>) outs(%v48 : tensor<1x1x128x576xf32>) {
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
  bufferization.materialize_in_destination %v49 in restrict writable %a12 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // MUL
  %v50 = bufferization.to_tensor %a13 restrict : memref<1x1x1x576xf32> to tensor<1x1x1x576xf32>
  %v51 = tensor.collapse_shape %v50 [[0], [1], [2, 3]] : tensor<1x1x1x576xf32> into tensor<1x1x576xf32>
  %v52 = tensor.empty() : tensor<1x1x128x576xf32>
  %v53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v49, %v51 : tensor<1x1x128x576xf32>, tensor<1x1x576xf32>) outs(%v52 : tensor<1x1x128x576xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %y = arith.mulf %in0, %in1 : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v53 in restrict writable %a14 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // MUL_MAT
  %v54 = bufferization.to_tensor %a15 restrict : memref<1x1x1536x576xf16> to tensor<1x1x1536x576xf16>
  %v56 = tensor.collapse_shape %v54 [[0, 1, 2], [3]] : tensor<1x1x1536x576xf16> into tensor<1536x576xf16>
  %v57 = tensor.collapse_shape %v53 [[0, 1, 2], [3]] : tensor<1x1x128x576xf32> into tensor<128x576xf32>
  %v59 = tensor.empty() : tensor<128x1536xf32>
  %v60 = arith.constant 0.0 : f32
  %v61 = linalg.fill ins(%v60 : f32) outs(%v59 : tensor<128x1536xf32>) -> tensor<128x1536xf32>
  %v58 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v57, %v56 : tensor<128x576xf32>, tensor<1536x576xf16>) outs(%v61 : tensor<128x1536xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x1536xf32>
  %v55 = tensor.expand_shape %v58 [[0, 1, 2], [3]] output_shape [1, 1, 128, 1536] : tensor<128x1536xf32> into tensor<1x1x128x1536xf32>
  bufferization.materialize_in_destination %v55 in restrict writable %a16 : (tensor<1x1x128x1536xf32>, memref<1x1x128x1536xf32>) -> ()
  // MUL_MAT
  %v62 = bufferization.to_tensor %a17 restrict : memref<1x1x1536x576xf16> to tensor<1x1x1536x576xf16>
  %v64 = tensor.collapse_shape %v62 [[0, 1, 2], [3]] : tensor<1x1x1536x576xf16> into tensor<1536x576xf16>
  %v65 = tensor.collapse_shape %v53 [[0, 1, 2], [3]] : tensor<1x1x128x576xf32> into tensor<128x576xf32>
  %v67 = tensor.empty() : tensor<128x1536xf32>
  %v68 = arith.constant 0.0 : f32
  %v69 = linalg.fill ins(%v68 : f32) outs(%v67 : tensor<128x1536xf32>) -> tensor<128x1536xf32>
  %v66 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v65, %v64 : tensor<128x576xf32>, tensor<1536x576xf16>) outs(%v69 : tensor<128x1536xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x1536xf32>
  %v63 = tensor.expand_shape %v66 [[0, 1, 2], [3]] output_shape [1, 1, 128, 1536] : tensor<128x1536xf32> into tensor<1x1x128x1536xf32>
  bufferization.materialize_in_destination %v63 in restrict writable %a18 : (tensor<1x1x128x1536xf32>, memref<1x1x128x1536xf32>) -> ()
  // SWIGLU
  %v70 = tensor.empty() : tensor<1x1x128x1536xf32>
  %v71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v55, %v63 : tensor<1x1x128x1536xf32>, tensor<1x1x128x1536xf32>) outs(%v70 : tensor<1x1x128x1536xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %one = arith.constant 1.0 : f32
    %neg = arith.negf %in0 : f32
    %exp = math.exp %neg : f32
    %den = arith.addf %one, %exp : f32
    %act = arith.divf %in0, %den : f32
    %y = arith.mulf %act, %in1 : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x1536xf32>
  bufferization.materialize_in_destination %v71 in restrict writable %a19 : (tensor<1x1x128x1536xf32>, memref<1x1x128x1536xf32>) -> ()
  // MUL_MAT
  %v72 = bufferization.to_tensor %a20 restrict : memref<1x1x576x1536xf16> to tensor<1x1x576x1536xf16>
  %v74 = tensor.collapse_shape %v72 [[0, 1, 2], [3]] : tensor<1x1x576x1536xf16> into tensor<576x1536xf16>
  %v75 = tensor.collapse_shape %v71 [[0, 1, 2], [3]] : tensor<1x1x128x1536xf32> into tensor<128x1536xf32>
  %v77 = tensor.empty() : tensor<128x576xf32>
  %v78 = arith.constant 0.0 : f32
  %v79 = linalg.fill ins(%v78 : f32) outs(%v77 : tensor<128x576xf32>) -> tensor<128x576xf32>
  %v76 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v75, %v74 : tensor<128x1536xf32>, tensor<576x1536xf16>) outs(%v79 : tensor<128x576xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x576xf32>
  %v73 = tensor.expand_shape %v76 [[0, 1, 2], [3]] output_shape [1, 1, 128, 576] : tensor<128x576xf32> into tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v73 in restrict writable %a21 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // ADD
  %v80 = tensor.empty() : tensor<1x1x128x576xf32>
  %v81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v73, %v43 : tensor<1x1x128x576xf32>, tensor<1x1x128x576xf32>) outs(%v80 : tensor<1x1x128x576xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %y = arith.addf %in0, %in1 : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v81 in restrict writable %a22 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // RMS_NORM
  %v82 = tensor.empty() : tensor<1x1x128xf64>
  %v83 = arith.constant 0.0 : f64
  %v84 = linalg.fill ins(%v83 : f64) outs(%v82 : tensor<1x1x128xf64>) -> tensor<1x1x128xf64>
  %v85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%v81 : tensor<1x1x128x576xf32>) outs(%v84 : tensor<1x1x128xf64>) {
  ^bb0(%in: f32, %acc: f64):
    %sq = arith.mulf %in, %in : f32
    %sqd = arith.extf %sq : f32 to f64
    %s = arith.addf %acc, %sqd : f64
    linalg.yield %s : f64
  } -> tensor<1x1x128xf64>
  %v86 = tensor.empty() : tensor<1x1x128x576xf32>
  %v87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v81, %v85 : tensor<1x1x128x576xf32>, tensor<1x1x128xf64>) outs(%v86 : tensor<1x1x128x576xf32>) {
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
  bufferization.materialize_in_destination %v87 in restrict writable %a23 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // MUL
  %v88 = bufferization.to_tensor %a24 restrict : memref<1x1x1x576xf32> to tensor<1x1x1x576xf32>
  %v89 = tensor.collapse_shape %v88 [[0], [1], [2, 3]] : tensor<1x1x1x576xf32> into tensor<1x1x576xf32>
  %v90 = tensor.empty() : tensor<1x1x128x576xf32>
  %v91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v87, %v89 : tensor<1x1x128x576xf32>, tensor<1x1x576xf32>) outs(%v90 : tensor<1x1x128x576xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %y = arith.mulf %in0, %in1 : f32
    linalg.yield %y : f32
  } -> tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v91 in restrict writable %a25 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  // MUL_MAT
  %v92 = bufferization.to_tensor %a26 restrict : memref<1x1x576x576xf16> to tensor<1x1x576x576xf16>
  %v94 = tensor.collapse_shape %v92 [[0, 1, 2], [3]] : tensor<1x1x576x576xf16> into tensor<576x576xf16>
  %v95 = tensor.collapse_shape %v91 [[0, 1, 2], [3]] : tensor<1x1x128x576xf32> into tensor<128x576xf32>
  %v97 = tensor.empty() : tensor<128x576xf32>
  %v98 = arith.constant 0.0 : f32
  %v99 = linalg.fill ins(%v98 : f32) outs(%v97 : tensor<128x576xf32>) -> tensor<128x576xf32>
  %v96 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v95, %v94 : tensor<128x576xf32>, tensor<576x576xf16>) outs(%v99 : tensor<128x576xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x576xf32>
  %v93 = tensor.expand_shape %v96 [[0, 1, 2], [3]] output_shape [1, 1, 128, 576] : tensor<128x576xf32> into tensor<1x1x128x576xf32>
  bufferization.materialize_in_destination %v93 in restrict writable %a27 : (tensor<1x1x128x576xf32>, memref<1x1x128x576xf32>) -> ()
  %v100 = tensor.collapse_shape %v93 [[0, 1, 2, 3]] : tensor<1x1x128x576xf32> into tensor<73728xf32>
  %v101 = tensor.expand_shape %v100 [[0, 1, 2, 3]] output_shape [1, 128, 9, 64] : tensor<73728xf32> into tensor<1x128x9x64xf32>
  // ROPE
  %v102 = arith.constant dense<[0x3F800000, 0x3F32A506, 0x3EF953CD, 0x3EADFCFD, 0x3E72D420, 0x3E297407, 0x3DEC7FD1, 0x3DA50953, 0x3D6655BD, 0x3D20BC18, 0x3CE054CB, 0x3C9C8B91, 0x3C5A7BE9, 0x3C1876FF, 0x3BD4CA0B, 0x3B947DA7, 0x3B4F3E2D, 0x3B109ED3, 0x3AC9D751, 0x3A8CD9D3, 0x3A449481, 0x3A092DFA, 0x39BF74CB, 0x39859AA0, 0x393A7746, 0x39021F22, 0x38B59B0E, 0x387D75C2, 0x3830DF44, 0x37F6DA83, 0x37AC4310, 0x37706B59]> : tensor<32xf32>
  %v103 = bufferization.to_tensor %a28 restrict : memref<1x1x1x128xi32> to tensor<1x1x1x128xi32>
  %v104 = tensor.collapse_shape %v103 [[0, 1, 2, 3]] : tensor<1x1x1x128xi32> into tensor<128xi32>
  %v105 = tensor.empty() : tensor<128x32xf32>
  %v106 = tensor.empty() : tensor<128x32xf32>
  %v107, %v107s = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%v104, %v102 : tensor<128xi32>, tensor<32xf32>) outs(%v105, %v106 : tensor<128x32xf32>, tensor<128x32xf32>) {
  ^bb0(%p: i32, %f: f32, %o0: f32, %o1: f32):
    %pf = arith.sitofp %p : i32 to f32
    %ext = arith.mulf %pf, %f : f32
    %fs = arith.constant 0x3F800000 : f32
    %theta = arith.mulf %fs, %ext : f32
    %mscale = arith.constant 0x3F800000 : f32
    %c0 = math.cos %theta : f32
    %s0 = math.sin %theta : f32
    %c = arith.mulf %c0, %mscale : f32
    %s = arith.mulf %s0, %mscale : f32
    linalg.yield %c, %s : f32, f32
  } -> (tensor<128x32xf32>, tensor<128x32xf32>)
  %v108 = tensor.expand_shape %v101 [[0], [1], [2], [3, 4]] output_shape [1, 128, 9, 32, 2] : tensor<1x128x9x64xf32> into tensor<1x128x9x32x2xf32>
  %v109 = tensor.extract_slice %v108[0, 0, 0, 0, 0] [1, 128, 9, 32, 1] [1, 1, 1, 1, 1] : tensor<1x128x9x32x2xf32> to tensor<1x128x9x32xf32>
  %v110 = tensor.extract_slice %v108[0, 0, 0, 0, 1] [1, 128, 9, 32, 1] [1, 1, 1, 1, 1] : tensor<1x128x9x32x2xf32> to tensor<1x128x9x32xf32>
  %v111 = tensor.empty() : tensor<1x128x9x32xf32>
  %v112 = tensor.empty() : tensor<1x128x9x32xf32>
  %v113, %v113b = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v109, %v110, %v107, %v107s : tensor<1x128x9x32xf32>, tensor<1x128x9x32xf32>, tensor<128x32xf32>, tensor<128x32xf32>) outs(%v111, %v112 : tensor<1x128x9x32xf32>, tensor<1x128x9x32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %c: f32, %s: f32, %o0: f32, %o1: f32):
    %x0c = arith.mulf %x0, %c : f32
    %x1s = arith.mulf %x1, %s : f32
    %y0 = arith.subf %x0c, %x1s : f32
    %x0s = arith.mulf %x0, %s : f32
    %x1c = arith.mulf %x1, %c : f32
    %y1 = arith.addf %x0s, %x1c : f32
    linalg.yield %y0, %y1 : f32, f32
  } -> (tensor<1x128x9x32xf32>, tensor<1x128x9x32xf32>)
  %v114 = tensor.empty() : tensor<1x128x9x32x2xf32>
  %v115 = tensor.insert_slice %v113 into %v114[0, 0, 0, 0, 0] [1, 128, 9, 32, 1] [1, 1, 1, 1, 1] : tensor<1x128x9x32xf32> into tensor<1x128x9x32x2xf32>
  %v116 = tensor.insert_slice %v113b into %v115[0, 0, 0, 0, 1] [1, 128, 9, 32, 1] [1, 1, 1, 1, 1] : tensor<1x128x9x32xf32> into tensor<1x128x9x32x2xf32>
  %v117 = tensor.collapse_shape %v116 [[0], [1], [2], [3, 4]] : tensor<1x128x9x32x2xf32> into tensor<1x128x9x64xf32>
  bufferization.materialize_in_destination %v117 in restrict writable %a29 : (tensor<1x128x9x64xf32>, memref<1x128x9x64xf32>) -> ()
  // MUL_MAT
  %v118 = bufferization.to_tensor %a30 restrict : memref<1x1x192x576xf16> to tensor<1x1x192x576xf16>
  %v120 = tensor.collapse_shape %v118 [[0, 1, 2], [3]] : tensor<1x1x192x576xf16> into tensor<192x576xf16>
  %v121 = tensor.collapse_shape %v91 [[0, 1, 2], [3]] : tensor<1x1x128x576xf32> into tensor<128x576xf32>
  %v123 = tensor.empty() : tensor<128x192xf32>
  %v124 = arith.constant 0.0 : f32
  %v125 = linalg.fill ins(%v124 : f32) outs(%v123 : tensor<128x192xf32>) -> tensor<128x192xf32>
  %v122 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v121, %v120 : tensor<128x576xf32>, tensor<192x576xf16>) outs(%v125 : tensor<128x192xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x192xf32>
  %v119 = tensor.expand_shape %v122 [[0, 1, 2], [3]] output_shape [1, 1, 128, 192] : tensor<128x192xf32> into tensor<1x1x128x192xf32>
  bufferization.materialize_in_destination %v119 in restrict writable %a31 : (tensor<1x1x128x192xf32>, memref<1x1x128x192xf32>) -> ()
  %v126 = tensor.collapse_shape %v119 [[0, 1, 2, 3]] : tensor<1x1x128x192xf32> into tensor<24576xf32>
  %v127 = tensor.expand_shape %v126 [[0, 1, 2, 3]] output_shape [1, 128, 3, 64] : tensor<24576xf32> into tensor<1x128x3x64xf32>
  // MUL_MAT
  %v128 = bufferization.to_tensor %a32 restrict : memref<1x1x192x576xf16> to tensor<1x1x192x576xf16>
  %v130 = tensor.collapse_shape %v128 [[0, 1, 2], [3]] : tensor<1x1x192x576xf16> into tensor<192x576xf16>
  %v131 = tensor.collapse_shape %v91 [[0, 1, 2], [3]] : tensor<1x1x128x576xf32> into tensor<128x576xf32>
  %v133 = tensor.empty() : tensor<128x192xf32>
  %v134 = arith.constant 0.0 : f32
  %v135 = linalg.fill ins(%v134 : f32) outs(%v133 : tensor<128x192xf32>) -> tensor<128x192xf32>
  %v132 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%v131, %v130 : tensor<128x576xf32>, tensor<192x576xf16>) outs(%v135 : tensor<128x192xf32>) {
  ^bb0(%a: f32, %b: f16, %acc: f32):
    %bw = arith.extf %b : f16 to f32
    %p = arith.mulf %a, %bw : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<128x192xf32>
  %v129 = tensor.expand_shape %v132 [[0, 1, 2], [3]] output_shape [1, 1, 128, 192] : tensor<128x192xf32> into tensor<1x1x128x192xf32>
  bufferization.materialize_in_destination %v129 in restrict writable %a33 : (tensor<1x1x128x192xf32>, memref<1x1x128x192xf32>) -> ()
  %v136 = tensor.collapse_shape %v129 [[0, 1, 2, 3]] : tensor<1x1x128x192xf32> into tensor<24576xf32>
  %v137 = tensor.expand_shape %v136 [[0, 1, 2, 3]] output_shape [1, 128, 3, 64] : tensor<24576xf32> into tensor<1x128x3x64xf32>
  // ROPE
  %v138 = arith.constant dense<[0x3F800000, 0x3F32A506, 0x3EF953CD, 0x3EADFCFD, 0x3E72D420, 0x3E297407, 0x3DEC7FD1, 0x3DA50953, 0x3D6655BD, 0x3D20BC18, 0x3CE054CB, 0x3C9C8B91, 0x3C5A7BE9, 0x3C1876FF, 0x3BD4CA0B, 0x3B947DA7, 0x3B4F3E2D, 0x3B109ED3, 0x3AC9D751, 0x3A8CD9D3, 0x3A449481, 0x3A092DFA, 0x39BF74CB, 0x39859AA0, 0x393A7746, 0x39021F22, 0x38B59B0E, 0x387D75C2, 0x3830DF44, 0x37F6DA83, 0x37AC4310, 0x37706B59]> : tensor<32xf32>
  %v139 = tensor.collapse_shape %v103 [[0, 1, 2, 3]] : tensor<1x1x1x128xi32> into tensor<128xi32>
  %v140 = tensor.empty() : tensor<128x32xf32>
  %v141 = tensor.empty() : tensor<128x32xf32>
  %v142, %v142s = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%v139, %v138 : tensor<128xi32>, tensor<32xf32>) outs(%v140, %v141 : tensor<128x32xf32>, tensor<128x32xf32>) {
  ^bb0(%p: i32, %f: f32, %o0: f32, %o1: f32):
    %pf = arith.sitofp %p : i32 to f32
    %ext = arith.mulf %pf, %f : f32
    %fs = arith.constant 0x3F800000 : f32
    %theta = arith.mulf %fs, %ext : f32
    %mscale = arith.constant 0x3F800000 : f32
    %c0 = math.cos %theta : f32
    %s0 = math.sin %theta : f32
    %c = arith.mulf %c0, %mscale : f32
    %s = arith.mulf %s0, %mscale : f32
    linalg.yield %c, %s : f32, f32
  } -> (tensor<128x32xf32>, tensor<128x32xf32>)
  %v143 = tensor.expand_shape %v137 [[0], [1], [2], [3, 4]] output_shape [1, 128, 3, 32, 2] : tensor<1x128x3x64xf32> into tensor<1x128x3x32x2xf32>
  %v144 = tensor.extract_slice %v143[0, 0, 0, 0, 0] [1, 128, 3, 32, 1] [1, 1, 1, 1, 1] : tensor<1x128x3x32x2xf32> to tensor<1x128x3x32xf32>
  %v145 = tensor.extract_slice %v143[0, 0, 0, 0, 1] [1, 128, 3, 32, 1] [1, 1, 1, 1, 1] : tensor<1x128x3x32x2xf32> to tensor<1x128x3x32xf32>
  %v146 = tensor.empty() : tensor<1x128x3x32xf32>
  %v147 = tensor.empty() : tensor<1x128x3x32xf32>
  %v148, %v148b = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v144, %v145, %v142, %v142s : tensor<1x128x3x32xf32>, tensor<1x128x3x32xf32>, tensor<128x32xf32>, tensor<128x32xf32>) outs(%v146, %v147 : tensor<1x128x3x32xf32>, tensor<1x128x3x32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %c: f32, %s: f32, %o0: f32, %o1: f32):
    %x0c = arith.mulf %x0, %c : f32
    %x1s = arith.mulf %x1, %s : f32
    %y0 = arith.subf %x0c, %x1s : f32
    %x0s = arith.mulf %x0, %s : f32
    %x1c = arith.mulf %x1, %c : f32
    %y1 = arith.addf %x0s, %x1c : f32
    linalg.yield %y0, %y1 : f32, f32
  } -> (tensor<1x128x3x32xf32>, tensor<1x128x3x32xf32>)
  %v149 = tensor.empty() : tensor<1x128x3x32x2xf32>
  %v150 = tensor.insert_slice %v148 into %v149[0, 0, 0, 0, 0] [1, 128, 3, 32, 1] [1, 1, 1, 1, 1] : tensor<1x128x3x32xf32> into tensor<1x128x3x32x2xf32>
  %v151 = tensor.insert_slice %v148b into %v150[0, 0, 0, 0, 1] [1, 128, 3, 32, 1] [1, 1, 1, 1, 1] : tensor<1x128x3x32xf32> into tensor<1x128x3x32x2xf32>
  %v152 = tensor.collapse_shape %v151 [[0], [1], [2], [3, 4]] : tensor<1x128x3x32x2xf32> into tensor<1x128x3x64xf32>
  bufferization.materialize_in_destination %v152 in restrict writable %a34 : (tensor<1x128x3x64xf32>, memref<1x128x3x64xf32>) -> ()
  return
}
