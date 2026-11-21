#!/usr/bin/env bash
# Set up AMD's open NPU compilers (MLIR-AIR, MLIR-AIE, and the llvm-aie code generator)
# in an x86-64 Ubuntu 24.04 container. No NPU hardware or XRT runtime is needed to compile.
#
#   scripts/amd-toolchain.sh          create the container "ggml-air" (repo mounted at /work)
#   docker exec -it ggml-air bash     then: source /opt/env.sh
#
# On Apple Silicon the container runs under x86 emulation. Tool versions are whatever AMD's
# wheel pages serve today (nightly builds), so record `pip list` next to any result.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"

docker rm -f ggml-air >/dev/null 2>&1 || true
docker run -d --name ggml-air --platform linux/amd64 -v "$repo":/work ubuntu:24.04 sleep infinity

docker exec ggml-air bash -euo pipefail -c '
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git make clang lld ninja-build unzip

python3 -m venv /opt/airenv
. /opt/airenv/bin/activate
pip install -q --upgrade pip
pip install -q "mlir_air[aie]" \
    -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-5 \
    -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

cat > /opt/env.sh <<"ENV"
. /opt/airenv/bin/activate
site=$(python3 -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")
export MLIR_AIR_INSTALL_DIR=$site/mlir_air
export MLIR_AIE_INSTALL_DIR=$site/mlir_aie
export PEANO_INSTALL_DIR=$site/llvm-aie
export PATH=$MLIR_AIR_INSTALL_DIR/bin:$MLIR_AIE_INSTALL_DIR/bin:$PATH
export PYTHONPATH=$MLIR_AIR_INSTALL_DIR/python:$MLIR_AIE_INSTALL_DIR/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$MLIR_AIR_INSTALL_DIR/lib:$MLIR_AIE_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}
ENV

# AMD examples (the wheels do not include them)
git clone -q --depth 1 https://github.com/Xilinx/mlir-air.git /opt/mlir-air

. /opt/env.sh
pip list 2>/dev/null | grep -iE "mlir|llvm"
air-opt --version | grep "LLVM version"
'
