#!/bin/bash
# Build unsloth llama.cpp branch glm5next/upstream (GLM-5.3-Flash support) for V100 (sm_70), ppc64le.
# Sources: tarball of the commit in ~/llama.cpp-glm5next/UNSLOTH_COMMIT (separate from ~/llama.cpp).
set -e
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
cd ~/llama.cpp-glm5next
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release -j 120 --target llama-server llama-cli llama-bench
echo BUILD_OK
ls -la build/bin/
