set CMAKE_BUILD_PARALLEL_LEVEL=%NUMBER_OF_PROCESSORS%
set MAKEFLAGS=-j%NUMBER_OF_PROCESSORS%
set CMAKE_ARGS=-DGGML_CUDA=ON
python -m pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --verbose
