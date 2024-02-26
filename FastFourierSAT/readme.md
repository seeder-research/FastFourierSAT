## About
- `FastFourierSAT` is an extension of the continuous local search SAT solver `FourierSAT` with GPU support

## Installation

1. If you have a CUDA-capable with pre-installed NVIDIA driver (verify with `nvidia-smi`), please install CUDA 12.0 from [this link](https://developer.nvidia.com/cuda-downloads).
2. Add CUDA to PATH
    ```
    export CUDA_HOME="/usr/local/cuda-12.0"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
    export PATH="$CUDA_HOME/bin:$PATH"
    ```
3. Please use `nvcc -V` to verify the CUDA installation
4. Following the instruction on [this link](https://github.com/google/jax#installation) to install `JAX`. Generally we use
    ```
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```
5. If the above steps are unsuccessful, you can install the CPU version
    ```
    pip install --upgrade "jax[cpu]"
    ```
6. Install `JAXopt`
    ```
    pip install jaxopt
    ```

## Usage
- Cardinality constraints
    ```
    python card.py [CNF+ File]
    ```
- Parity learning with error
    ```
    python xor.py [CNF+ File] --tolerance 8
    ```
- Weighted Max-Cut
    ```
    python xor.py [WCNF+ File] --weighted 1
    ```