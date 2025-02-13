# FastFourierSAT

🎉 We are happy to share that our paper has been accepted by AAAI'25!
Find us on Saturday, March 1, 12:30 pm - 2:30 pm (Poster) or Sunday, March 2, 3:45 pm - 5:00 pm (Oral).

- `FastFourierSAT` is the vectorized version of [FourierSAT](https://github.com/vardigroup/FourierSAT) which enables GPU to achieve better performance.
- `FastFourierMaxSAT` is a partial MaxSAT solver, which uses CDCL to solve the hard constraints as other local search solvers. 

Paper: Massively Parallel Continuous Local Search for Hybrid SAT Solving on GPUs ([arxiv:2308.15020](https://arxiv.org/abs/2308.15020)).

If you have questions or thoughts regarding the tool or this work, please contact cenyunuo@u.nus.edu.

## Installation

1. If you have a CUDA-capable with a pre-installed NVIDIA driver (verify with `nvidia-smi`), please install CUDA 12.0 from [this link](https://developer.nvidia.com/cuda-downloads).
2. Add CUDA to PATH
    ```
    export CUDA_HOME="/usr/local/cuda-12.0"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
    export PATH="$CUDA_HOME/bin:$PATH"
    ```
3. Please use `nvcc -V` to verify the CUDA installation
4. Following the instruction on [this link](https://github.com/google/jax#installation) to install `JAX`. Generally, we use
    ```
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```
5. (Optional) If the above steps are unsuccessful, you can install the CPU version
    ```
    pip install --upgrade "jax[cpu]"
    ```
6. Install `JAXopt`
    ```
    pip install jaxopt
    ```
7. (Optional) Install `PySAT` for `FastFourierMaxSAT`
    ```
    pip install python-sat
    ```

## Benchmark Instances Download and Generation
1. Unweighted instances from anytime track on MaxSAT Evaluation 2023 can be downloaded from [this link](https://www.cs.helsinki.fi/group/coreo/MSE2023-anytime-instances/MSE2023-anytime-UW-benchmarks.zip).
2. To download instances from SAT Competition 2023, first download [track_main_2023.uri](https://benchmark-database.de/getinstances?track=main_2023), then run
    ```
    wget --content-disposition -i track_main_2023.uri
    ```
3. Generate hybrid SAT formulas for Benchmark 2:
    ```
    python Benchmark/Cardinality_Constraint/generate.py
    python Benchmark/Parity_Learning_w_Error/generate.py
    ```
4. Generate hybrid MaxSAT formulas for Benchmark 3:
    ```
    ./Benchmark/Graph_Coloring/generate.sh
    ./Benchmark/Hamiltonian_Cycle/generate.sh
    ```

## Usage
- Cardinality constraints instances from Benchmark 2
    ```
    python Solver/FastFourierSAT/card.py [Benchmark/Cardinality_Constraint/*/*.hybrid File]
    ```
- Parity learning with error instances from Benchmark 2
    ```
    python Solver/FastFourierSAT/ple.py [Benchmark/Parity_Learning_w_Error/*/*.hybrid File] --tolerance [tolerance]
    ```
- Graph instances from Benchmark 3
    ```
    python Solver/FastFourierMaxSAT/glucard_init_hybrid.py [Weighted .hybrid File]
    python Solver/FastFourierSAT/no_init_hybrid.py [Weighted .hybrid File]
    ```
- CNF instances
    ```
    python Solver/FastFourierSAT/cnf.py [.cnf File]
    ```
- WCNF instances
    ```
    python Solver/FastFourierMaxSAT/cadical_init_cnf.py [.wcnf File]
    python Solver/FastFourierSAT/no_init_wcnf.py [.wcnf File]
    ```
