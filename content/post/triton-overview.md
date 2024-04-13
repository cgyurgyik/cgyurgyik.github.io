+++ 
title = "So what is the Triton Language?"
date = 2024-04-11
draft = true

[extra] 
summary = "An outsider's perspective on the Triton language."
+++

# Triton Language: A Thousand Foot Perspective

This is an introduction to Triton, a domain specific language (DSL) embedded in Python, intermediate representation, and compiler. Triton is designed to write efficient GPU code with little CUDA experience by letting the compiler handle the heavy lifting: memory management and instruction selection.

## The GPU Programming Language △

First, I define a GPU programming language as one that can, by means of compilation or interpretation, run on a GPU. Many programming languages and libraries exist today to write GPU programs. These range from functional languages that hide the entire GPU execution model, e.g., Futhark (REF), to bare-metal languages, e.g., CUDA (REF), that are just a stone's throw away from assembly. Presented below is
the programming language triangle, but in this case for languages intended to lower to GPU:

TODO: insert triangle picture here

Safety: Descend
Productivity: Futhark, Numpy (don't need to think at all about GPU)
Prod-Perf: CuPy -> Triton -> Numba
Performance: CUDA, OpenCL, HIP

Like any claim related to programming languages, this is highly subjective. Here, we define *productivity* as a function of the cognitive effort required to reason about the GPU programming model. Consider the example matrix multiply `C = A @ B` below, written in Futhark:

```haskell
-- (Written in Futhark v0.25.15)
def matmul [i][k][j] (A: [i][k]i32) (B: [k][j]i32) : [i][j]i32 =
  map (\Ai ->
    map (\Bj ->
      reduce (+) 0 (map \(x, y) -> x * y (zip Ai Bj)))
        (transpose B)
  ) 
    A
```

This programming model does not allude to threads, blocks, or memory - instead, the user relies entirely upon the compiler to ensure the code is efficient. 

Conversely, Descend (REF) is a "safe-by-construction" imperative language. We define *safety* as a measurement of how much guarantee you have that your program is safe. Inspired by Rust, Descend guarantees legal CPU and GPU memory management at compile time by tracking Ownership and Lifetimes. Following the previous example, we write matrix multiply in Descend:

```rust
// (Descend does not have releases; written on 12 April 2024.)
fn matmul<BLOCK: nat, i: nat, j: nat, k: nat, r: prv>(
    A: &r shrd gpu.global [i32; i*k],
    B: &r shrd gpu.global [i32; k*j],
    C: &r uniq gpu.global [i32; i*j]
) -[grid: gpu.grid<XY<j/BLOCK, i/BLOCK>, XY<BLOCK, BLOCK>>] -> () {
    sched(Y) block_row in grid {
        sched(X) block in block_row {
            sched(Y) thread_row in block {
                sched(X) thread in thread_row {
                    let Ai = &shrd (*A)
                        .to_view
                        .grp::<k>
                        .grp::<BLOCK>[[block_row]][[thread_row]];
                    let Bj = &shrd (*B)
                        .to_view
                        .grp::<j>
                        .transp
                        .grp::<BLOCK>[[block]][[thread]];
                    let mut Cij = &uniq (*C)
                        .to_view
                        .grp::<j>
                        .grp::<BLOCK>[[block_row]]
                        .transp
                        .grp::<BLOCK>[[block]]
                        .transp[[thread_row]][[thread]];
                    
                    let mut sum = 0;
                    for e in 0..k { 
                      sum = sum + (*Ai)[e] * (*Bj)[e] 
                    };
                    *Cij = sum
                }
            }
        }
    }
}
```

Lastly, a language like CUDA allows a performance engineer to squeeze out every drop of performance. We define *performance* as a metric of how much control a user has over the GPU. Writing a truly efficient program in CUDA is a non trivial task, as we shall present soon. Moreover, transforming an inefficient program to an efficient one, e.g., using shared memories, requires a substanial code rewrite. Consider a naive matrix multiply kernel written in CUDA versus one that uses shared memory:

TODO: Side-by-side diff.

```cpp
__global__ void matmul(const int *A, const int *B, int *C, int n) {
  int Ai = blockIdx.y * blockDim.y + threadIdx.y;
  int Bj = blockIdx.x * blockDim.x + threadIdx.x;

  int temporary = 0;
  for (int k = 0; k < n; k++) {
    temporary += A[Ai * n + k] * B[k * n + Bj];
  }
  C[Ai * n + Bj] = temporary;
}

__global__ void matmul(const int *A, const int *B, int *C, int n) {
  int Ai = blockIdx.y * blockDim.y + threadIdx.y;
  int Bj = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int shA[n];
  __shared__ int shB[n];

  int temporary = 0;
  for (int i = 0; i < n; i += blockDim.x) {
    shA[threadIdx.y * blockDim.x + threadIdx.x] = A[Ai * n + i + threadIdx.x    ];
    shB[threadIdx.y * blockDim.x + threadIdx.x] = B[Bj + n * i + threadIdx.y * n];

    __syncthreads(); // Wait until all threads finish loading to shared memory.
    for (int j = 0; j < blockDim.x; j++) {
      temporary +=
        shA[threadIdx.y * blockDim.x + j] * shB[j * blockDim.x + threadIdx.x];
    }
    __syncthreads(); // Wait until all threads finish reading from shared memory.
  }
  C[Ai * n + Bj] = temporary;
}
```

To this end, Numba (REF) provides a similar interface to CUDA but written in Python, with some additional support for obtaining the thread position in a grid. From this point on, we'll use Numba to compare the CUDA SIMT programming model with the Triton programming model.

## What is Triton?
Triton is an imperative language and compiler stack to simplify the arduous process of writing GPU kernels. Antithetical to the SIMT model, users write programs that load, compute upon, and store *blocks* of memory. These blocks are accessed via a pointer interface. Then, the compiler automatically handles optimizations such as multi-threading, using fast memory, tensor cores, etc. So, the user must handle the outermost level of tiling, via loads and stores to global memory, and then the compiler handles the rest. To begin, we'll compare a simple program `B = A + 1`, where `|A| = |B| = n`.


TODO: draw picture of each programming model underneath
TODO: side-by-side

```py
@triton.jit
def add1_kernel(A, B, n, BLOCK):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offsets, mask=offsets < n)
    out = a + 1
    tl.store(B + offsets, out, mask=offsets < n)


@cuda.jit
def add1_kernel(A, B, n, BLOCK):
    tid = blockIdx.x * BLOCK + threadIdx.x
    if tid < n:
        out = A[tid] + 1
        B[tid] = out
```

The Triton kernel is working on a block of threads, whose position in a grid is indicated by the `program_id`. The offsets are a block of pointers that will be used to determine where in global memory we want to load. These are masked off to ensure we don't exceed the length of the array `n`.

### Who uses it?
The Triton language and compiler stack is currently open source under OpenAI (REF). Additionally, PyTorch 2.0 (REF) translates PyTorch programs into Triton in its new compiler backend, TorchInductor. Lastly, JAX (ref) uses Triton as a GPU backend target for its new kernel programming model, Pallas (REF).

### Strengths
Writing optimal GPU programs is hard. For a memory-bound kernel, one must consider, at a minimum, the following:
- Memory coalescing (in global memory): thread access patterns are important to ensure we minimize the number of fetches.
- Memory hierarchy (global $\rightarrow$ shared $\rightarrow$ registers). Using a lower-latency memory is better, but requires synchronization and low-level instructions such as intra-warp shuffles.
- Bank conflicts (in shared memory): data structure layout is important to avoid bank conflicts, e.g., Area of Structures (AoS) versus Structure of Arrays (SoA).

For a bandwidth-bound kernel such as matrix multiply or convolution, one must map instructions to Tensor Cores. This requires carefully choosing tile size, SM count, etc. Such optimizations are not always trivial; this can be illustrated by the number of NVIDIA Developer blog posts (REF).

By performing block-level data flow analysis, the Triton language can *automatically* unlock optimizations such as memory coalescing, thread swizzling, pre-fetching, vectorization, instruction selection (e.g. Tensor Core), shared memory allocation and synchronization, and more.

Additionally, Triton provides a few other important features. It has native auto-tuning support, 

### Weaknesses
This is a performance *savant*'s worst nightmare: we ultimately become victim to a "black box" compiler that we can only hope will optimize our kernel. Worse yet, we don't really have a way out - there exist optimizations that aren't clearly accessible from the high-level abstraction of Triton.

There is an interesting (yet slightly outdated (FOOTNOTE)) blog post by a senior engineer at NVIDIA (REF), who inspects the emitted code from the Triton compiler.

FOOTNOTE: Add footnote saying that readers should be aware this is from Triton 1.0. I have run a few of these on Triton 2.1.0, and found that they improved since, e.g., removal of unnecessary thread synchronizations in reduction.

## More complex example to motivate the need for Triton
Triton provides a much simpler programming model that realizes the simplicity in most GPU programs: load data, perform operations on that data, store data.

Matrix multiply naive: (CUDA)
Matrix multiply tiled: (CUDA)

Matrix multiply naive: (TRITON)
Matrix multiply tiled: (TRITON)

(MAYBE): Demonstrate how easy fusion is, i.e., add a leaky ReLU
`acc = tl.where(acc >= 0, acc, alpha * acc)`


Demystifying OpenAI Triton: https://fkong.tech/posts/2023-04-23-triton-cuda/

Ask: Brennan Shacklett, Rohan, Nathan for feedback.

Blocked algorithms: https://dl.acm.org/doi/pdf/10.1145/106973.106981

PyTorch 2: https://pytorch.org/assets/pytorch2-2.pdf

Using shared memory in CUDA: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/

Transpose using shared memory: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

CUDA warp-level primitives: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

Using atomics for fast histogram kernel: https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/

Programming Tensor Cores: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
