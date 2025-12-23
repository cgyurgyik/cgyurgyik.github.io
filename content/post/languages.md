+++
title = "Why scheduling languages exist"
date = 2025-12-23
draft = false
[extra] 
summary = "What Hoare wants and Rice forbids"
+++

C.A.R. Hoare's work ["Recursive data structures"](https://dl.acm.org/doi/10.5555/63445.C1104369) motivates a simple rule that good language design should abide by:

> (1) Implementation details that don't affect program correctness are syntactically *inexpressible*.

Codd's [relational model](https://www.seas.upenn.edu/~zives/03f/cis550/codd.pdf) attempts to realize this: you declare *what* you want (projections, joins) and the implementation (e.g., join order) is determined by the optimizer. In practice, programmers still reason about implementation details, shaping queries to optimize well. Programming languages generally abandon (1) entirely, giving us:

> (2a) Implementation details that don't affect program correctness are syntactically *expressible*. 

C++ is the obvious case, but even [Haskell qualifies](https://dl.acm.org/doi/10.5555/645420.652528). You make explicit decisions about memory representation:

```haskell
-- unboxed (raw machine integer)
foo :: Int# -> Int#
foo x = x +# 1#

-- boxed (heap-allocated, thunk-able)
foo :: Int -> Int
foo x = x + 1
```

These choices are semantically invisible but syntactically required for performance reasons. Scheduling languages like [Halide](https://dl.acm.org/doi/10.1145/2491956.2462176) and [TACO](https://dl.acm.org/doi/10.1145/3133901) take a different path:

> (3) Implementation details that don't affect program correctness are confined to a separate language.

In Halide, you write the algorithm once as a pure functional description, and then separately write a *schedule* specifying optimizations such as tiling and parallelism. The schedule does not change the program semantics, only how the compute is performed. This separation is enforced syntactically.

```
// algorithm
f(x) = a(x) * b(x);

// schedule
f.parallel(x);
```

Why not just achieve (1) with a sufficiently clever compiler? Because determining which details "don't affect correctness" requires deciding program equivalence (in general, [Rice's theorem](https://en.wikipedia.org/wiki/Rice%27s_theorem) forbids this). Scheduling languages today sidestep the problem for a small domain. They don't ask the compiler to discover semantics-preserving transformations, they provide a language for the programmer to express them, while [guaranteeing by construction](https://dl.acm.org/doi/10.1145/3519939.3523446) that only semantics-preserving transformations are expressible.

There's more to say about scheduling languages (e.g., encapsulation benefits, ergonomic costs), but that's for another post. For now, (1) is the ideal, (3) is the pragmatic alternative when optimal performance is desired.

