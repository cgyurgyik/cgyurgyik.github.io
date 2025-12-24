+++
title = "Why scheduling languages exist"
date = 2025-12-23
draft = false
[extra] 
summary = "What Hoare wants and Rice forbids"
+++

C.A.R. Hoare's work ["Recursive data structures"](https://dl.acm.org/doi/10.5555/63445.C1104369) motivates a simple rule that good language design should abide by:

> (1) Implementation details that don't affect program correctness should be syntactically *inexpressible*.

Codd's [relational model](https://www.seas.upenn.edu/~zives/03f/cis550/codd.pdf) attempts to realize this: you declare *what* you want and the implementation is determined by the query optimizer. For example,

```sql
SELECT age FROM users WHERE age > 21
```

The declarative language specifies nothing about index usage, join order, or memory allocation. In practice, relational database programmers still reason about implementation details, shaping queries to optimize well. Most programming languages abandon (1) entirely, giving us:

> (2) Implementation details that don't affect program correctness are syntactically *expressible*. 

Imperative languages like C are the obvious case, but even [Haskell qualifies](https://dl.acm.org/doi/10.5555/645420.652528), where you make explicit decisions about memory representation:

```haskell
-- boxed (heap-allocated, thunk-able)
foo :: Int -> Int
foo x = x + 1

-- unboxed (raw machine integer)
foo :: Int# -> Int#
foo x = x +# 1#
```

Both snippets add `1` to the variable `x` with different data layouts. The choices are semantically invisible[^1] but syntactically different, and the latter exists solely for performance reasons. Scheduling languages like [Halide](https://dl.acm.org/doi/10.1145/2491956.2462176) and [TACO](https://dl.acm.org/doi/10.1145/3133901) take a different path:

> (3) Implementation details that don't affect program correctness are confined to a separate language.

In Halide, you write the algorithm once as a pure functional description, and then separately write a *schedule* specifying how the program should run. The schedule does not change the program semantics, only the execution strategy. This separation is enforced syntactically.

```cpp
// algorithm
f(x) = a(x) * b(x);

// schedule
f.parallel(x);
```

The algorithm `f` is a simple element wise multiplication of two dense arrays `a` and `b`, and the schedule specifies that `f` should be run in parallel for all `x`.

Why not just achieve (1) with a sufficiently clever compiler? Because determining which details "don't affect correctness" requires deciding program equivalence (in general, [Rice's theorem](https://en.wikipedia.org/wiki/Rice%27s_theorem) forbids this). Languages abiding to (2) entangle performance control and algorithm semantics, and languages following (3) sidestep the problem by narrowing the domain to something tractable. Further, (3) doesn't require the compiler to discover all semantics-preserving transformations[^2], they provide a language for the programmer to express them, while [guaranteeing](https://dl.acm.org/doi/10.1145/3519939.3523446) that only semantics-preserving transformations are expressible.

So, why does (3) exist? (1) provides clarity but surrenders performance control, and (2) provides control at the cost of entanglement. Programmers in performance-critical domains want both semantic clarity and performance control. Scheduling languages deliver: reason about correctness in one language, optimize in another. This separation is *one* motivation for scheduling languages. There's more to say (e.g., tractable search space, encapsulation benefits, ergonomic costs), but that's for another post.


<p style="font-size:10px"><b>Thank you to AJ Root and Rohan Yadav for their valuable feedback.</b></p>


[^1]: This is not true in general, the domain of unboxed types has no bottom element which just means they cannot be lazily evaluated. However, the point remains.

[^2]: Generally, scheduling languages also involve a compiler to realize some of the more "trivially" discoverable optimizations, e.g., constant folding or dead code elimination.