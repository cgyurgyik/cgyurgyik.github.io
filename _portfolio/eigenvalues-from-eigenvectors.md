---
title: "Eigenvectors from Eigenvalues"
excerpt: "Based off a recent mathematical discovery, this implements the formula necessary to compute normed eigenvectors given only the eigenvalues of a Hermitian matrix.<br/><img src='/images/eigenvalues-from-eigenvectors/graph.png'>"
collection: portfolio
---

[Github repository](https://github.com/cgyurgyik/eigenvectors-from-eigenvalues)

In ["EIGENVECTORS FROM EIGENVALUES: A SURVEY OF A BASIC IDENTITY IN LINEAR ALGEBRA"](https://arxiv.org/pdf/1908.03795.pdf) by Denton et al., the _eigenvalue-eigenvector identity_ was rediscovered, and with it comes the following formula: 

<img src="https://latex.codecogs.com/gif.latex?\newline&space;\left&space;|&space;v_{i,j}&space;\right&space;|^{2}&space;=&space;\dfrac{\prod_{k=1}^{n-1}&space;(\lambda_{i}(H)&space;-&space;\lambda_{k}(h_{j}))}&space;{&space;\prod_{1&space;\leq&space;k&space;\leq&space;n}^{i&space;\neq&space;k}&space;(\lambda_{i}(H)&space;-&space;\lambda_{k}(H))}&space;\newline\newline\newline&space;{h_{j}}:&space;(n-1)\times(n-1)&space;\texttt{&space;matrix&space;with&space;jth&space;row&space;and&space;jth&space;column&space;removed.}&space;\newline&space;\lambda(H):&space;\texttt{&space;eigenvalues&space;of&space;}&space;H.&space;\newline&space;\lambda(h_{j}):&space;\texttt{&space;eigenvalues&space;of&space;}&space;h_{j}." title="\newline \left | v_{i,j} \right |^{2} = \dfrac{\prod_{k=1}^{n-1} (\lambda_{i}(H) - \lambda_{k}(h_{j}))} { \prod_{1 \leq k \leq n}^{i \neq k} (\lambda_{i}(H) - \lambda_{k}(H))} \newline\newline\newline {h_{j}}: (n-1)\times(n-1) \texttt{ matrix with jth row and jth column removed.} \newline \lambda(H): \texttt{ eigenvalues of } H. \newline \lambda(h_{j}): \texttt{ eigenvalues of } h_{j}." />

This project reimplements the formula in both MATLAB and C++, throwing in some mischievous comparisons to MATLAB's [eig](https://www.mathworks.com/help/matlab/ref/eig.html) function, which also produces the eigenvectors of a given matrix. This was provoked by commentary from Cornell University's [Professor A. Townsend](https://github.com/ajt60gaibb) during a Linear Algebra lecture, which pondered the idea that this formula may indeed be useful in cases where only a few eigenvectors are needed from large matrices. While I cannot think of such a case in practice, it was an interesting idea to play with.