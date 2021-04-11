---
title: "Eigenvectors from Eigenvalues"
excerpt: "Based off a recent mathematical discovery, this implements the formula necessary to compute normed eigenvectors given only the eigenvalues of a Hermitian matrix.<br/><img src='/images/eigenvalues-from-eigenvectors/graph.png'>"
collection: portfolio
---

In ["EIGENVECTORS FROM EIGENVALUES: A SURVEY OF A BASIC IDENTITY IN LINEAR ALGEBRA"](https://arxiv.org/pdf/1908.03795.pdf) by Denton et al., the _eigenvalue-eigenvector identity_ was rediscovered, and with it comes the following formula: 

![Equation](images/eigenvalues-from-eigenvectors/equation.png)

This project reimplements the formula in both MATLAB and C++, throwing in some mischievous comparisons to MATLAB's [eig](https://www.mathworks.com/help/matlab/ref/eig.html) function, which also produces the eigenvectors of a given matrix. This was provoked by commentary from Cornell University's [Professor A. Townsend](https://github.com/ajt60gaibb) during a Linear Algebra lecture, which pondered the idea that this formula may indeed be useful in cases where only a few eigenvectors are needed from large matrices. While I cannot think of such a case in practice, it was an interesting idea to play with.