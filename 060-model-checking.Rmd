# Model Checking {#model-checking}

- The problem of simulating multivariate data with arbitrary marginal distributions
- Copula approach
  - Nonlinear transformation that invalidates the correlation structure
- Kendall and Spearman matching
  - Nearest Positive Semidefinite correlation matrix
    - Semidefinte Programming (ProxSDP.jl)
    - https://arxiv.org/abs/1810.05231
    - Qi and Sun 2006 (quadratically convergent method)
- Pearson matching
  - Chen 2001 (NORTARA)
  - Xiao, Zhou 2019 (Numeric Approximation)
- Using synthetic data to design experiments
  - Bayesian p-value
  - How much data to notice an effect
  - Bayesian hypothesis testing via predictive performance

> Complementing a calibration free work ow is a pure simulation study that studies the potential problems of a model, and the experimental design it encodes, solely within the assumptions of that model. This is a powerful way both to evaluate experiments before the experiment is build – let alone any data are collected – and to study the behavior of particular modeling techniques in isolation.
