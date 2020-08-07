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
