# Model Fitting/Checking {#model-checking}

As we leave simpler, analytically convenient models (often using conjugancy), Bayesian inference must use MCMC.
MCMC is powerful yet care and skill are needed to trust the results.
Trace plots were the original tool people used to assess convergence (mixing and stationarity) for earlier samplers (M-H, Gibbs).
Later split R_hat was used as quantitative evidence (p. 284-285, ch. 11.4, Gelman et al. Bayesian Data Analysis 3rd Edition).
One of the biggest benefits of HMC is that when it fails we often now that the sample had issues (energy was not conserved and so we have divergences).

*fitting in HMC*
- dignostics (all the common stan output ones please describe and interpret), effect sample size, divergences, etc.
- Devil's Funnel?

*prior predictive checks*
- fake data as a critical tool
- know that the software works (describe this well)
- know if our priors put too much mass in completely implausible regions
- I would avoid SBC as we didn't get that done.

transition to posterior predictive checks chapter.
remember this is a cycle of model refinement/comparison and that need to be conveyed somewhere 
fig 10 in for posterior predictive Gabry, Jonah, et al. "Visualization in Bayesian workflow." Journal of the Royal Statistical Society: Series A (Statistics in Society) 182.2 (2019): 389-402.
