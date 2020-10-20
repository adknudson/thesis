


## Iteration 2 (Electric Boogaloo)

### Model Development {#mod-dev-iter2}

In this iteration we will now add in the treatment and age group levels. Instead of modeling the prior distribution of the slope as log-normal, we model it as a normal distribution and then take the exponential. This allows us to also model the age group and treatment slopes as normally distributed and with an additive affect.


\begin{align*}
\beta &\sim \mathcal{N}(2.5, 1^2) \\
\beta_G &\sim \mathcal{N}(0, \sigma_{\beta G}^2) \\
\beta_T &\sim \mathcal{N}(0, 1^2) \\
\beta_{TG} &\sim \mathcal{N}(0, \sigma_{\beta TG}^2) \\
\mu_\beta &\sim \exp(\beta + \beta_G + (\beta_T + \beta_{TG})\times trt)
\end{align*}


In the above formulation, $\mu_\beta$ is a log-normal random variable with mean-log $2.5$ and variance-log $1^2 + \sigma_{\beta G}^2$ if it's the pre-adaptation block, and $\left(\sqrt{1^2 + 1^2}\right)^2 + \sigma_{\beta G}^2 + \sigma_{\beta TG}^2$ if it's the post-adaptation block. Values that are negative reduce the slope (increase the JND), and values that are positive increase the slope (reduce the JND). 

But wait! this model implies that there is more uncertainty about the post-adaptation trials compared to the baseline trials, and this is not necessarily true. Furthermore, as we'll see in the linear part of model, the intercept, $\alpha$, is no longer the average response probability of the sample, but is instead exclusively the average for the pre-adaptation trials. This may not matter in certain analyses, but one nice property of multilevel models is the separation of population level estimates and group level estimates. So we modify the model for the slope to be:


\begin{align*}
\beta &\sim \mathcal{N}(2.5, 1^2) \\
\beta_G &\sim \mathcal{N}(0, \sigma_{\beta G}^2) \\
\beta_T &\sim \mathcal{N}(0, \sigma_{\beta T}^2) \\
\mu_\beta &= \exp(\beta + \beta_G + \beta_T)
\end{align*}


Now $\mu_\beta$ is a log-normal random variable with mean-log $2.5$ and variance-log $1^2 + \sigma_{\beta G}^2 + \sigma_{\beta T}^2$, regardless of whether it's the pre-adaptation or the post-adaptation block.

The intercept term can be specified similarly. Conservatively we choose the prior for the intercepts to be normally distributed with mean 0.


\begin{align*}
\alpha &\sim \mathcal{N}(0, 0.05^2) \\
\alpha_G &\sim \mathcal{N}(0, \sigma_{\alpha G}^2) \\
\alpha_T &\sim \mathcal{N}(0, \sigma_{\alpha T}^2) \\
\mu_\alpha &= \alpha + \alpha_{G} + \alpha_{T}
\end{align*}


The parameters and model of the Stan program is

```
parameters {
  real a;
  real aG[N_G];
  real aT[N_T];
  
  real b;
  real bG[N_G];
  real bT[N_T];
  
  real<lower=machine_precision()> sd_aG;
  real<lower=machine_precision()> sd_aT;
  real<lower=machine_precision()> sd_bG;
  real<lower=machine_precision()> sd_bT;
}
model {
  a ~ normal(0, 0.05);
  aG ~ normal(0, sd_aG);
  aT ~ normal(0, sd_aT);
  
  b ~ normal(2.5, 1.0);
  bG ~ normal(0, sd_bG);
  bT ~ normal(0, sd_bT);
  
  sd_aG ~ cauchy(0, 0.05);
  sd_aT ~ cauchy(0, 0.05);
  sd_bG ~ cauchy(0, 0.5);
  sd_bT ~ cauchy(0, 0.5);
  
  vector[N] p;
  for (i in 1:N) {
    real mu_b = exp(b + bG[G[i]] + bT[T[i]]);
    real mu_a = a + aG[G[i]] + aT[T[i]];
    p[i] = mu_b * (x[i] - mu_a);
  }
  k ~ binomial_logit(n, p); // Observational model
}
```




###  Simulate bayesian ensemble






### Prior Checks








### Configure algorithm

### Fit simulated ensemble







### Algorithmic calibration



Additionally we were given the warning that the Bulk ESS is too low, and that running the chains for more iterations can help. So we do just that, and also increase the adapt delta.





### Inferential Calibration

### Fit Observation





### Diagnose posterior fit



It's looking alright!

### Posterior retrodictive checks








We're getting closer to an acceptable model, but our model's retrodictions still do not represent the younger pre-adaptation data very well. Specifically the retrodictions are still under-dispersed.
