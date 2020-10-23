


## Iteration 2 (Electric Boogaloo) {#iter2}

### Model Development {#iter2-model-dev}

In this iteration we will now add in the treatment and age group levels. Instead of modeling the prior distribution of the slope as log-normal, we model it as a normal distribution and then take the exponential. This allows us to also model the age group and treatment slopes as normally distributed and with an additive affect.


\begin{align*}
\beta &\sim \mathcal{N}(3.0, 1.5^2) \\
\beta_G &\sim \mathcal{N}(0, \sigma_{\beta G}^2) \\
\beta_T &\sim \mathcal{N}(0, 1.0^2) \\
\beta_{TG} &\sim \mathcal{N}(0, \sigma_{\beta TG}^2) \\
\mu_\beta &\sim \exp(\beta + \beta_G + (\beta_T + \beta_{TG})\times trt)
\end{align*}


In the above formulation, $\mu_\beta$ is a log-normal random variable with mean-log $3.0$ and variance-log $1.5^2 + \sigma_{\beta G}^2$ if it's the pre-adaptation block, and $\left(\sqrt{1.5^2 + 1.0^2}\right)^2 + \sigma_{\beta G}^2 + \sigma_{\beta TG}^2$ if it's the post-adaptation block. Values that are negative reduce the slope (increase the JND), and values that are positive increase the slope (reduce the JND).

But wait! this model implies that there is more uncertainty about the post-adaptation trials compared to the baseline trials, and this is not necessarily true. Furthermore, as we'll see in the linear part of model, the intercept, $\alpha$, is no longer the average response probability of the sample, but is instead exclusively the average for the pre-adaptation trials. This may not matter in certain analyses, but one nice property of multilevel models is the separation of population level estimates and group level estimates. So we modify the model for the slope to be:


\begin{align*}
\beta &\sim \mathcal{N}(3.0, 1.5^2) \\
\beta_G &\sim \mathcal{N}(0, \sigma_{\beta G}^2) \\
\beta_T &\sim \mathcal{N}(0, \sigma_{\beta T}^2) \\
\mu_\beta &= \exp(\beta + \beta_G + \beta_T)
\end{align*}


Now $\mu_\beta$ is a log-normal random variable with mean-log $3.0$ and variance-log $1.5^2 + \sigma_{\beta G}^2 + \sigma_{\beta T}^2$, regardless of whether it's the pre-adaptation or the post-adaptation block.

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
  
  real<lower=0> sd_aG;
  real<lower=0> sd_aT;
  real<lower=0> sd_bG;
  real<lower=0> sd_bT;
}
model {
  a ~ normal(0, 0.05);
  aG ~ normal(0, sd_aG);
  aT ~ normal(0, sd_aT);
  
  b ~ normal(3.0, 1.5);
  bG ~ normal(0, sd_bG);
  bT ~ normal(0, sd_bT);
  
  sd_aG ~ cauchy(0, 0.01);
  sd_aT ~ cauchy(0, 0.01);
  sd_bG ~ cauchy(0, 0.5);
  sd_bT ~ cauchy(0, 0.5);
  
  vector[N] p;
  for (i in 1:N) {
    real mu_b = exp(b + bG[G[i]] + bT[T[i]]);
    real mu_a = a + aG[G[i]] + aT[T[i]];
    p[i] = mu_b * (x[i] - mu_a);
  }
  k ~ binomial_logit(n, p);
}
```


Just like in the first iteration, we begin by simulating the observational model.


###  Simulate bayesian ensemble {#iter2-sim}






### Prior Checks {#iter2-prior-check}

<img src="051-bayesian-workflow_files/figure-html/ch051-Surreal Comic-1.png" width="70%" style="display: block; margin: auto;" />


<img src="051-bayesian-workflow_files/figure-html/ch051-Severe Kangaroo-1.png" width="70%" style="display: block; margin: auto;" />


```
#> , , 1
#> 
#>        
#>              [,1]      [,2]      [,3]
#>   50%   8.400e-02 8.600e-02 8.000e-02
#>   95%   7.270e+00 5.816e+00 5.376e+00
#>   99%   1.689e+02 3.366e+02 1.350e+02
#>   99.9% 5.760e+06 3.344e+09 1.785e+07
#> 
#> , , 2
#> 
#>        
#>              [,1]      [,2]      [,3]
#>   50%   8.000e-02 7.800e-02 7.400e-02
#>   95%   5.205e+00 5.288e+00 4.041e+00
#>   99%   8.859e+02 1.368e+03 3.881e+02
#>   99.9% 2.720e+06 1.299e+08 1.875e+05
```

### Configure algorithm {#iter2-config-algo}

### Fit simulated ensemble {#iter2-fit-sim}










### Algorithmic calibration {#iter2-algo-calibration}


```
#> 
#> Divergences:
#> 103 of 5000 iterations ended with a divergence (2.06%).
#> Try increasing 'adapt_delta' to remove the divergences.
#> 
#> Tree depth:
#> 0 of 5000 iterations saturated the maximum tree depth of 10.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```

Let's see if we can do a little better by reparameterizing the model, and tuning the algorithm a bit. As I'll discuss more in the [model checking](#model-checking) section, we can use the non-centered parameterization of the normal and Cauchy distributions to make it easier for the Hamiltonian Monte Carlo algorithm to explore the posterior. Additionally Stan suggests increasing the `adapt_delta` parameter to remove divergences, so we will do that. Finally, to take care of the message about R-hat and effective sample sizes, I will run the chains for more iterations.







```
#> 
#> Divergences:
#> 627 of 20000 iterations ended with a divergence (3.135%).
#> Try increasing 'adapt_delta' to remove the divergences.
#> 
#> Tree depth:
#> 0 of 20000 iterations saturated the maximum tree depth of 10.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```

Excellent! Now only about half a percent of the transitions are divergent, and there are no longer any warnings about the R-hat statistic.

### Inferential Calibration {#iter2-inferential-calibration}

### Fit Observation {#iter2-fit-obs}







### Diagnose posterior fit {#iter2-diagnose-post}


```
#> 
#> Divergences:
#> 48 of 20000 iterations ended with a divergence (0.24%).
#> Try increasing 'adapt_delta' to remove the divergences.
#> 
#> Tree depth:
#> 0 of 20000 iterations saturated the maximum tree depth of 10.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```

It's looking alright!


```
#>          mean se_mean     sd    2.5%  97.5% n_eff Rhat
#> a      0.0317   2e-04 0.0174 -0.0083 0.0629  8585    1
#> aG[1] -0.0043   1e-04 0.0142 -0.0309 0.0270  9696    1
#> aG[2]  0.0212   2e-04 0.0150 -0.0032 0.0557  8552    1
#> aG[3] -0.0099   1e-04 0.0143 -0.0387 0.0202 10241    1
#> aT[1]  0.0066   1e-04 0.0122 -0.0122 0.0362  9371    1
#> aT[2] -0.0034   1e-04 0.0116 -0.0263 0.0212 10156    1
#>          mean se_mean     sd    2.5%  97.5% n_eff Rhat
#> b      2.2242  0.0045 0.3639  1.5030 3.0457  6409    1
#> bG[1]  0.0461  0.0021 0.1964 -0.3710 0.4348  8561    1
#> bG[2]  0.0973  0.0021 0.1964 -0.3136 0.4869  8559    1
#> bG[3] -0.1766  0.0021 0.1971 -0.6036 0.2027  8477    1
#> bT[1] -0.1445  0.0040 0.3155 -0.8963 0.4765  6073    1
#> bT[2]  0.0762  0.0040 0.3148 -0.6675 0.7076  6156    1
#>            mean se_mean     sd   2.5%  97.5% n_eff   Rhat
#> pss[1,1] 0.0340   1e-04 0.0076 0.0192 0.0489 18735 0.9999
#> pss[1,2] 0.0240   1e-04 0.0082 0.0077 0.0396 18099 0.9999
#> pss[2,1] 0.0595   1e-04 0.0079 0.0441 0.0749 13674 1.0000
#> pss[2,2] 0.0495   1e-04 0.0083 0.0327 0.0653 15479 1.0001
#> pss[3,1] 0.0284   1e-04 0.0084 0.0120 0.0450 17367 0.9999
#> pss[3,2] 0.0184   1e-04 0.0091 0.0003 0.0357 16956 0.9999
#>            mean se_mean     sd   2.5%  97.5% n_eff   Rhat
#> jnd[1,1] 0.1981   1e-04 0.0084 0.1821 0.2150 20727 0.9999
#> jnd[1,2] 0.1589   1e-04 0.0075 0.1446 0.1742 20006 1.0001
#> jnd[2,1] 0.1882   1e-04 0.0077 0.1736 0.2038 18843 1.0001
#> jnd[2,2] 0.1510   1e-04 0.0075 0.1370 0.1662 19879 1.0001
#> jnd[3,1] 0.2475   1e-04 0.0101 0.2282 0.2679 18233 1.0000
#> jnd[3,2] 0.1985   1e-04 0.0097 0.1804 0.2180 19050 1.0000
```


### Posterior retrodictive checks {#iter2-post-retro}




<img src="051-bayesian-workflow_files/figure-html/ch051-Sleepy Roadrunner-1.png" width="70%" style="display: block; margin: auto;" />


The retrodictive data are matching well with the observed data, which means that we are getting closer to a model that we can use for inferences.





<img src="051-bayesian-workflow_files/figure-html/ch051-Furious Jazz-1.png" width="70%" style="display: block; margin: auto;" />


It's difficult to determine from this graph if there any difference between the age groups. Looking at the density plot of the PSS and JND across the different conditions paints a much clearer image.


<img src="051-bayesian-workflow_files/figure-html/ch051-Discarded Torpedo-1.png" width="70%" style="display: block; margin: auto;" />


Using ocular analysis^[Often referred to in the non-sciences as eyeballs] we can see that recalibration has a significant affect on the just noticeable difference for each age group; specifically recalibration heightens temporal sensitivity and thus reduces the just noticeable difference. Comparing between age groups, the young and middle age groups are very similar in both the pre- and post-adaptation trials, and temporal sensitivity is lower in the older age group.

The point of subjective simultaneity between trials is not as well separated, but the model still consistently estimates that the subjects will perceive simultaneity at a value closer to zero post-adaptation.
