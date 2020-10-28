


## Iteration 3 (The one for Me) {#iter3}

In this iteration of the model building process, we are going to add the individual subjects into the multilevel model, and because this is a simple addition, we are going to skip the prior predictive simulations.

### Model Development {#iter3-model-dev}



```stan
data {
  int N;
  int N_G;
  int N_T;
  int N_S;
  int n[N];
  int k[N];
  vector[N] x;
  int G[N];
  int trt[N];
  int S[N];
}
parameters {
  real a_raw;
  real<lower=machine_precision(),upper=pi()/2> aG_unif;
  real<lower=machine_precision(),upper=pi()/2> aT_unif;
  real<lower=machine_precision(),upper=pi()/2> aS_unif;
  vector[N_G] aG_raw;
  vector[N_T] aT_raw;
  vector[N_S] aS_raw;

  real b_raw;
  real<lower=machine_precision(),upper=pi()/2> bG_unif;
  real<lower=machine_precision(),upper=pi()/2> bT_unif;
  real<lower=machine_precision(),upper=pi()/2> bS_unif;
  vector[N_G] bG_raw;
  vector[N_T] bT_raw;
  vector[N_S] bS_raw;
}
transformed parameters {
  real a;
  vector[N_G] aG;
  vector[N_T] aT;
  vector[N_S] aS;
  real sd_aG;
  real sd_aT;
  real sd_aS;
  
  real b;
  vector[N_G] bG;
  vector[N_T] bT;
  vector[N_S] bS;
  real sd_bG;
  real sd_bT;
  real sd_bS;
  
  // Z * sigma ~ N(0, sigma^2)
  a = a_raw * 0.05;
  
  // mu + tau * tan(U) ~ cauchy(mu, tau)
  sd_aG = 0.01 * tan(aG_unif);
  sd_aT = 0.01 * tan(aT_unif);
  sd_aS = 0.05 * tan(aS_unif);
  
  aG = aG_raw * sd_aG;
  aT = aT_raw * sd_aT;
  aS = aS_raw * sd_aS;
  
  // mu + Z * sigma ~ N(mu, sigma^2)
  b = 3.0 + b_raw * 1.5;
  
  // mu + tau * tan(U) ~ cauchy(mu, tau)
  sd_bG = 0.5 * tan(bG_unif);
  sd_bT = 0.5 * tan(bT_unif);
  sd_bS = 0.5 * tan(bS_unif);
  
  bG = bG_raw * sd_bG;
  bT = bT_raw * sd_bT;
  bS = bS_raw * sd_bS;
}
model {
  vector[N] theta;

  a_raw ~ std_normal();
  aG_raw ~ std_normal();
  aT_raw ~ std_normal();
  aS_raw ~ std_normal();

  b_raw ~ std_normal();
  bG_raw ~ std_normal();
  bT_raw ~ std_normal();
  bS_raw ~ std_normal();

  for (i in 1:N) {
    real mu_b = exp(b + bG[G[i]] + bT[trt[i]] + bS[S[i]]);
    real mu_a = a + aG[G[i]] + aT[trt[i]] + aS[S[i]];
    theta[i] = mu_b * (x[i] - mu_a);
  }

  k ~ binomial_logit(n, theta);
}
generated quantities {
  int y_post_pred[N];
  matrix[N_G, N_T] pss;
  matrix[N_G, N_T] jnd;

  for (i in 1:N_G) {
    for (j in 1:N_T) {
      real mu_b = exp(b + bG[i] + bT[j]);
      real mu_a = a + aG[i] + aT[j];
      pss[i, j] = mu_a;
      jnd[i, j] = logit(0.84) / mu_b;
    }
  }

  for (i in 1:N) {
    real mu_b = exp(b + bG[G[i]] + bT[trt[i]] + bS[S[i]]);
    real mu_a = a + aG[G[i]] + aT[trt[i]] + aS[S[i]];
    real theta = inv_logit(mu_b * (x[i] - mu_a));
    y_post_pred[i] = binomial_rng(n[i], theta);
  }
}
```




### Diagnose posterior fit {#iter3-diagnose-post}


```
#> 
#> Divergences:
#> 325 of 20000 iterations ended with a divergence (1.625%).
#> Try increasing 'adapt_delta' to remove the divergences.
#> 
#> Tree depth:
#> 0 of 20000 iterations saturated the maximum tree depth of 10.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```



```
#>          mean se_mean     sd    2.5%  97.5% n_eff  Rhat
#> a      0.0321   3e-04 0.0171 -0.0047 0.0634  4551 1.001
#> aG[1] -0.0010   1e-04 0.0121 -0.0281 0.0248  8336 1.000
#> aG[2]  0.0067   2e-04 0.0141 -0.0118 0.0451  4854 1.000
#> aG[3] -0.0028   1e-04 0.0122 -0.0320 0.0193  7753 1.000
#> aT[1]  0.0055   1e-04 0.0113 -0.0119 0.0329  7144 1.000
#> aT[2] -0.0028   1e-04 0.0109 -0.0239 0.0203  7570 1.000
#>          mean se_mean     sd    2.5%  97.5% n_eff  Rhat
#> b      2.3742  0.0072 0.3784  1.6341 3.2295  2776 1.002
#> bG[1]  0.0796  0.0025 0.2153 -0.3495 0.5552  7624 1.001
#> bG[2]  0.0592  0.0023 0.2149 -0.3778 0.5171  8376 1.000
#> bG[3] -0.1689  0.0030 0.2253 -0.6833 0.2064  5723 1.001
#> bT[1] -0.1493  0.0081 0.3223 -0.9116 0.5265  1582 1.002
#> bT[2]  0.1007  0.0084 0.3221 -0.6519 0.8034  1482 1.002
#>            mean se_mean     sd    2.5%  97.5% n_eff  Rhat
#> pss[1,1] 0.0365   2e-04 0.0140  0.0075 0.0632  4147 1.002
#> pss[1,2] 0.0283   2e-04 0.0142 -0.0012 0.0555  4085 1.002
#> pss[2,1] 0.0443   3e-04 0.0148  0.0168 0.0766  3427 1.001
#> pss[2,2] 0.0360   3e-04 0.0150  0.0085 0.0685  3491 1.001
#> pss[3,1] 0.0348   2e-04 0.0144  0.0042 0.0614  4209 1.001
#> pss[3,2] 0.0265   2e-04 0.0146 -0.0048 0.0535  4130 1.001
#>            mean se_mean     sd   2.5%  97.5% n_eff  Rhat
#> jnd[1,1] 0.1666   3e-04 0.0194 0.1303 0.2059  4725 1.001
#> jnd[1,2] 0.1298   2e-04 0.0155 0.1009 0.1618  4903 1.001
#> jnd[2,1] 0.1700   3e-04 0.0192 0.1342 0.2093  4633 1.001
#> jnd[2,2] 0.1324   2e-04 0.0154 0.1040 0.1643  4939 1.002
#> jnd[3,1] 0.2139   5e-04 0.0274 0.1673 0.2726  3374 1.001
#> jnd[3,2] 0.1667   4e-04 0.0221 0.1292 0.2144  3393 1.001
```



```
#>         mean se_mean     sd   2.5%  97.5% n_eff  Rhat
#> sd_aG 0.0114   2e-04 0.0132 0.0003 0.0451  5168 1.000
#> sd_aT 0.0117   1e-04 0.0138 0.0005 0.0465  9787 1.000
#> sd_aS 0.0689   2e-04 0.0090 0.0535 0.0880  2106 1.002
#>         mean se_mean     sd   2.5%  97.5%  n_eff  Rhat
#> sd_bG 0.2804  0.0032 0.2453 0.0168 0.9295 5707.0 1.001
#> sd_bT 0.4261  0.0266 0.4193 0.0837 1.5445  247.9 1.013
#> sd_bS 0.4403  0.0008 0.0577 0.3405 0.5657 5653.6 1.000
```


The number of effective samples and the R-hat indicate that there is no problem with the posterior samples.

### Posterior retrodictive checks {#iter3-post-retro}




<img src="042-bayesian-workflow_files/figure-html/ch042-Leather Lucky-1.png" width="70%" style="display: block; margin: auto;" />





<img src="042-bayesian-workflow_files/figure-html/ch042-Forsaken Purple Moose-1.png" width="70%" style="display: block; margin: auto;" />


It's difficult to determine from this graph if there any difference between the age groups. Looking at the density plot of the PSS and JND across the different conditions paints a much clearer image.


<img src="042-bayesian-workflow_files/figure-html/ch042-Severe Lion-1.png" width="70%" style="display: block; margin: auto;" />

Okay what gives? The differences within and between age groups is not a separated as it was in the previous iteration. This is due to the fact that the previous model averaged over the variation at the subject level. We'll consider the task of making predictions at the different levels in the hierarchical model.

