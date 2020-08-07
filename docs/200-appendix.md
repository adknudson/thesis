

# (APPENDIX) Appendix {-}

# Supplementary Code

One model, Three Implementations. There are a few ways to specify a hierarchical model in R. Below I describe three common frameworks that require varying levels of mathematical and programmatic competence. Frameworks with lower barriers for entry are great for researchers in many fields, but they lack fine control over the parameters in a model. As the framework complexity increases, so too does the ability to generate complex models that are typically not possible. 

Novice
\setstretch{1.0}

```r
library(rstanarm)
stan_glmer(cbind(k, n-k) ~ 1 + x + (1 + x | G1) + (1 + x | G2), 
           family = binomial(link = "logit"),
           data = dat)
```
\setstretch{2.0}

Intermediate

\setstretch{1.0}

```r
library(rethinking)
ulam(alist(
  k ~ binomial(n, pi)
  logit(pi) <- (a + aG1[G1] + aG2[G2]) + (b + bG1[G1] + bG2[G2]) * x,
  
  a ~ normal(0, 10),
  aG1[G1] ~ normal(0, sd_aG1),
  aG2[G2] ~ normal(0, sd_aG2),
  c(sd_aG1, sd_aG2) ~ half_cauchy(0, 10),

  b ~ normal(0, 10),
  bG1[G1] ~ normal(0, sd_bG1),
  bG2[G2] ~ normal(0, sd_bG2),
  c(sd_bG1, sd_bG2) ~ half_cauchy(0, 10)
), data = dat, log_lik = TRUE)
```
\setstretch{2.0}

Advanced

\setstretch{1.0}
```
data{
    int<lower=0> N;
    int<lower=0> N_G1;
    int<lower=0> N_G2;
    int n[N];
    int k[N];
    int G1[N];
    int G2[N];
    int trt[N];
    vector[N] x;
}
parameters{
    real a;
    vector[N_G1] aG1;
    vector[N_G2] aG2;
    real b;
    vector[N_G1] bG1;
    vector[N_G2] bG2;
    real<lower=0> sd_aG1;
    real<lower=0> sd_aG2;
    real<lower=0> sd_bG1;
    real<lower=0> sd_bG2;
}
model{
    vector[N] p;
    a   ~ normal(0, 10);
    aG1 ~ normal(0, sd_aG1);
    aG2 ~ normal(0, sd_aG2);
    b   ~ normal(0, 10);
    bG1 ~ normal(0, sd_bG1);
    bG2 ~ normal(0, sd_bG2);
    sd_aG1 ~ cauchy(0, 10);
    sd_aG2 ~ cauchy(0, 10);
    sd_bG1 ~ cauchy(0, 10);
    sd_bG2 ~ cauchy(0, 10);
    for ( i in 1:N ) {
        p[i] = (a + aG1[G1[i]] + aG2[G2[i]]) + (b + bG1[G1[i]] + bG2[G2[i]]) * x[i];
    }
    k ~ binomial_logit(n , p);
}
```
\setstretch{2.0}
