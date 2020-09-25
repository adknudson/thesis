// logit(pi) = b * (x - a)
data {
  int N;
  int n[N];
  int k[N];
  vector[N] x;
}
parameters {
  real alpha;
  real<lower=0> beta;
}
model {
  vector[N] p = beta * (x - alpha);
  alpha ~ normal(0, 0.05);
  beta ~ lognormal(3.96, 1.2);
  k ~ binomial_logit(n, p);
}
generated quantities {
  real pss = alpha;
  real jnd = logit(0.84) / beta;
}
