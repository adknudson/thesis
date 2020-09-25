// logit(pi) = b * (x - a)
data {
  int<lower=0> N;
  int n[N];
  vector[N] x;
}
generated quantities {
  real alpha = normal_rng(0, 0.05);
  real beta = lognormal_rng(3.96, 1.2);
  vector[N] theta = inv_logit( beta * (x - alpha) );
  int y_sim[N] = binomial_rng(n, theta);
  real pss = alpha;
  real jnd = logit(0.84) / beta;
}
