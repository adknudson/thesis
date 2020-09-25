// mu_a = a + aG + aT
// mu_b = b + bG + bT
// logit(pi) = exp(mu_b) * (x - mu_a)
data {
  int N;
  int N_G;
  int N_T;
  int n[N];
  int k[N];
  vector[N] x;
  int G[N];
  int trt[N];
}
parameters {
  real a;
  real<lower=0> sd_aG;
  real<lower=0> sd_aT;
  real aG[N_G];
  real aT[N_T];

  real b;
  real<lower=0> sd_bG;
  real<lower=0> sd_bT;
  real bG[N_G];
  real bT[N_T];
}
model {
  vector[N] theta;

  a  ~ normal(0, 0.05);
  aG ~ normal(0, sd_aG);
  aT ~ normal(0, sd_aT);
  sd_aG ~ cauchy(2.5, 2.5);
  sd_aT ~ cauchy(2.5, 2.5);

  b  ~ normal(2.5, 1.0);
  bG ~ normal(0, sd_bG);
  bT ~ normal(0, sd_bT);
  sd_bG ~ cauchy(2.5, 2.5);
  sd_bT ~ cauchy(2.5, 2.5);

  for (i in 1:N) {
    real mu_b = exp(b + bG[G[i]] + bT[trt[i]]);
    real mu_a = a + aG[G[i]] + aT[trt[i]];
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
    real mu_b = exp(b + bG[G[i]] + bT[trt[i]]);
    real mu_a = a + aG[G[i]] + aT[trt[i]];
    y_post_pred[i] = binomial_rng(n[i], inv_logit(mu_b * (x[i] - mu_a)));
  }
}
