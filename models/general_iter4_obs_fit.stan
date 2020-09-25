// mu_a = a + aG + aT + aS
// mu_b = b + bG + bT + bS
// mu_l = l + lG
// pi = mu_l + (1 - 2*mu_l) * inv_logit( exp(mu_b) * (x - mu_a) )
functions {
  real inv_Psi(real p, real a, real b, real l) {
    return logit((p - l) / (1 - 2 * l)) / b + a;
  }
}
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
  real a;
  real<lower=machine_precision()> sd_aG;
  real<lower=machine_precision()> sd_aT;
  real<lower=machine_precision()> sd_aS;
  real aG[N_G];
  real aT[N_T];
  real aS[N_S];

  real b;
  real<lower=machine_precision()> sd_bG;
  real<lower=machine_precision()> sd_bT;
  real<lower=machine_precision()> sd_bS;
  real bG[N_G];
  real bT[N_T];
  real bS[N_S];

  real<lower=0,upper=1> l;
  real lG[N_G];
}
model {
  vector[N] theta;

  a  ~ normal(0, 0.05);
  aG ~ normal(0, sd_aG);
  aT ~ normal(0, sd_aT);
  aS ~ normal(0, sd_aS);
  sd_aG ~ cauchy(2.5, 2.5);
  sd_aT ~ cauchy(2.5, 2.5);
  sd_aS ~ cauchy(2.5, 2.5);

  b  ~ normal(2.5, 1.0);
  bG ~ normal(0, sd_bG);
  bT ~ normal(0, sd_bT);
  bS ~ normal(0, sd_bS);
  sd_bG ~ cauchy(2.5, 2.5);
  sd_bT ~ cauchy(2.5, 2.5);
  sd_bS ~ cauchy(2.5, 2.5);

  l ~ beta(4, 96);
  lG ~ normal(0, 0.005);

  for (i in 1:N) {
    real mu_l = l + lG[G[i]];
    real mu_a = a + aG[G[i]] + aT[trt[i]] + aS[S[i]];
    real mu_b = b + bG[G[i]] + bT[trt[i]] + bS[S[i]];
    theta[i] = mu_l + (1 - 2*mu_l) * inv_logit(exp(mu_b) * (x[i] - mu_a));
  }

  k ~ binomial(n, theta);
}
generated quantities {
  int y_post_pred[N];
  matrix[N_G, N_T] pss;
  matrix[N_G, N_T] jnd;

  // Estimate Group/Treatment PSS and JND
  for (i in 1:N_G) {
    for (j in 1:N_T) {
      real mu_l = l + lG[G[i]];
      real mu_a = a + aG[i] + aT[j];
      real mu_b = b + bG[i] + bT[j];
      pss[i, j] = inv_Psi(0.50, mu_a, exp(mu_b), mu_l);
      jnd[i, j] = inv_Psi(0.84, mu_a, exp(mu_b), mu_l) - pss[i, j];
    }
  }

  // Simulate the posterior predictions
  for (i in 1:N) {
    real mu_l = l + lG[G[i]];
    real mu_a = a + aG[G[i]] + aT[trt[i]] + aS[S[i]];
    real mu_b = b + bG[G[i]] + bT[trt[i]] + bS[S[i]];
    real theta = mu_l + (1 - 2*mu_l) * inv_logit(mu_b * (x[i] - mu_a));
    y_post_pred[i] = binomial_rng(n[i], theta);
  }
}
