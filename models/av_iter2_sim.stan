// mu_a = a + aG + aT
// mu_b = b + bG + bT
// logit(pi) = exp(mu_b) * (x - mu_a)
functions {
  real half_cauchy_rng(real sigma) {
    real u = uniform_rng(0.5, 0.99);
    real y = sigma * tan(pi() * (u - 0.5));
    return y;
  }
}
data {
  int N;
  int N_G;
  int N_T;
  int n[N];
  vector[N] x;
  int G[N];
  int trt[N];
}
generated quantities {
  vector[N] theta;
  int y_sim[N];
  matrix[N_G, N_T] pss;
  matrix[N_G, N_T] jnd;

  real alpha = normal_rng(0, 0.05);
  real<lower=0> sigma_aG = half_cauchy_rng(0.05);
  real<lower=0> sigma_aT = half_cauchy_rng(0.05);
  real alpha_G[N_G] = normal_rng(rep_vector(0, N_G), sigma_aG);
  real alpha_T[N_T] = normal_rng(rep_vector(0, N_T), sigma_aT);

  real beta = normal_rng(2.5, 1.0);
  real<lower=0> sigma_bG = half_cauchy_rng(0.5);
  real<lower=0> sigma_bT = half_cauchy_rng(0.5);
  real beta_G[N_G] = normal_rng(rep_vector(0, N_G), sigma_bG);
  real beta_T[N_T] = normal_rng(rep_vector(0, N_T), sigma_bT);

  for (i in 1:N) {
    real gamma = exp(beta + beta_G[G[i]] + beta_T[trt[i]]);
    real delta = alpha + alpha_G[G[i]] + alpha_T[trt[i]];

    theta[i] = inv_logit( gamma * (x[i] - delta) );
    y_sim[i] = binomial_rng(n[i], theta[i]);
  }

  for (i in 1:N_G) {
    for (j in 1:N_T) {
      real mu_b = exp(beta + beta_G[i] + beta_T[j]);
      real mu_a = alpha + alpha_G[i] + alpha_T[j];
      pss[i, j] = mu_a;
      jnd[i, j] = logit(0.84) / mu_b;
    }
  }
}
