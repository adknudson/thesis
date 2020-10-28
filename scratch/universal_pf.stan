data {
  int N;        // Number of observations
  int N_G;      // Number of age groups
  int N_T;      // Number of treatments
  int N_S;      // Number of subjects
  int n[N];     // Number of Bernoulli trials
  int k[N];     // Number of "positive" responses
  vector[N] x;  // SOA values
  int G[N];     // Age group index variable
  int trt[N];   // Treatment index variable
  int S[N];     // Subject index variable
}
parameters {
  real a_raw;
  real<lower=machine_precision(),upper=pi()/2> aG_unif;
  real<lower=machine_precision(),upper=pi()/2> aT_unif;
  real<lower=machine_precision(),upper=pi()/2> aGT_unif;
  real<lower=machine_precision(),upper=pi()/2> aS_unif;
  vector[N_G] aG_raw;
  vector[N_T] aT_raw;
  matrix[N_G, N_T] aGT_raw;
  vector[N_S] aS_raw;

  real b_raw;
  real<lower=machine_precision(),upper=pi()/2> bG_unif;
  real<lower=machine_precision(),upper=pi()/2> bT_unif;
  real<lower=machine_precision(),upper=pi()/2> bGT_unif;
  real<lower=machine_precision(),upper=pi()/2> bS_unif;
  vector[N_G] bG_raw;
  vector[N_T] bT_raw;
  matrix[N_G, N_T] bGT_raw;
  vector[N_S] bS_raw;

  vector[N_G] lG;
}
transformed parameters {
  real a;
  vector[N_G] aG;
  vector[N_T] aT;
  real aGT[N_G, N_T];
  vector[N_S] aS;
  real sd_aG;
  real sd_aT;
  real sd_aGT;
  real sd_aS;

  real b;
  vector[N_G] bG;
  vector[N_T] bT;
  real bGT[N_G, N_T];
  vector[N_S] bS;
  real sd_bG;
  real sd_bT;
  real sd_bGT;
  real sd_bS;

  // Z * sigma ~ N(0, sigma^2)
  a = a_raw * 0.05;

  // mu + tau * tan(U) ~ cauchy(mu, tau)
  sd_aG = 0.01 * tan(aG_unif);
  sd_aT = 0.01 * tan(aT_unif);
  sd_aGT = 0.05 * tan(aGT_unif);
  sd_aS = 0.05 * tan(aS_unif);

  aG = aG_raw * sd_aG;
  aT = aT_raw * sd_aT;
  aS = aS_raw * sd_aS;

  // mu + Z * sigma ~ N(mu, sigma^2)
  b = 3.0 + b_raw * 1.5;

  // mu + tau * tan(U) ~ cauchy(mu, tau)
  sd_bG = 0.5 * tan(bG_unif);
  sd_bT = 0.5 * tan(bT_unif);
  sd_bGT = 0.5 * tan(bGT_unif);
  sd_bS = 0.5 * tan(bS_unif);

  bG = bG_raw * sd_bG;
  bT = bT_raw * sd_bT;
  bS = bS_raw * sd_bS;

  for (i in 1:N_G) {
    for (j in 1:N_T) {
      aGT[i, j] = aGT_raw[i, j] * sd_aGT;
      bGT[i, j] = bGT_raw[i, j] * sd_bGT;
    }
  }
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

  lG ~ beta(4, 96);

  to_vector(aGT_raw) ~ std_normal();
  to_vector(bGT_raw) ~ std_normal();

  // Compute probability values
  for (i in 1:N) {
    real mu_b = exp(b + bG[G[i]] + bT[trt[i]] + bGT[G[i], trt[i]] + bS[S[i]]);
    real mu_a = a + aG[G[i]] + aT[trt[i]] + aGT[G[i], trt[i]] + aS[S[i]];
    theta[i] = lG[G[i]] + (1 - 2*lG[G[i]]) * inv_logit(mu_b * (x[i] - mu_a));
  }

  k ~ binomial(n, theta);
}
generated quantities {
  vector[N] log_lik;

  for (i in 1:N) {
    real beta = exp(b + bG[G[i]] + bT[trt[i]] + bGT[G[i], trt[i]] + bS[S[i]]);
    real alpha = a + aG[G[i]] + aT[trt[i]] + aGT[G[i], trt[i]] + aS[S[i]];
    real lambda = lG[G[i]];

    real p = lambda + (1 - 2*lambda) * inv_logit(beta * (x[i] - alpha));

    log_lik[i] = binomial_lpmf(k[i] | n[i], p);
  }
}
