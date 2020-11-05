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
  real<lower=machine_precision(),upper=pi()/2> aGT_unif;
  real<lower=machine_precision(),upper=pi()/2> aS_unif;
  matrix[N_G, N_T] aGT_raw;
  vector[N_S] aS_raw;

  real b_raw;
  real<lower=machine_precision(),upper=pi()/2> bGT_unif;
  real<lower=machine_precision(),upper=pi()/2> bS_unif;
  matrix[N_G, N_T] bGT_raw;
  vector[N_S] bS_raw;

  vector[N_G] lG;
}
transformed parameters {
  real a;
  matrix[N_G, N_T] aGT;
  vector[N_S] aS;
  real sd_aGT;
  real sd_aS;

  real b;
  matrix[N_G, N_T] bGT;
  vector[N_S] bS;
  real sd_bGT;
  real sd_bS;

  a = a_raw * 0.06;
  sd_aGT = tan(aGT_unif);
  sd_aS  = tan(aS_unif);
  aS = aS_raw * sd_aS;

  b = 3.0 + b_raw;
  sd_bGT = 2 * tan(bGT_unif);
  sd_bS  = 2 * tan(bS_unif);
  bS = bS_raw * sd_bS;

  for (i in 1:N_G) {
    for (j in 1:N_T) {
      aGT[i, j] = aGT_raw[i, j] * sd_aGT;
      bGT[i, j] = bGT_raw[i, j] * sd_bGT;
    }
  }
}
model {
  vector[N] p;

  a_raw ~ std_normal();
  b_raw ~ std_normal();
  lG ~ beta(4, 96);

  aS_raw ~ std_normal();
  bS_raw ~ std_normal();
  to_vector(aGT_raw) ~ std_normal();
  to_vector(bGT_raw) ~ std_normal();

  for (i in 1:N) {
    real alpha = a + aGT[G[i], trt[i]] + aS[S[i]];
    real beta = b + bGT[G[i], trt[i]] + bS[S[i]];
    real lambda = lG[G[i]];
    p[i] = lambda + (1 - 2*lambda) * inv_logit(exp(beta) * (x[i] - alpha));
  }

  k ~ binomial(n, p);
}
generated quantities {
  vector[N] log_lik;
  vector[N] k_pred;

  for (i in 1:N) {
    real alpha = a + aGT[G[i], trt[i]] + aS[S[i]];
    real beta  = b + bGT[G[i], trt[i]] + bS[S[i]];
    real lambda = lG[G[i]];

    real p = lambda + (1 - 2*lambda) * inv_logit(exp(beta) * (x[i] - alpha));

    log_lik[i] = binomial_lpmf(k[i] | n[i], p);
    k_pred[i]  = binomial_rng(n[i], p);
  }
}
