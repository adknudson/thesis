data {
  int N;        // Number of observations
  int N_T;      // Number of treatments
  int N_S;      // Number of subjects
  int n[N];     // Number of Bernoulli trials
  int k[N];     // Number of "positive" responses
  vector[N] x;  // SOA values
  int trt[N];   // Treatment index variable
  int S[N];     // Subject index variable
}
parameters {
  real a_raw;
  real<lower=machine_precision(),upper=pi()/2> aT_unif;
  real<lower=machine_precision(),upper=pi()/2> aS_unif;
  vector[N_T] aT_raw;
  vector[N_S] aS_raw;

  real b_raw;
  real<lower=machine_precision(),upper=pi()/2> bT_unif;
  real<lower=machine_precision(),upper=pi()/2> bS_unif;
  vector[N_T] bT_raw;
  vector[N_S] bS_raw;

  real lambda;
}
transformed parameters {
  real a;
  vector[N_T] aT;
  vector[N_S] aS;
  real sd_aT;
  real sd_aS;

  real b;
  vector[N_T] bT;
  vector[N_S] bS;
  real sd_bT;
  real sd_bS;

  a = a_raw * 0.06;
  sd_aT = tan(aT_unif);
  sd_aS = tan(aS_unif);
  aT = aT_raw * sd_aT;
  aS = aS_raw * sd_aS;

  b = 3.0 + b_raw;
  sd_bT = 2 * tan(bT_unif);
  sd_bS = 2 * tan(bS_unif);
  bT = bT_raw * sd_bT;
  bS = bS_raw * sd_bS;
}
model {
  vector[N] p;

  a_raw ~ std_normal();
  b_raw ~ std_normal();

  lambda ~ beta(4, 96);

  aS_raw ~ std_normal();
  bS_raw ~ std_normal();
  aT_raw ~ std_normal();
  bT_raw ~ std_normal();

  for (i in 1:N) {
    real alpha = a + aT[trt[i]] + aS[S[i]];
    real beta =  b + bT[trt[i]] + bS[S[i]];
    p[i] = lambda + (1 - 2*lambda) * inv_logit(exp(beta) * (x[i] - alpha));
  }

  k ~ binomial(n, p);
}
generated quantities {
  vector[N] log_lik;
  vector[N] k_pred;

  for (i in 1:N) {
    real alpha = a + aT[trt[i]] + aS[S[i]];
    real beta  = b + bT[trt[i]] + bS[S[i]];

    real p = lambda + (1 - 2*lambda) * inv_logit(exp(beta) * (x[i] - alpha));

    log_lik[i] = binomial_lpmf(k[i] | n[i], p);
    k_pred[i]  = binomial_rng(n[i], p);
  }
}
