library(FangPsychometric)
library(dplyr)
library(ggplot2)
library(patchwork)
library(rstan)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

logit <- function(p) qlogis(p)
inv_logit <- function(x) plogis(x)
logistic <- function(x) inv_logit(x)
fn <- function(x, a, b) logistic(b * (x - a))

obs_dat <- function(data) {
  dat <- data %>%
    filter(trial %in% c("pre", "post1")) %>%
    mutate(x = soa / 1000,
           rid = factor(rid),
           sid = factor(sid),
           trial = factor(trial)) %>%
    as.list()
  dat$N <- length(dat$x)
  dat$N_G <- length(levels(dat$age_group))
  dat$N_S <- length(levels(dat$sid))
  dat$N_T <- length(levels(dat$trial))
  dat
}

plot_pf <- function(n, post, age_group, trt, xlim = c(-0.5, 0.5)) {
  n_smp <- 100
  idx <- sample(1:length(post$a), n_smp, replace = TRUE)

  alpha <- with(post, a[idx] + aG[idx, age_group] + aT[idx, trt])
  beta <- with(post, exp(b[idx] + bG[idx, age_group] + bT[idx, trt]))

  p <- tibble(x = xlim, y = c(0, 1)) %>%
    ggplot(aes(x, y)) +
    scale_x_continuous(breaks = seq(xlim[1], xlim[2], 0.1)) +
    scale_y_continuous(breaks = c(0, 0.5, 1))
  for (i in 1:n_smp) {
    p <- p + geom_line(stat = "function", fun = fn,
                       args = list(a = alpha[i],
                                   b = beta[i]),
                       alpha = 0.05)
  }
  p
}

stan_summary <- function(object, ...) {
  round(summary(object, ...)$summary, 4)[,c(1,3,4,8,9,10)]
}

universal_pf <- "functions {
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

  vector[N_G] lG;
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

  lG ~ beta(4, 96);

  for (i in 1:N) {
    real mu_b = exp(b + bG[G[i]] + bT[trt[i]] + bS[S[i]]);
    real mu_a = a + aG[G[i]] + aT[trt[i]] + aS[S[i]];
    theta[i] = lG[G[i]] + (1 - 2*lG[G[i]]) * inv_logit(mu_b * (x[i] - mu_a));
  }

  k ~ binomial(n, theta);
}
generated quantities {
  int y_post_pred[N];
  matrix[N_G, N_T] pss;
  matrix[N_G, N_T] jnd;

  for (i in 1:N_G) {
    for (j in 1:N_T) {
      real mu_b = exp(b + bG[i] + bT[j]);
      real mu_a = a + aG[i] + aT[j];
      real mu_l = lG[i];
      pss[i, j] = inv_Psi(0.50, mu_a, mu_b, mu_l);
      jnd[i, j] = inv_Psi(0.84, mu_a, mu_b, mu_l) - pss[i, j];
    }
  }

  for (i in 1:N) {
    real mu_b = exp(b + bG[G[i]] + bT[trt[i]] + bS[S[i]]);
    real mu_a = a + aG[G[i]] + aT[trt[i]] + aS[S[i]];
    real mu_l = lG[G[i]];
    real theta = mu_l + (1 - 2*mu_l) * inv_logit(mu_b * (x[i] - mu_a));
    y_post_pred[i] = binomial_rng(n[i], theta);
  }
}"


dat <- obs_dat(audiovisual_binomial)
stan_dat <- with(dat, list(
  N = N,
  N_G = N_G,
  N_T = N_T,
  N_S = N_S,
  x = x,
  k = k,
  n = n,
  G = as.integer(age_group),
  trt = as.integer(trial),
  S = as.integer(sid)
))


keep_pars <- c(
  "a", "b", "lG",
  "pss", "jnd",
  "aG", "bG",
  "aT", "bT",
  "aS", "bS",
  "sd_aG", "sd_bG",
  "sd_aT", "sd_bT",
  "sd_aS", "sd_bS",
  "y_post_pred"
)

n_chains <- 4L

init <- replicate(n_chains, list(
  a_raw = rnorm(1),
  aG_raw = rnorm(dat$N_G, 0, 0.5),
  aT_raw = rnorm(dat$N_T, 0, 0.5),
  aS_raw = rnorm(dat$N_S, 0, 0.5),
  aG_unif = runif(1, 0, pi/4),
  aT_unif = runif(1, 0, pi/4),
  aS_unif = runif(1, 0, pi/4),
  b_raw = rnorm(1),
  bG_raw = rnorm(dat$N_G, 0, 0.5),
  bT_raw = rnorm(dat$N_T, 0, 0.5),
  bS_raw = rnorm(dat$N_S, 0, 0.5),
  bG_unif = runif(1, 0, pi/4),
  bT_unif = runif(1, 0, pi/4),
  bS_unif = runif(1, 0, pi/4),
  lG = runif(dat$N_G, 0, 0.05)),
simplify = FALSE)

m <- stan_model(model_code = universal_pf)

f <- sampling(
  object = m,
  data = stan_dat,
  chains = n_chains,
  cores = n_chains,
  iter = 7000,
  warmup = 2000,
  refresh = 500,
  init = init,
  control = list(adapt_delta = 0.95),
  pars = keep_pars
)

stan_summary(f, c("a", "aG", "aT"))
stan_summary(f, c("b", "bG", "bT"))
stan_summary(f, c("pss"))
stan_summary(f, c("jnd"))
stan_summary(f, pars = paste0("sd_a", c("G", "T", "S")))
stan_summary(f, pars = paste0("sd_b", c("G", "T", "S")))
stan_summary(f, "aS")
stan_summary(f, "bS")

post <- extract(f)
post_pred <- t(apply(post$y_post_pred, 2, quantile,
                     probs = c(1.5, 5.5, 50, 94.5, 98.5) / 100))

idx <- sample(1:nrow(post$y_post_pred), 1)

m_pred <- cbind(post_pred,
                post_mean = colMeans(post$y_post_pred),
                post_rand = post$y_post_pred[idx,]) %>%
  sweep(1, dat$n, FUN = "/") %>%
  bind_cols(stan_dat, trial = dat$trial, age_group = dat$age_group) %>%
  select(-N) %>%
  mutate(p = k / n)

m_pred %>%
  ggplot(aes(x, p)) +
  scale_x_continuous(breaks = seq(-0.5, 0.5, 0.25)) +
  scale_y_continuous(breaks = c(0, 0.5, 1)) +
  geom_jitter(width = 0.0125, height = 0.01,
              col = rgb(123, 28, 212, maxColorValue = 255)) +
  geom_jitter(aes(y = post_rand),
              col = rgb(28, 214, 68, 120, maxColorValue = 255),
              width = 0.0125, height = 0.01) +
  facet_grid(age_group ~ trial)


ypre <- plot_pf(100, post, 1, 1, c(-0.3, 0.3))
ypos <- plot_pf(100, post, 1, 2, c(-0.3, 0.3))
mpre <- plot_pf(100, post, 2, 1, c(-0.3, 0.3))
mpos <- plot_pf(100, post, 2, 2, c(-0.3, 0.3))
opre <- plot_pf(100, post, 3, 1, c(-0.3, 0.3))
opos <- plot_pf(100, post, 3, 2, c(-0.3, 0.3))
(ypre + ypos) / (mpre + mpos) / (opre + opos)

age_trt <- expand.grid(a = 1:3, t = 1:2)

dat_pssjnd <- lapply(1:nrow(age_trt), function(i) {
  a <- age_trt$a[i]
  t <- age_trt$t[i]
  tibble(PSS = post$pss[,a,t],
         JND = post$jnd[,a,t],
         a = a,
         t = t)
}) %>% do.call(what = bind_rows) %>%
  mutate(a = factor(a, levels = 1:3, labels = levels(av_dat$age_group)),
         t = factor(t, levels = 1:2, labels = levels(av_dat$trial))) %>%
  rename(age_group = a, trial = t) %>%
  tidyr::pivot_longer(c("PSS", "JND"), names_to = "Name", values_to = "Seconds")

dat_pssjnd %>%
  ggplot(aes(Seconds, fill = trial)) +
  geom_density() +
  scale_fill_manual("trial",
                    values = c(rgb(29/255, 149/255, 219/255, 0.5),
                               rgb(143/255, 19/255, 19/255, 0.5))) +
  facet_grid(age_group ~ Name, scales = "free_x")
