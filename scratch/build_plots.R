library(FangPsychometric)
library(rstan)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

source("R/stan_utils.R")

n <- 6L

m <- stan_model("models/general_iter4_obs_fit.stan",
                model_name = "m")

d <- make_stan_dat(audiovisual_binomial, function(x) x / 1000)
s <- replicate(n, list(a = 0,
                       aG = rep(0, d$N_G),
                       aT = rep(0, d$N_T),
                       aS = rep(0, d$N_S),
                       b = 2.5,
                       bG = rep(0, d$N_G),
                       bT = rep(0, d$N_T),
                       bS = rep(0, d$N_S)),
               simplify = FALSE)
f <- sampling(m, d, iter=2000, warmup=1000, refresh=100, chains=n, init=s,
              control=list(adapt_delta=0.95, max_treedepth=12))
p <- rstan::extract(f)
