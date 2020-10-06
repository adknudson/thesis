library(FangPsychometric)
library(tidyverse)

process_binomial_dat <- function(data) {
  data %>%
    filter(trial %in% c("pre", "post1")) %>%
    mutate(trt = droplevels(trial),
           sid = droplevels(sid)) %>%
    sample_frac(replace = FALSE)
}

# process_binomial_dat(audiovisual_binomial)

make_stan_dat <- function(data, mutate_soa_fun = identity) {
  data <- process_binomial_dat(data)
  data <- data %>%
    select(soa, k, n, age_group, sid, trt) %>%
    rename(x = soa, G = age_group, S = sid) %>%
    mutate(G = as.integer(G),
           trt = as.integer(trt),
           S = as.integer(S),
           x = mutate_soa_fun(x))
  N <- nrow(data)
  data <- as.list(data)
  data[["N"]]   <- N
  data[["N_G"]] <- max(data$G)
  data[["N_T"]] <- max(data$trt)
  data[["N_S"]] <- max(data$S)
  data
}

# make_stan_dat(audiovisual_binomial, function(x) (x - mean(x)) / (2*sd(x)))
# make_stan_dat(visual_binomial)
# make_stan_dat(duration_binomial)


plot_iter4 <- function(stan_data, post) {
  NULL
}
