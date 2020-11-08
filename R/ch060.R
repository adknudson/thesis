logit <- function(p) qlogis(p)
inv_logit <- function(x) plogis(x)
logistic <- function(x) inv_logit(x)

Q <- function(p, a, b, l) {
  logit((p - l) / (1 - 2*l)) / exp(b) + a
}

combine_samples <- function(post, age_group, block) {
  with(post, data.frame(
    age_group = age_group,
    block = block,
    alpha = a + aGT[,age_group,block],
    beta  = b + bGT[,age_group,block],
    lambda = lG[,age_group]
  ))
}

post_table <- function(post) {
  age_blk <- expand_grid(G=1:3, B=1:2)
  pmap(age_blk, ~ combine_samples(post, ..1, ..2)) %>%
    do.call(what = bind_rows) %>%
    mutate(age_group = factor(age_group,
                              levels = 1:3,
                              labels = c("Young", "Middle", "Older")),
           block = factor(block,
                          levels = 1:2,
                          labels = c("Pre", "Post"))) %>%
    rename(`Age Group` = age_group, Block = block) %>%
    mutate(gamma = 2 * lambda,
           PSS = Q(0.5, alpha, beta, lambda),
           JND = Q(0.84, alpha, beta, lambda) - PSS)
}

plot_pss <- function(df) {
  p1 <- ggplot(df, aes(PSS, fill = Block)) +
    geom_density(alpha = 0.75) +
    facet_grid(`Age Group` ~ .) +
    scale_fill_manual(values = two_colors) +
    theme(legend.position = "bottom",
          axis.text.y = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank())

  p2 <- ggplot(df, aes(PSS, fill = `Age Group`)) +
    geom_density(alpha = 0.66) +
    facet_grid(Block ~ .) +
    scale_fill_manual(values = three_colors) +
    theme(legend.position = "bottom",
          axis.text.y = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank())

  p1 + p2
}

plot_jnd <- function(df) {
  p1 <- ggplot(df, aes(JND, fill = Block)) +
    geom_density(alpha = 0.75) +
    facet_grid(`Age Group` ~ .) +
    scale_fill_manual(values = two_colors) +
    theme(legend.position = "bottom",
          axis.text.y = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank())

  p2 <- ggplot(df, aes(JND, fill = `Age Group`)) +
    geom_density(alpha = 0.66) +
    facet_grid(Block ~ .) +
    scale_fill_manual(values = three_colors) +
    theme(legend.position = "bottom",
          axis.text.y = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank())

  p1 + p2
}

plot_pss_average <- function(df) {
  p1 <- ggplot(df, aes(PSS, fill = Block)) +
    geom_density(alpha = 0.75) +
    scale_fill_manual(values = two_colors) +
    theme(legend.position = "bottom",
          axis.text.y = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank())

  p2 <- ggplot(df, aes(PSS, fill = `Age Group`)) +
    geom_density(alpha = 0.66) +
    scale_fill_manual(values = three_colors) +
    theme(legend.position = "bottom",
          axis.text.y = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank())

  p1 + p2
}

plot_jnd_average <- function(df) {
  p1 <- ggplot(df, aes(JND, fill = Block)) +
    geom_density(alpha = 0.75) +
    scale_fill_manual(values = two_colors) +
    theme(legend.position = "bottom",
          axis.text.y = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank())

  p2 <- ggplot(df, aes(JND, fill = `Age Group`)) +
    geom_density(alpha = 0.66) +
    scale_fill_manual(values = three_colors) +
    theme(legend.position = "bottom",
          axis.text.y = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank())

  p1 + p2
}

density_with_shade <- function(x, prob) {

  ci <- rethinking::HPDI(x, prob = prob)

  p <- ggplot(data.frame(x = x), aes(x)) +
    geom_density() +
    theme(axis.text.y = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank())

  p_df <- ggplot_build(p)
  x1 <- min(which(p_df$data[[1]]$x >= ci[1]))
  x2 <- max(which(p_df$data[[1]]$x <= ci[2]))

  p +
    geom_area(data=data.frame(x=p_df$data[[1]]$x[x1:x2],
                              y=p_df$data[[1]]$y[x1:x2]),
              aes(x=x, y=y), fill="grey") +
    geom_hline(yintercept = 0) +
    geom_density()
}
