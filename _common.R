set.seed(12)
options(digits = 4)

knitr::opts_chunk$set(
  # Text Results
  eval = TRUE,
  echo = FALSE,
  results = 'markup',
  collapse = TRUE,
  warning = FALSE,
  error = FALSE,
  message = FALSE,
  # Code Decoration
  comment = "#>",
  background = '#F7F7F7',
  # Cache
  cache = TRUE,
  # Plots
  fig.show = "hold",
  fig.asp = 0.618,  # 1 / phi
  out.width = "70%",
  fig.align = 'center'
)

# options(dplyr.print_min = 6, dplyr.print_max = 6)

logit <- function(p) qlogis(p)
inv_logit <- function(x) plogis(x)
logistic <- function(x) inv_logit(x)
fn <- function(x, a, b) logistic(b * (x - a))

stan_summary <- function(object, ...) {
  round(summary(object, ...)$summary, 4)[,c(1,2,3,4,8,9,10)]
}
