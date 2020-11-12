


# Model Development {#application}


Multilevel models should be the default. The alternatives are models with complete pooling, or models with no pooling. Pooling vs. no pooling considers modeling all the data as a whole, or each of the the smallest components individually. The former implies that the variation between groups is zero (all groups are the same), and the latter implies that the variation between groups is infinite (no groups are the same). Multilevel models assume that the truth is somewhere between of zero and infinity.


Hierarchical models are a specific kind of multilevel model where one or more groups are nested within a larger one. In the case of the psychometric data, there are three age groups, and within each age group are individual subjects. Multilevel modeling provides a way to quantify and apportion the variation within the data to each level in the model. For an in-depth introduction to multilevel modeling, see @gelman2006data.


## Iteration 1 {#iter1}


**Pre-Model, Pre-Data**


_Conceptual Analysis_

In section \@ref(toj-task) we discussed the experimental setup and data collection. To reiterate, subjects are presented with two stimuli separated by some temporal delay, and they are asked to respond as to their perception of the temporal order. There are 45 subjects with 15 each in the young, middle, and older age groups. As the SOA becomes larger in the positive direction, subjects are expected to give more "positive" responses, and as the SOA becomes larger in the negative direction, more "negative" responses are expected. By the way the experiment and responses are constructed, there is no expectation to see a reversal of this trend unless there was an issue with the subject's understanding of the directions given to them or an error in the recording device.


After the first experimental block the subjects go through a temporal recalibration period, and repeat the experiment again. The interest is in seeing if the recalibration has an effect on temporal sensitivity and perceptual synchrony, and if the effect is different for each age group.


_Define Observational Space_

The response that subjects give during a TOJ task is recorded as a zero or a one, and their relative performance is determined by the SOA value. Let $y$ represent the binary outcome of a trial and let $x$ be the SOA value.


\begin{align*}
y_i &\in \lbrace 0, 1\rbrace \\
x_i &\in \mathbb{R}
\end{align*}


If the SOA values are fixed like in the audiovisual task, then the responses can be aggregated into binomial counts, $k$.


$$k_i, n_i \in \mathbb{Z}_0^+, k_i \le n_i$$


In the above expression, $\mathbb{Z}_0^+$ represents the set of non-negative integers. Notice that the number of trials $n$ has an index variable $i$. This is because the number of trials per SOA is not fixed between blocks. In the pre-adaptation block, there are five trials per SOA compared to three in the post-adaptation block. So if observation 32 is recorded during a "pre" block, $n_{32} = 5$, and if observation 1156 is during a "post" block, $n_{1156} = 3$. Of course this is assuming that each subject completed all trials in the block, but the flexibility of the indexing can manage even if they didn't.


Then there are also three categorical variables: age group, subject ID, and trial (block). The first two are treated as factor variables (also known as index variable or categorical variable). Rather than using one-hot encoding or dummy variables, the age levels are left as categories and a coefficient is fit for each level. Among the benefits of this approach is the ease of interpretation and ease of working with the data programmatically. This is especially true at the subject level. If dummy variables were used for all 45 subjects, there would be 44 different dummy variables to work with times the number of coefficients that make estimates at the subject level. The number of parameters in the model grows rapidly as the model complexity grows.


Age groups and individual subjects can be indexed in the same way that number of trials is indexed. $S_i$ refers to the subject in record $i$, and similarly $G_i$ refers to the age group of that subject. Observation 63 is for record ID av-post1-M-f-HG, so then $S_{63}$ is M-f-HG and $G_{63}$ is middle_age. Under the hood of `R`, these factor levels are represented as integers (e.g. middle age group level is stored internally as the number 2).



```r
(x <- factor(c("a", "a", "b", "c")))
#> [1] a a b c
#> Levels: a b c
storage.mode(x)
#> [1] "integer"
```


This data storage representation can later be exploited for the `Stan` model.


The pre- and post-adaptation categories are treated as a binary indicator referred to as $trt$ (short for treatment) since there are only two levels in the category. In this setup, a value of 1 indicates a post-adaptation block. This encoding is chosen over the reverse because the pre-adaptation block is like the baseline performance, and it is more appropriate to interpret the post-adaptation block as turning on some effect. Using a binary indicator in a regression setting may not be the best practice as we discuss in section \@ref(iter2).


_Construct Summary Statistics_

In order to effectively challenge the validity of the model, a set of summary statistics are constructed that help answer the questions of domain expertise consistency and model adequacy. We are studying the affects of age and temporal recalibration through the PSS and JND (see section \@ref(psycho-experiments)), so it is natural to define summary statistics around these quantities to verify model consistency. Additionally the PSS and JND can be computed regardless of the model parameterization or chosen psychometric function.


By the experimental setup and recording process, it is impossible that a properly conducted block would result in a JND less than 0 (i.e. the psychometric function is always non-decreasing), so that can be a lower limit for its threshold. On the other end it is unlikely that it will be beyond the limits of the SOA values, but even more concretely it seems unlikely (though not impossible) that the just noticeable difference would be more than a second.


The lower bound on the JND can be further refined if we draw information from other sources. Some studies show that we cannot perceive time differences below 30 ms, and others show that an input lag as small as 100ms can impair a person's typing ability. Then according to these studies, a time delay of 100ms is enough to notice, and so a just noticeable difference should be much less than one second -- much closer to 100ms. We will continue to use one second as an extreme estimate indicator, but will incorporate this knowledge when it comes to selecting priors.


As for the point of subjective simultaneity, it can be either positive or negative, with the belief that larger values are more rare. Some studies suggest that for audio-visual temporal order judgment tasks, the separation between stimuli need to be as little as 20ms for subjects to be able to determine which modality came first [@vatakis2007influence]. Other studies suggest that our brains can detect temporal differences as small as 30ms. If these values are to be believed then we should be skeptical of PSS estimates larger than say 150ms in absolute value, just to be safe.


A histogram of computed PSS and JND values will suffice for summary statistics. We can estimate the proportion of values that fall outside of our limits defined above, and use them as indications of problems with the model fitting or conceptual understanding.


**Post-Model, Pre-Data**


It is now time to define priors for the model, while still not having looked at the data. The priors should be motivated by domain expertise and *prior knowledge*, not the data. There are also many choices when it comes to selecting a psychometric (sigmoid) function. Common choices are logistic, Gaussian, and Weibull.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch041-pf-assortment-1} 

}

\caption{Assortment of psychometric functions.}(\#fig:ch041-pf-assortment)
\end{figure}


The Weibull psychometric function is more common when it comes to 2-alternative forced choice (2-AFC) psychometric experiments where the independent variable is a stimulus intensity (non-negative) and the goal is signal detection. The data in this paper includes both positive and negative SOA values, so the Weibull is not a natural choice. Our first choice is the logistic function as it is the canonical choice for Binomial count data. Additionally, the data in this study are exchangeable The label of a positive response can be swapped with the label of a negative response and the inferences should remain the same. Since there is no natural ordering, it makes more sense for the psychometric function to be symmetric, e.g. the logistic and Gaussian. We use symmetric loosely to mean that probability density function (PDF) is symmetric about its middle. More specifically, the distribution has zero skewness. In practice, there is little difference in inferences between the _logit_ and _probit_ links, but computationally the logit link is more efficient.


_Develop Model_

Before moving on to specifying priors, it is appropriate to provide a little more background into generalized linear models (GLMs) and their role in working with psychometric functions. A GLM allows the linear model to be related to the outcome variable via a _link_ function. An example of this is the logit link -- the inverse of the logistic function. The logistic function, $F$, takes $x \in \mathbb{R}$ and constrains the output to be in $(0, 1)$.


\begin{equation}
  F(\theta) = \frac{1}{1 + \exp\left(-\theta\right)}
  (\#eq:logistic)
\end{equation}


Since $F$ is a strictly increasing and continuous function, it has an inverse, and the link for \@ref(eq:logistic) is the log-odds or logit function.


\begin{equation}
  F^{-1}(\pi) = \mathrm{logit}(\pi) = \ln\left(\frac{\pi}{1 - \pi}\right)
  (\#eq:logit)
\end{equation}


By taking $(F^{-1} \circ F)(\theta)$ we can arrive at a relationship that is linear in $\theta$.


\begin{align*}
  \pi = F(\theta) \Longleftrightarrow F^{-1}(\pi) &= F^{-1}(F(\theta)) \\
  & = \ln\left(\frac{F(\theta)}{1 - F(\theta)}\right) \\
  &= \ln(F(\theta)) - \ln(1 - F(\theta)) \\
  &= \ln\left(\frac{1}{1 + \exp(-\theta)}\right) - \ln\left(\frac{\exp(-\theta)}{1 + \exp(-\theta)}\right) \\
  &= - \ln(1 + \exp(-\theta)) - \ln(\exp(-\theta)) + \ln(1 + \exp(-\theta)) \\
  &= - \ln(\exp(-\theta)) \\
  &= \theta
\end{align*}


The purpose of all this setup is to show that a model for the psychometric function can be specified using a linear predictor, $\theta$. Given a simple slope-intercept model, linear predictor would typically be written as:


\begin{equation}
  \theta = \alpha + \beta x
  (\#eq:linearform1)
\end{equation}


This isn't the only acceptable form; it could be written in the centered parameterization:


\begin{equation}
  \theta = \beta(x - a)
  (\#eq:linearform2)
\end{equation}


Both parameterizations will describe the same geometry, so why should it matter which form is chosen? Clearly the interpretation of the parameters change between the two models, but the reason becomes clear when we consider how the linear model relates back to the physical properties that the psychometric model describes. Take equation \@ref(eq:linearform1), substitute it in to \@ref(eq:logistic), and then take the logit of both sides:


\begin{equation}
  \mathrm{logit}(\pi) = \alpha+\beta x
  (\#eq:pfform1)
\end{equation}


Now recall that the PSS is defined as the SOA values such that the response probability, $\pi$, is $0.5$. Substituting $\pi = 0.5$ into \@ref(eq:pfform1) and solving for $x$ yields:


$$pss = -\frac{\alpha}{\beta}$$


Similarly, the JND is defined as the difference between the SOA value at the 84% level and the PSS. Substituting $\pi = 0.84$ into \@ref(eq:pfform1), solving for $x$, and subtracting off the pss yields:


\begin{equation}
  jnd = \frac{\mathrm{logit}(0.84)}{\beta}
  (\#eq:jnd1)
\end{equation}


From the conceptual analysis, it is easy to define priors for the PSS and JND, but then how does one set the priors for $\alpha$ and $\beta$? Let's say the prior for the just noticeable difference is $jnd \sim \pi_j$. Then the prior for $\beta$ would be



$$\beta \sim \frac{\mathrm{logit}(0.84)}{\pi_j}$$


The log-normal distribution has a nice property where its multiplicative inverse is still a log-normal distribution. We could let $\pi_j = \mathrm{Lognormal}(\mu, \sigma^2)$ and then $\beta$ would be distributed as


$$
\beta \sim \mathrm{Lognormal}(-\mu + \ln(\mathrm{logit}(0.84)), \sigma^2)
$$


This is acceptable, as it was determined that the slope must always be positive, and a log-normal distribution constrains the support to positive real numbers. Next suppose that the prior distribution for the PSS is $pss \sim \pi_p$. Then the prior for $\alpha$ is 



$$\alpha \sim -\pi_p \cdot \beta$$


If $\pi_p$ is set to a log-normal distribution as well, then $\pi_p \cdot \beta$ would also be log-normal, but there is still the problem of the negative sign. If $\alpha$ is always negative, then the PSS will also always be negative, which is certainly not always true. Furthermore, we don't want to _a priori_ put more weight on positive PSS values compared to negative ones, for which a log-normal distribution would do.


Let's now go back and consider using equation \@ref(eq:linearform2) and repeat the above process.


\begin{equation}
  \mathrm{logit}(\pi) = \beta(x - a)
  (\#eq:pfform2)
\end{equation}


The just noticeable difference is still given by \@ref(eq:jnd1) and so the same method for choosing a prior can be used, but the PSS is now given by


$$pss = \alpha$$


This is a fortunate consequence of using \@ref(eq:linearform2) because now the JND only depends on $\beta$ and the PSS only depends on $\alpha$, and now $\alpha$ can be interpreted as the PSS of the estimated psychometric function. Also thrown in is the ability to set a prior for $\alpha$ that is symmetric around $0$ such as a Gaussian distribution.


This also brings us to point out the first benefit of using a modeling language like `Stan` over others. For fitting GLMs in `R`, there are a handful of functions that utilize MLE like `stats::glm` and others that use Bayesian methods like `rstanarm::stan_glm` and `arm::bayesglm` [@R-rstanarm; @R-arm]. Each of these functions requires the linear predictor to be in the form of \@ref(eq:linearform1). The `stan_glm` function uses Stan in the back-end to fit a model, but is limited to priors from the Student-t family of distributions. By writing the model directly in `Stan`, the linear model can be parameterized in any way and with any prior distribution, and so allows for much more expressive modeling -- a key aspect of this principled workflow.


For the first iteration of this model, we begin with the simplest model that captures the structure of the data without including information about age group, treatment, or subject. Here is a simple model that draws information from the conceptual analysis. 


\begin{align*}
  k_i &\sim \mathrm{Binomial}(n_i, p_i) \\
  \mathrm{logit}(p_i) &= \beta ( x_i - \alpha )
\end{align*}


Since we are using the linear model from \@ref(eq:linearform2), setting the priors for $\alpha$ and $\beta$ is relatively straightforward. The PSS can be positive or negative without any expected bias towards either, so a symmetric distribution like the Gaussian is a fine choice for $\alpha$ without having any other knowledge about the distribution of PSS values. We determined earlier that a PSS value more than 150ms in absolute value is unlikely, so we can define a Gaussian prior such that $P(|pss| > 0.150) \approx 0.01$. Since the prior does not need to be exact, the following mean and variance suffice:


$$
pss \sim \mathcal{N}(0, 0.06^2) \Longleftrightarrow \alpha \sim \mathcal{N}(0, 0.06^2)
$$


For the just noticeable difference, we continue to use the log-normal distribution because it is constrained to positive values and has the nice reciprocal property. The JND is expected to be close to 100ms and extremely unlikely to exceed 1 second. This implies a prior such that the mean is around 100ms and the bulk of the distribution is below 1 second - i.e. $E[X] \approx 0.100$ and $P(X < 1) \approx 0.99$. This requires solving a system of nonlinear equations in two variables


$$
\begin{cases}
E[X] = 0.100 = \exp\left(\mu + \sigma^2 / 2\right) \\
P(X < 1) = 0.99 = 0.5 + 0.5 \cdot \mathrm{erf}\left[\frac{\ln (1) - \mu}{\sqrt{2} \cdot \sigma}\right]
\end{cases}
$$


This nonlinear system can be solved using `Stan`'s algebraic solver (code provided in the [appendix](#code)).






```r
fit <- sampling(prior_jnd, 
                iter=1, warmup=0, chains=1, refresh=0,
                seed=31, algorithm="Fixed_param")
sol <- extract(fit)
sol$y
#>           
#> iterations   [,1]  [,2]
#>       [1,] -7.501 3.225
```


The solver has determined that $\mathrm{Lognormal}(-7.5, 3.2^2)$ is the appropriate prior. However, simulating some values from this distribution produces a lot of extremely small values ($<10^{-5}$) and a few extremely large values ($\approx 10^2$). This is because the expected value of a log-normal random variable depends on both the mean and standard deviation. If the median is used in place for the mean, then a more acceptable prior may be determined.







```r
fit <- sampling(prior_jnd_using_median, 
                iter=1, warmup=0, chains=1, refresh=0,
                seed=31, algorithm="Fixed_param")
sol <- extract(fit)
sol$y
#>           
#> iterations   [,1]   [,2]
#>       [1,] -2.303 0.9898
```


Sampling from a log-normal distribution with these parameters and plotting the histogram shows no inconsistency with the domain expertise.



\begin{center}\includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch041-Risky-Lion-1} \end{center}


So now with a prior for the JND, the prior for $\beta$ can be determined.


$$
jnd \sim \mathrm{Lognormal}(-2.3, 0.99^2) \Longleftrightarrow \frac{1}{jnd} \sim \mathrm{Lognormal}(2.3, 0.99^2)
$$


and 


$$
\beta = \frac{\mathrm{logit}(0.84)}{jnd} \sim \mathrm{Lognormal}(2.8, 0.99^2)
$$


The priors do not need to be too exact. Rounding the parameters for $\beta$, the simple model is


\begin{align*}
  k_i &\sim \mathrm{Binomial}(n_i, p_i) \\
  \mathrm{logit}(p_i) &= \beta ( x_i - \alpha ) \\
  \alpha &\sim \mathcal{N}(0, 0.06^2) \\
  \beta &\sim \mathrm{Lognormal}(3, 1^2)
\end{align*}


and in Stan, the model code is



```stan
data {
  int N;
  int n[N];
  int k[N];
  vector[N] x;
}
parameters {
  real alpha;
  real<lower=0> beta;
}
model {
  vector[N] p = beta * (x - alpha);
  alpha ~ normal(0, 0.06);
  beta ~ lognormal(3.0, 1.0);
  k ~ binomial_logit(n, p);
}
generated quantities {
  vector[N] log_lik;
  vector[N] k_pred;
  vector[N] theta = beta * (x - alpha);
  vector[N] p = inv_logit(theta);
  for (i in 1:N) {
    log_lik[i] = binomial_logit_lpmf(k[i] | n[i], theta[i]);
    k_pred[i]  = binomial_rng(n[i], p[i]);
  }
}
```


Notice that the model block is nearly identical to the mathematical model specified above.


_Construct Summary Functions_

That was a lot of work to define the priors for just two parameters. Going forward, not as much work will need to be done to expand the model. The next step is to construct any relevant summary functions. Since the distribution of posterior PSS and JND values are needed for the summary statistics, it will be nice to have a function that can take in the posterior samples for $\alpha$ and $\beta$ and return the PSS and JND values. We define $Q$ as a more general function that takes in the two parameters and a target probability, $\pi$, and returns the distribution of SOA values at $\pi$.


\begin{equation}
  Q(\pi; \alpha, \beta) = \frac{\mathrm{logit(\pi)}}{\beta} + \alpha
  (\#eq:summfun1)
\end{equation}


The function can be defined in `R` as



```r
Q <- function(p, a, b) qlogis(p) / b + a
```


With $Q$, the PSS and JND can be calculated as


\begin{align}
  pss &= Q(0.5) \\
  jnd &= Q(0.84) - Q(0.5)
\end{align}


_Simulate Bayesian Ensemble_

During this step, we simulate the Bayesian model and later feed the prior values into the summary functions in order to verify that there are no other inconsistencies with domain knowledge. Since the model is fairly simple, we simulate directly in `R`.



```r
set.seed(124)
n <- 10000

a <- rnorm(n, 0, 0.06)
b <- rlnorm(n, 3.0, 1)

dat <- with(av_dat, list(N = N, x = x, n = n)) 
n_obs <- length(dat$x)

idx <- sample(1:n, n_obs, replace = TRUE)
probs <- logistic(b[idx] * (dat$x - a[idx]))
sim_k <- rbinom(n_obs, dat$n, probs)
```


_Prior Checks_

This step pertains to ensuring that prior estimates are consistent with domain expertise. We already did that in the model construction step by sampling values for the just noticeable difference. The first prior chosen was not producing JND estimates that were consistent with domain knowledge, so we adjusted accordingly. That check would normally be done during this step, and we would have had to return to the model development step at the violation of domain expertise consistency.


Figure \@ref(fig:ch041-prior-pf-plot) shows the distribution of prior psychometric functions derived from the simulated ensemble. There are a few very steep and very shallow curves, but the majority fall within a range that appears likely.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch041-prior-pf-plot-1} 

}

\caption{Prior distribution of psychometric functions using the priors for alpha and beta.}(\#fig:ch041-prior-pf-plot)
\end{figure}


Additionally most of the PSS values are within $\pm 0.1$ with room to allow for some larger values. Let's check the prior distribution of PSS and JND values.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch041-prior-pss-plot-1} 

}

\caption{PSS prior distribution.}(\#fig:ch041-prior-pss-plot)
\end{figure}


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch041-prior-jnd-plot-1} 

}

\caption{JND prior distribution.}(\#fig:ch041-prior-jnd-plot)
\end{figure}


We are satisfied with the prior coverage of the PSS and JND values, and there are only a few samples that go beyond the extremes that were specified in the summary statistics step.


_Configure Algorithm_

There are a few parameters that can be set for `Stan`. On the user side, the main parameters are the number of iterations, the number of warm-up iterations, the target acceptance rate, and the number of chains to run. By default, `Stan` will use half the number of iterations for warm-up and the other half for actual sampling. The full details of `Stan`'s HMC algorithm is described in the Stan reference manual. For now we use the default algorithm parameters in `Stan`, and will tweak them later if and when issues arise.


_Fit Simulated Ensemble_

We now fit the model to the simulated data.



```r
sim_dat <- with(av_dat, list(N = N, x = x, n = n, k = sim_k)) 
m041 <- rstan::sampling(m041_stan, data = sim_dat, 
                        chains = 4, cores = 4, refresh = 0)
```


_Algorithmic Calibration_

One benefit of using HMC over other samplers like Gibbs sampling is that HMC offers diagnostic tools for the health of chains and the ability to check for _divergent transitions_ (discussed in \@ref(model-fitting)). To check the basic diagnostics of the model, we run the following code.



```r
check_hmc_diagnostics(m041)
#> 
#> Divergences:
#> 0 of 4000 iterations ended with a divergence.
#> 
#> Tree depth:
#> 0 of 4000 iterations saturated the maximum tree depth of 10.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```


There is no undesirable behavior from this model, so next we check the summary statistics of the estimated parameters.


\begin{table}[!h]

\caption{(\#tab:ch041-Cloudy-Toupee)Summary statistics of the fitted Bayesian ensemble.}
\centering
\begin{tabular}[t]{lrrrrrrr}
\toprule
parameter & mean & se\_mean & sd & 2.5\% & 97.5\% & n\_eff & Rhat\\
\midrule
alpha & 0.0061 & 0.0001 & 0.0038 & -0.0012 & 0.0136 & 4039 & 0.9995\\
beta & 10.7681 & 0.0051 & 0.2404 & 10.3043 & 11.2313 & 2202 & 1.0003\\
\bottomrule
\end{tabular}
\end{table}


Both the $\hat{R}$ and $N_{\mathrm{eff}}$ look fine for both $\alpha$ and $\beta$, though it is slightly concerning that $\alpha$ is centered relatively far from zero. This could just be due to sampling variance, so we will continue on to the next step.


**Post-Model, Post-Data**

_Fit Observed Data_

All of the work up until now has been done without peaking at the observed data. Satisfied with the model so far, we go ahead and run the data through.






```r
m041 <- sampling(m041_stan, data = obs_dat, 
                 chains = 4, cores = 4, refresh = 200)
```


_Diagnose Posterior Fit_

Here we repeat the diagnostic checks that were used after fitting the simulated data. 



```r
check_hmc_diagnostics(m041)
#> 
#> Divergences:
#> 0 of 4000 iterations ended with a divergence.
#> 
#> Tree depth:
#> 0 of 4000 iterations saturated the maximum tree depth of 10.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```


\begin{table}[!h]

\caption{(\#tab:ch041-Maroon-Oyster)Summary statistics of the fitted Bayesian ensemble.}
\centering
\begin{tabular}[t]{lrrrrrrr}
\toprule
parameter & mean & se\_mean & sd & 2.5\% & 97.5\% & n\_eff & Rhat\\
\midrule
alpha & 0.0373 & 0.0001 & 0.0043 & 0.029 & 0.0458 & 3765 & 1.000\\
beta & 8.4259 & 0.0039 & 0.1839 & 8.070 & 8.7897 & 2249 & 1.001\\
\bottomrule
\end{tabular}
\end{table}


There are no indications of an ill-behaved posterior fit. Let's also check the posterior distribution of $\alpha$ and $\beta$ against the prior density (\@ref(fig:ch041-m041-posterior-alpha-beta)).


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch041-m041-posterior-alpha-beta-1} 

}

\caption{Comparison of posterior distributions for alpha and beta to their respective prior distributions.}(\#fig:ch041-m041-posterior-alpha-beta)
\end{figure}


The posterior distributions for $\alpha$ and $\beta$ are well within the range determined by domain knowledge, and highly concentrated due to both the large amount of data and the fact that this is a completely pooled model -- all subject data is used to estimate the parameters. As expected, the prior for the JND could have been tighter with more weight below half a second compared to the one second limit used, but this is not prior information, so it is not prudent to change the prior in this manner after having seen the posterior. As a rule of thumb, priors should only be updated as motivated by domain expertise and not by posterior distributions.


_Posterior Retrodictive Checks_

It is time to run the posterior samples through the summary functions and then perform _retrodictive_ checks. A retrodiction is using the posterior model to predict and compare to the observed data. This is simply done by drawing samples from the posterior and feeding in the observational data. This may be repeated to gain a retrodictive distribution.



```r
posterior_pss <- Q(0.5, p041$alpha, p041$beta)
posterior_jnd <- Q(0.84, p041$alpha, p041$beta) - posterior_pss
```


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch041-posterior-pss-jnd-plot-1} 

}

\caption{Posterior distribution of the PSS and JND.}(\#fig:ch041-posterior-pss-jnd-plot)
\end{figure}

Neither of the posterior estimates for the PSS or JND exceed the extreme cutoffs set in the earlier steps, so we can be confident that the model is consistent with domain expertise. Let's also take a second to appreciate how simple it is to visualize and summarize the distribution of values for these measures. Using classical techniques like MLE might require using bootstrap methods to estimate the distribution of parameter values, or one might approximate the parameter distributions using the mean and standard error of the mean to simulate new values. Since we have the entire posterior distribution we can calculate the distribution of transformed parameters by working directly with the posterior samples and be sure that the intervals are credible.


Next is to actually do the posterior retrodictions. We do this in two steps to better show how the distribution of posterior psychometric functions relates to the observed data, and then compare the observed data to the retrodictions. Figure \@ref(fig:ch041-posterior-pf-plot) shows the result of the first step.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch041-posterior-pf-plot-1} 

}

\caption{Posterior distribution of psychometric functions using pooled observations.}(\#fig:ch041-posterior-pf-plot)
\end{figure}


Next we sample parameter values from the posterior distribution and use them to simulate a new data set. In the next iteration we show how to get `Stan` to automatically produce retrodictions for the model fitting step. The results of the posterior retrodictions are shown in figure \@ref(fig:ch041-obs-vs-retro-plot).



```r
alpha <- sample(p041$alpha, n_obs, replace = TRUE)
beta  <- sample(p041$beta, n_obs, replace = TRUE)
logodds <- beta * (av_dat$x - alpha)
probs <- logistic(logodds)
sim_k <- rbinom(n_obs, av_dat$n, probs)
```


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch041-obs-vs-retro-plot-1} 

}

\caption{Observed data compared to the posterior retrodictions. The data is post-stratified by block for easier visualization.}(\#fig:ch041-obs-vs-retro-plot)
\end{figure}


Let's be clear exactly what the first iteration of this model tells us. It is the average distribution of underlying psychometric functions across all subjects and blocks. It cannot tell us what the differences are between pre- and post-adaptation blocks are, or even what the variation between subjects is. As such, it is only useful in determining if the average value for the PSS is different from 0 or if the average JND is different from some other predetermined level. This model is still useful given the right question, but this model cannot answer questions about group-level effects.


Figure \@ref(fig:ch041-obs-vs-retro-plot) shows that the model captures the broad structure of the observed data, but is perhaps a bit under-dispersed in the tail ends of the SOA values. Besides this one issue, we am satisfied with the first iteration of this model and are ready to proceed to the next iteration.


## Iteration 2 {#iter2}


In this iteration we will be adding in the treatment and age groups into the model. There are no changes with the conceptual understanding of the experiment, and nothing to change with the observational space. As such we will be skipping the first three steps and go straight to the model development step. As we build the model, the number of changes from one iteration to the next should go to zero as the model _expands_ to become only as complex as necessary to answer the research questions.


**Post-Model, Pre-Data**

_Develop Model_

To start, let's add in the treatment indicator and put off consideration of adding in the age group levels. In classical statistics, it is added as an indicator variable -- a zero or one -- for both the slope and intercept (varying slopes, varying intercepts model). Let $trt$ be $0$ if it is the pre-adaptation block and $1$ if the observation comes from the post-adaptation block.


$$
\theta = \alpha + \alpha_{trt} \times trt + \beta \times x + \beta_{trt}\times trt \times x
$$


Now when an observation comes from the pre-adaptation block ($trt=0$) the linear predictor is given by


$$
\theta_{pre} = \alpha + \beta \times x
$$


and when an observation comes from the post-adaptation block ($trt=1$) the linear predictor is


$$
\theta_{post} = (\alpha + \alpha_{trt}) + (\beta + \beta_{trt}) \times x
$$


This may seem like a natural way to introduce an indicator variable, but it comes with serious implications. This model implies that there is more uncertainty about the post-adaptation block compared to the baseline block, and this is not necessarily true. 


\begin{align*}
\mathrm{Var}(\theta_{post}) &= \mathrm{Var}((\alpha + \alpha_{trt}) + (\beta + \beta_{trt}) \times x) \\
&= \mathrm{Var}(\alpha) + \mathrm{Var}(\alpha_{trt}) + x^2 \mathrm{Var}(\beta) + x^2\mathrm{Var}(\beta_{trt})
\end{align*}


On the other hand, the variance of $\theta_{pre}$ is


$$
\mathrm{Var}(\theta_{pre}) = \mathrm{Var}(\alpha) + x^2 \mathrm{Var}(\beta) \le \mathrm{Var}(\theta_{post})
$$


Furthermore, the intercept, $\alpha$, is no longer the average response probability at $x=0$ for the entire data set, but is instead exclusively the average for the pre-adaptation block. This may not matter in certain analyses, but one nice property of multilevel models is the separation of population level estimates and group level estimates (fixed vs. mixed effects).


Instead the treatment variable is introduced into the linear model as a factor variable. This essentially means that each level in the treatment gets its own parameter estimate, and this also makes it easier to set priors when there are many levels in a group (such as for the subject level). The linear model, using equation \@ref(eq:linearform2), with the treatment is written as


\begin{equation}
  \theta = (\beta + \beta_{trt[i]}) \left[x_i - (\alpha + \alpha_{trt[i]})\right]
  (\#eq:linearmodel2)
\end{equation}


As predictors and groups are added in, equation \@ref(eq:linearmodel2) will start to be more difficult to read. What we can do is break up the slope and intercept parameters and write the linear model as


\begin{align*}
\mu_\alpha &= \alpha + \alpha_{trt[i]} \\
\mu_\beta &= \beta + \beta_{trt[i]} \\
\theta &= \mu_\beta (x - \mu_\alpha)
\end{align*}


In this way the combined parameters can be considered separately from the linear parameterization. Which leads us to consider the priors for $\alpha_{trt}$ and $\beta_{trt}$. The way that we can turn a normal model with categorical predictors into a multilevel model is by allowing the priors to borrow information from other groups. This is accomplished by putting priors on priors. It is easier to write down the model first before explaining how it works.


\begin{align*}
k_i &\sim \mathrm{Binomial}(n_i, p_i) \\
\mu_\alpha &= \alpha + \alpha_{trt[i]} \\
\mu_\beta &= \beta + \beta_{trt[i]} \\
\mathrm{logit}(p_i) &= \mu_\beta (x_i - \mu_\alpha) \\
\alpha &\sim \mathcal{N}(0, 0.06^2) \\
\alpha_{trt} &\sim \mathcal{N}(0, \sigma_{trt}^2) \\
\sigma_{trt} &\sim \textrm{to be defined}
\end{align*}


In the above model, $\alpha$ gets a fixed prior (the same as in the first iteration), and $\alpha_{trt}$ gets a Gaussian prior with an adaptive variance term that is allowed to be learned from the data. This notation is compact, but $\alpha_{trt}$ is actually two parameters - one each for the pre- and post-adaptation blocks, but they both share the same variance term $\sigma_{trt}$. This produces a _regularizing_ effect where both treatment estimates are shrunk towards the mean, $\alpha$.


We will discuss selecting a prior for the variance term shortly, but now we want to discuss setting the prior for the slope terms. Instead of modeling $\beta$ with a log-normal prior, we can sample from a normal distribution and take the exponential of it to produce a log-normal distribution. I.e.


\begin{align*}
X &\sim \mathcal{N}(3, 1^2) \\
Y = \exp\left\lbrace X \right\rbrace &\Longleftrightarrow Y \sim \mathrm{Lognormal(3, 1^2)}
\end{align*}


The motivation behind this transformation is that it is now easier to include new slope variables as an additive affect. If both $\beta$ and $\beta_{trt}$ are specified with Gaussian priors, then the exponential of the sum will be a log-normal distribution. The model now gains


\begin{align*}
\mathrm{logit}(p_i) &= \exp(\mu_\beta) (x_i - \mu_\alpha) \\
\beta &\sim \mathcal{N}(3, 1^2) \\
\beta_{trt} &\sim \mathcal{N}(0, \gamma_{trt}^2) \\
\gamma_{trt} &\sim \textrm{to be defined}
\end{align*}


Deciding on priors for the variance term requires some careful consideration. In one sense, the variance term is the within-group variance. @gelman2006prior recommends that for multilevel models with groups with less than say 5 levels to use a half Cauchy prior. This weakly informative prior still has a regularizing affect and dissuades larger variance estimates. Even though the treatment group only has two levels, there is still value in specifying an adaptive prior for them, and there is also a lot of data for each treatment so partial pooling won't have an extreme regularizing effect.


\begin{align*}
\sigma_{trt} &\sim \mathrm{HalfCauchy}(0, 1) \\
\gamma_{trt} &\sim \mathrm{HalfCauchy}(0, 1)
\end{align*}


Finally we add in the age group level effects and specify the variance terms.


\begin{align*}
\alpha_{G} &\sim \mathcal{N}(0, \tau_{G}^2)\\
\beta_{G} &\sim \mathcal{N}(0, \nu_{G}^2) \\
\tau_{G} &\sim \mathrm{HalfCauchy}(0, 2) \\
\nu_{G} &\sim \mathrm{HalfCauchy}(0, 2)
\end{align*}


The corresponding `Stan` model is becoming quite long, so we omit it from here on out. The final `Stan` model code may be found in the [supplementary code](#code) of the appendix.





**Post-Model, Post-Data**

_Fit Observed Data_

We are choosing to skip the prior checks this time around and use the observed data to configure the algorithm and diagnose the posterior fit.






```r
m042 <- sampling(m042_stan, data = obs_dat, seed = 124,
                 chains = 4, cores = 4, refresh = 100)
```


_Diagnose Posterior Fit_


```r
check_hmc_diagnostics(m042)
#> 
#> Divergences:
#> 4 of 4000 iterations ended with a divergence (0.1%).
#> Try increasing 'adapt_delta' to remove the divergences.
#> 
#> Tree depth:
#> 0 of 4000 iterations saturated the maximum tree depth of 10.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```


As well as the 4 divergent transitions, there was also a message about the effective sample size (ESS) being too low. The recommended prescription for low ESS is to run the chains for more iterations. The posterior summary shows that $N_{\mathrm{eff}}$ is low for the age group level parameters (table \@ref(tab:ch042-Liquid-Strawberry-Eagle)).


\begin{table}[!h]

\caption{(\#tab:ch042-Liquid-Strawberry-Eagle)Summary statistics of the second iteration.}
\centering
\begin{tabular}[t]{lrrrrrrr}
\toprule
parameter & mean & se\_mean & sd & 2.5\% & 97.5\% & n\_eff & Rhat\\
\midrule
a & 0.0222 & 0.0014 & 0.0412 & -0.0683 & 0.1024 & 824.6 & 1.002\\
aG[1] & -0.0009 & 0.0012 & 0.0313 & -0.0531 & 0.0714 & 703.5 & 1.003\\
aG[2] & 0.0274 & 0.0012 & 0.0316 & -0.0218 & 0.0990 & 698.3 & 1.003\\
aG[3] & -0.0078 & 0.0012 & 0.0311 & -0.0609 & 0.0609 & 714.3 & 1.004\\
b & 2.4114 & 0.0216 & 0.5665 & 1.4902 & 3.8499 & 688.2 & 1.003\\
\addlinespace
bG[1] & 0.0030 & 0.0170 & 0.2942 & -0.7681 & 0.5013 & 301.3 & 1.004\\
bG[2] & 0.0538 & 0.0170 & 0.2940 & -0.7101 & 0.5499 & 299.9 & 1.004\\
bG[3] & -0.2223 & 0.0172 & 0.2955 & -1.0150 & 0.2597 & 296.9 & 1.004\\
\bottomrule
\end{tabular}
\end{table}


We can return to the algorithm configuration step and increase the number of iterations and warm-up iterations, as well as increase the adapt delta parameter to reduce the number of divergent transitions (which really isn't a problem right now).


Another technique we can employ is non-centered parameterization, and now is as good a time as any to introduce it. We have already quietly used non-centered parameterization in this iteration of the model without addressing it -- the transformation of $\beta$ from a Gaussian to a log-normal distribution.


Because HMC is a physics simulation, complicated geometry or posteriors with steep slopes can be difficult to traverse if the step size is too course. The solution is to explore a simpler geometry, and then transform the sample into the target distribution. Reparameterization is especially important for hierarchical models. The Cauchy distribution used for the variance term can be reparameterized by first drawing from a uniform distribution on $(-\pi/2, \pi/2)$. For a half Cauchy distribution, just sample from $\mathcal{U}(0, \pi/2)$.


\begin{align*}
X &\sim \mathcal{U}(-\pi/2, \pi/2) \\
Y &= \mu + \tau \cdot \tan(X) \Longrightarrow Y \sim \mathrm{Cauchy(\mu, \tau)}
\end{align*}


The Gaussian distributions can be reparameterized in a similar way. If $Z$ is a standard normal random variable, then $\mu + \sigma Z \sim \mathcal{N}(\mu, \sigma^2)$. For `Stan`, sampling from a standard normal or uniform distribution is very easy, and so the non-centered parameterization can alleviate divergent transitions. We now return to the model development step and incorporate the new methods.


_Develop Model_

The model changes consist of using the non-centered parameterizations discussed in the previous step. An example is in the parameterization of $\tau_{G}$. The other variance terms are parameterized in the same fashion.


\begin{align*}
U_\tau &\sim \mathcal{U}(0, \pi/2) \\
\tau_{G} &= 2 \cdot \tan(U_1) \Longrightarrow \tau_G \sim \mathrm{HalfCauchy}(0, 2)
\end{align*}





As an aside, a multilevel model can be fit in `R` using `lme4::glmer`, `brms::brm`, or `rstanarm::stan_glmer`, and they all use the same notation to specify the model. The notation is very compact, but easy to unpack. Values not in a grouping term are _fixed_ effects and values in a grouping term (e.g. `(1 + x | G)`) are _mixed_ or _random_ effects depending on which textbook you read.



```r
f <- formula(k|n ~ 1 + x + (1 + x | G) + (1 + x | trt))

lme4::glmer(f, data = data, family = binomial("logit"))
rstanarm::stan_glmer(f, data = data, family = binomial("logit"))
brms::brm(f, data = data, family = binomial("logit"))
```


The simpler notation and compactness of these methods are very attractive, and for certain analyses they may be more than sufficient. The goal here is to decide early on if these methods satisfy the model adequacy, and to use more flexible modeling tools like `Stan` if necessary.


_Fit Observed Data_

Moving on to refitting the data, this time with the non-centered parameterization. Since this model is sampling from intermediate parameters, we can choose to keep only the transformed parameters.






```r
m042nc <- sampling(m042nc_stan, data = obs_dat, seed = 143,
                   iter = 4000, warmup = 2000, pars = keep_pars,
                   control = list(adapt_delta = 0.95), thin = 2,
                   chains = 4, cores = 4, refresh = 100)
```


_Diagnose Posterior Fit_


```r
check_hmc_diagnostics(m042nc)
#> 
#> Divergences:
#> 32 of 4000 iterations ended with a divergence (0.8%).
#> Try increasing 'adapt_delta' to remove the divergences.
#> 
#> Tree depth:
#> 0 of 4000 iterations saturated the maximum tree depth of 10.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```


There are still a few divergent transitions ($<1\%$), but the effective sample size increased significantly (table \@ref(tab:ch042-Bleeding-Tuna)).


\begin{table}[!h]

\caption{(\#tab:ch042-Bleeding-Tuna)Summary statistics of the second iteration with non-centered parameterization.}
\centering
\begin{tabular}[t]{lrrrrrrr}
\toprule
parameter & mean & se\_mean & sd & 2.5\% & 97.5\% & n\_eff & Rhat\\
\midrule
a & 0.0192 & 0.0008 & 0.0419 & -0.0744 & 0.0956 & 2509 & 1.0005\\
aG[1] & -0.0025 & 0.0006 & 0.0326 & -0.0636 & 0.0739 & 2737 & 1.0014\\
aG[2] & 0.0262 & 0.0006 & 0.0328 & -0.0342 & 0.1044 & 2644 & 1.0014\\
aG[3] & -0.0093 & 0.0006 & 0.0326 & -0.0713 & 0.0652 & 2752 & 1.0011\\
aT[1] & 0.0185 & 0.0009 & 0.0425 & -0.0546 & 0.1242 & 2338 & 1.0005\\
\addlinespace
aT[2] & 0.0039 & 0.0009 & 0.0419 & -0.0679 & 0.1089 & 2404 & 1.0005\\
b & 2.3841 & 0.0115 & 0.5284 & 1.4762 & 3.6952 & 2109 & 1.0010\\
bG[1] & 0.0170 & 0.0049 & 0.2730 & -0.6323 & 0.4979 & 3106 & 1.0004\\
bG[2] & 0.0678 & 0.0049 & 0.2728 & -0.5773 & 0.5671 & 3113 & 1.0005\\
bG[3] & -0.2075 & 0.0050 & 0.2741 & -0.8506 & 0.2767 & 3026 & 1.0004\\
\addlinespace
bT[1] & -0.2764 & 0.0106 & 0.4914 & -1.6338 & 0.5427 & 2141 & 0.9999\\
bT[2] & -0.0501 & 0.0106 & 0.4909 & -1.4120 & 0.7778 & 2125 & 1.0000\\
\bottomrule
\end{tabular}
\end{table}


A more direct way to compare the efficiency is through the ratio of $N_{\mathrm{eff}} / N$ (figure \@ref(fig:ch042-Remote-Longitude)). 


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch042-Remote-Longitude-1} 

}

\caption{Model efficiency as measured by the N\_eff/N ratio.}(\#fig:ch042-Remote-Longitude)
\end{figure}


Figure \@ref(fig:ch042-traceplot-m042nc) shows the trace plot for the slope and intercept parameters. Each chain looks like it is sampling around the same average value as the others with identical spreads (stationary and homoscedastic). This also helps to solidify the idea that the $\hat{R}$ statistic is the measure of between chain variance compared to cross chain variance.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch042-traceplot-m042nc-1} 

}

\caption{Traceplot for the slope and intercept parameters.}(\#fig:ch042-traceplot-m042nc)
\end{figure}


The chains in figure \@ref(fig:ch042-traceplot-m042nc) look healthy as well as for the other parameters not shown. Since there are no algorithmic issues, we can proceed to the posterior retrodictive checks.


_Posterior Retrodictive Checks_

In this iteration of the model, we now have estimates for the age groups and the treatment. The posterior estimates for the PSS and JND are shown in figure \@ref(fig:ch042-posterior-pss-jnd-plot). There are many ways to visualize and compare the distributions across age groups and conditions, and it really depends on what question is being asked. If for example the question is "what is the qualitative difference between pre- and post-adaptation across age groups?", then figure \@ref(fig:ch042-posterior-pss-jnd-plot) could answer that because it juxtaposes the two blocks in the same panel. We will consider alternative ways of arranging the plots in [chapter 5](#results).





\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch042-posterior-pss-jnd-plot-1} 

}

\caption{Posterior distribution of the PSS and JND.}(\#fig:ch042-posterior-pss-jnd-plot)
\end{figure}


As for the posterior retrodictions, we can do something similar to last time. First note that we had `Stan` perform posterior retrodictions during the fitting step. This was achieved by adding a _generated quantities_ block to the Stan program that takes the posterior samples for the parameters, and then randomly generates a value from a binomial distribution for each observation in the data. In effect, we now have $4000$ simulated data sets.


We only need one to compare to the observed data, so it is selected randomly from the posterior.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch042-obs-vs-retro-plot-1} 

}

\caption{Observed data compared to the posterior retrodictions.}(\#fig:ch042-obs-vs-retro-plot)
\end{figure}


The posterior retrodictions  in figure \@ref(fig:ch042-obs-vs-retro-plot) show no disagreement between the model and the observed data. We could almost say that this model is complete, but this model has one more problem: it measures the average difference in blocks, and the average difference in age groups, but does not consider any interaction between the two. Implicitly it assumes that temporal recalibration affects all age groups the same which may not be true, so in the next iteration we will need to address that.


## Iteration 3 {#iter3}


Since there is no change in the pre-model analysis, we will again jump straight to the model development step, after which we will jump right to the posterior retrodictive checks. The changes to the model going forward are minor, and subsequent steps are mostly repetitions of the ones taken in the first two iterations.


**Post-Model, Pre-Data**

_Develop Model_

We need to model an interaction between age group and treatment. In a simple model in `R`, interactions between factor variable $A$ and factor variable $B$ can be accomplished by taking the cross-product of all the factor levels. For example, if $A$ has levels $a, b, c$ and $B$ has levels $x, y$, then the interaction variable $C=A:B$ will have levels $ax, ay, bx, by, cx, cy$. The concept is similar in `Stan`: create a new variable that is indexed by the cross of the two other factor variables.


$$
\beta_{G[i] \times trt[i]} \Longrightarrow bGT[G[i], trt[i]] 
$$


In the above expression, the interaction variable $\beta_{G[i] \times trt[i]}$ is between age group and treatment. The right hand side is the corresponding `Stan` parameter. Notice that it is an array-like object that is indexed by the age group at observation $i$ and the treatment at observation $i$. For example, observation $51$ is for a middle age adult subject during the post-adaptation block, so $bGT[G[51], trt[51]] = bGT[2, 2]$. An interaction term is added for both the slope and intercept in this iteration.





**Post-Model, Post-Data**





_Diagnose Posterior Fit_

This model has no divergent transitions or abnormally large $\hat{R}$ values. Furthermore the trace-rank plots show uniformity between chains indicating that the chains are all exploring the same regions.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch043-Morning-Rich-1} 

}

\caption{Trace-rank plots for the intercept interaction parameters.}(\#fig:ch043-Morning-Rich)
\end{figure}


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch043-Tuba-Intensive-1} 

}

\caption{Trace and trace-rank plots for the hierarchical variance terms. The chains are healthy and exploring the posterior efficiently.}(\#fig:ch043-Tuba-Intensive)
\end{figure}


_Posterior Retrodictive Checks_

Again we start with the PSS and JND posterior densities. Because the model now allows for the interaction of age group and block, there is no longer a fixed shift in the posterior distribution of the PSS and JND values. Figure \@ref(fig:ch043-posterior-pss-jnd-plot) shows that temporal recalibration had no discernible affect on the PSS estimates for the middle age group.





\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch043-posterior-pss-jnd-plot-1} 

}

\caption{Posterior distribution of the PSS and JND.}(\#fig:ch043-posterior-pss-jnd-plot)
\end{figure}


The posterior retrodictions for this model are going to be similar to the last iteration. Instead, we want to see how this model performs when it comes to the posterior retrodictions of the visual TOJ data. There is something peculiar about that data that is readily apparent when we try to fit a GLM using classical MLE.



```r
vis_mle <- glm(cbind(k, n-k) ~ 0 + sid + sid:soa,
               data = visual_binomial, family = binomial("logit"))
```


We get a message saying that the fitted probabilities are numerically 0 or 1. What does this mean? First this model estimates a slope and an intercept for each subject individually (no pooling model), so we can look at the estimates for each subject. Table \@ref(tab:ch043-Intensive-Oyster) shows the top 3 coefficients sorted by largest standard error of the estimate for both slope and intercept.


\begin{table}[!h]

\caption{(\#tab:ch043-Intensive-Oyster)Coefficients with the largest standard errors.}
\centering
\begin{tabular}[t]{llrrrr}
\toprule
Subject & Coefficient & Estimate & Std. Error & z value & Pr(>|z|)\\
\midrule
Y-m-CB & Slope & 0.6254 & 12.7380 & 0.0491 & 0.9608\\
M-f-DB & Slope & 0.1434 & 0.0442 & 3.2471 & 0.0012\\
M-f-CC & Slope & 0.1434 & 0.0442 & 3.2471 & 0.0012\\
O-f-MW & Intercept & -3.6313 & 1.2170 & -2.9837 & 0.0028\\
M-f-CC & Intercept & -2.4925 & 1.0175 & -2.4497 & 0.0143\\
\addlinespace
M-f-DB & Intercept & -1.0928 & 0.6389 & -1.7105 & 0.0872\\
\bottomrule
\end{tabular}
\end{table}

The standard error of the slope estimate for subject `Y-m-CB` is incredibly large in comparison to its own estimate and in comparison to the slope with the next largest standard error. To see what's going wrong, let's look at the graph for this subject.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch043-Y-m-CB-vis-response-1} 

}

\caption{There is almost complete separation in the data.}(\#fig:ch043-Y-m-CB-vis-response)
\end{figure}


Figure \@ref(fig:ch043-Y-m-CB-vis-response) shows that there is almost perfect separation in the data for this subject, and that is giving the MLE algorithm trouble. It also has serious consequences on the estimated JND as the estimated JND for this subject is just 3ms which is suspect.


Of course one remedy for this is to pool observations together as we have done for the model in this iteration. The data is pooled together at the age group level and variation in the subjects' responses removes the separation. This isn't always ideal, as sometimes we are interested in studying the individuals within the experiment. If we can't get accurate inferences about the individual, then the results are not valid. The better solution is to use a hierarchical model. With a hierarchical model, individual estimates are shrunk towards the group mean, and so inferences about individuals may be made along with inferences about the group that contains them. We are interested only in the group level inferences right now.


Figure \@ref(fig:ch043-Iron-Intensive) shows the posterior distribution of psychometric functions for the visual TOJ data. Notice that there is almost no difference between the pre- and post-adaptation blocks.





\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{figures/ch043-Iron-Intensive} 

}

\caption{Posterior distribution of psychometric functions for the visual TOJ data. There is almost no visual difference between the pre- and post-adaptation blocks.}(\#fig:ch043-Iron-Intensive)
\end{figure}


Furthermore, as shown by the posterior retrodictions (figure \@ref(fig:ch043-obs-vs-retro-plot)), the model is not fully capturing the variation in the responses near the outer SOA values -- the posterior retrodictions are tight around SOA values near zero.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch043-obs-vs-retro-plot-1} 

}

\caption{Observed visual TOJ data compared to the posterior retrodictions. The retrodictions are not capturing the variation at the outer SOA values.}(\#fig:ch043-obs-vs-retro-plot)
\end{figure}


Why is the model having difficulty expressing the data? As it turns out, there is one more concept pertaining to psychometric experiments that has been left out until now, and that is a lapse in judgment. Not a lapse in judgment on our part, but the actual act of having a lapse while performing an experiment.


## Iteration 4 {#iter4}


**Pre-Model, Pre-Data**


_Conceptual Analysis_

A lapse in judgment can happen for any reason, and is assumed to be random and independent of other lapses. They can come in the form of the subject accidentally blinking during the presentation of a visual stimulus, or unintentionally pressing the wrong button to respond. Whatever the case is, lapses can have a significant affect on estimating the psychometric function.


**Post-Model, Pre-Data**


_Develop Model_

Lapses can be modeled as occurring independently at some fixed rate. Fundamentally this means that the underlying performance function, $F$, is bounded by some lower and upper lapse rate. This manifests as a scaling and translation of $F$. For a given lower and upper lapse rate $\lambda$ and $\gamma$, the performance function $\Psi$ is 


$$
\Psi(x; \alpha, \beta, \lambda, \gamma) = \lambda + (1 - \lambda - \gamma) F(x; \alpha, \beta)
$$


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch044-plot-pf-with-lapse-1} 

}

\caption{Psychometric function with lower and upper performance bounds.}(\#fig:ch044-plot-pf-with-lapse)
\end{figure}


In certain psychometric experiments, $\lambda$ is interpreted as the lower performance bound or the guessing rate. For example, in certain 2-AFC tasks, subjects are asked to respond which of two masses is heavier, and the correctness of their response is recorded. When the masses are the same, the subject can do no better than random guessing. In this task, the lower performance bound is assumed to be 50% as their guess is split between two choices. As the absolute difference in mass grows, the subject's correctness rate increases, though lapses can still happen. In this scenario, $\lambda$ is fixed at $0.5$ and the lapse rate $\gamma$ is a parameter in the model.


The model we are building for this data does not explicitly record correctness, so we do not give $\lambda$ the interpretation of a guessing rate. Since the data are recorded as proportion of positive responses, we instead treat $\lambda$ and $\gamma$ as lapse rates for negative and positive SOAs. But why should the upper and lower lapse rates be treated separately? A lapse in judgment can occur independently of the SOA, so $\lambda$ and $\gamma$ should be the same no matter what. With this assumption in mind, we can throw away $\gamma$ and assume that the lower and upper performance bounds are restricted by the same amount. I.e.


\begin{equation}
  \Psi(x; \alpha, \beta, \lambda) = \lambda + (1 - 2\lambda) F(x; \alpha, \beta)
  (\#eq:Psi)
\end{equation}


While we are throwing in a lapse rate, we will also ask the question of if different age groups have different lapse rates. To answer this (or rather have the model answer this), we include the new parameter $\lambda_{G[i]}$ into the model so that the lapse rate is estimated for each age group.


It's okay to assume that lapses in judgment are rare, and it's also true that the rate (or probability) of a lapse is bounded in the interval $[0, 1]$. Because of this, we put a $\mathrm{Beta(4, 96)}$ prior on $\lambda$ which puts 99% of the weight below $0.1$ and an expected lapse rate of $0.04$.


We could also set up the model so that information about the lapse rate is shared between age groups (i.e. multilevel), but we leave that as an exercise for the reader.


_Construct Summary Functions_

Since the fundamental structure of the linear model has changed, it is worth updating the summary function that computes the distribution of SOA values for a given response probability. Given equation \@ref(eq:Psi), the summary function $Q$ is


$$
Q(\pi; \alpha, \beta, \lambda) = \frac{1}{\exp(\beta)} \cdot \mathrm{logit}\left(\frac{\pi - \lambda}{1-2\lambda}\right) + \alpha
$$





**Post-Model, Post-Data**


_Fit Observed Data_

Because it is the visual data that motivated this iteration, we will continue using that data to fit the model and perform posterior retrodictive checks.





_Posterior Retrodictive Checks_

The plot for the distribution of psychometric functions is repeated one more time below (figure \@ref(fig:ch044-Screaming-Proton)). There is now visual separation between the pre- and post-adaptation blocks, with the latter exhibiting a higher slope, which in turn implies a reduced just noticeable difference which is consistent with the audiovisual data in the previous model.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{figures/ch044-Screaming-Proton} 

}

\caption{There is now a visual distinction between the two blocks unlike in the model without lapse rate. The lapse rate acts as a balance between steep slopes near the PSS and variation near the outer SOA values.}(\#fig:ch044-Screaming-Proton)
\end{figure}


As for the posterior retrodictions, the model is now better capturing the outer SOA variation. This can best be seen in the comparison of the younger adult pre-adaptation block of figure \@ref(fig:ch044-Insane-Metaphor).


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch044-Insane-Metaphor-1} 

}

\caption{The lapse rate produces posterior retrodictions that are visually more similar to the observed data than in the previous model, suggesting that the model is now just complex enough to capture the relevant details of the data generating process.}(\#fig:ch044-Insane-Metaphor)
\end{figure}


We can also reintroduce the package `loo` from the previous chapter to evaluate the predicted predictive performance of the model with lapse rate to the model without the lapse rate as a way of justifying its inclusion. Subjectively the lapse rate model is already doing better, but it is necessary to have principled comparisons. Table \@ref(tab:ch044-Straw-Epsilon) shows the comparison.





When extracting the PSIS-LOO values, `loo` warns about some Pareto k diagnostic values that are slightly high. Let's take a look at the summary:



```r
pareto_k_table(l044)
#> Pareto k diagnostic values:
#>                          Count Pct.    Min. n_eff
#> (-Inf, 0.5]   (good)     2249  100.0%  1457      
#>  (0.5, 0.7]   (ok)          1    0.0%  794       
#>    (0.7, 1]   (bad)         0    0.0%  <NA>      
#>    (1, Inf)   (very bad)    0    0.0%  <NA>      
#> 
#> All Pareto k estimates are ok (k < 0.7).
```


There is one observation in the data set that has a $k$ value between $0.5$ and $0.7$. This means that the estimated Pareto distribution has infinite variance, but practically it is still usable for estimating predictive performance.


\begin{table}[!h]

\caption{(\#tab:ch044-Straw-Epsilon)Model without lapse rate compared to one with lapse rate.}
\centering
\begin{tabular}[t]{lrrrrr}
\toprule
Model & elpd\_diff & se\_diff & elpd\_loo & p\_loo & se\_p\_loo\\
\midrule
Lapse & 0.0 & 0.00 & -1001 & 19.22 & 1.902\\
No Lapse & -259.4 & 31.92 & -1260 & 23.10 & 2.259\\
\bottomrule
\end{tabular}
\end{table}


The model with the lapse rates proves to be better than the model without lapse rates as measured by PSIS-LOO. Surprisingly the effective number of parameters _shrinks_ after including the lapse rates. We can now perform one last iteration of the model by including the subject level estimates. Even though we're only interested in making inferences at the group level, including the subject level might improve predictive performance.


## Iteration 5 {#iter5}

The only change in this iteration is the addition of the subject level parameters for the slope and intercept. 


**Post-Model, Post-Data**





_Diagnose Posterior Fit_

There is only one divergent transition for this model indicating no issues with the algorithm configuration. Checking the trace plot for the multilevel variance terms also indicates no problems with the sampling.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch045-Mysterious-Neptune-1} 

}

\caption{The multilevel model with lapse and subject-level terms fits efficiently with no issues.}(\#fig:ch045-Mysterious-Neptune)
\end{figure}


This model also utilizes thinning while fitting for data saving reasons. As such the autocorrelation between samples is reduced and the model achieves a high $N_{\mathrm{eff}}/N$ ratio (figure \@ref(fig:ch045-Nocturnal-Temple)).


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-workflow_files/figure-latex/ch045-Nocturnal-Temple-1} 

}

\caption{The model with lapse rates and subject-level parameters achieves a sampling efficiency partially due to thinning.}(\#fig:ch045-Nocturnal-Temple)
\end{figure}


_Posterior Predictive Comparison_




In lieu of posterior retrodictions (which would) appear similar to those of the last iteration, we simply compare the model with subject-level parameters to the one without. There are a handful of observations in the subject-level model that a Pareto $k$ value of $0.7$ which indicates impractical convergence rates and unreliable Monte Carlo error estimates. For more accurate estimation of predictive performance, $k$-fold CV or LOOCV is recommended.



```
#> Pareto k diagnostic values:
#>                          Count Pct.    Min. n_eff
#> (-Inf, 0.5]   (good)     2174  96.6%   285       
#>  (0.5, 0.7]   (ok)         63   2.8%   170       
#>    (0.7, 1]   (bad)        13   0.6%   98        
#>    (1, Inf)   (very bad)    0   0.0%   <NA>
```


<!--
\begin{table}[!h]

\caption{(\#tab:ch045-Hideous-Compass)High Pareto k diagnostic values for the subject level lapse model.}
\centering
\begin{tabular}[t]{rrrlllr}
\toprule
soa & n & k & sid & trial & age\_group & k\_value\\
\midrule
-50 & 3 & 0 & M-m-BT & post1 & middle\_age & 0.8103\\
0 & 5 & 3 & M-f-DB & pre & middle\_age & 0.7948\\
75 & 5 & 5 & M-f-HG & pre & middle\_age & 0.7817\\
125 & 3 & 3 & O-m-GB & post1 & older\_adult & 0.7799\\
0 & 3 & 2 & Y-m-CB & post1 & young\_adult & 0.7743\\
\addlinespace
-100 & 5 & 0 & Y-f-IV & pre & young\_adult & 0.7645\\
75 & 3 & 3 & O-f-EM & post1 & older\_adult & 0.7389\\
0 & 5 & 2 & O-m-DC & pre & older\_adult & 0.7291\\
0 & 5 & 3 & Y-f-CM & pre & young\_adult & 0.7266\\
50 & 5 & 0 & O-m-BC & pre & older\_adult & 0.7074\\
\addlinespace
-50 & 5 & 0 & M-f-HG & pre & middle\_age & 0.7048\\
-50 & 5 & 0 & Y-m-PB & pre & young\_adult & 0.7031\\
75 & 5 & 5 & Y-m-DD & pre & young\_adult & 0.7005\\
\bottomrule
\end{tabular}
\end{table}
-->


Proceeding on to model comparison, including the subject-level information significantly improves the ELPD, and even though there are over 100 parameters in the model (slope and intercept for each of the 45 subjects), the effective number of parameters is much less. Since this new model is capable of making inferences at both the age group level and the subject level, we use it for drawing inferences in the results chapter.


\begin{table}[!h]

\caption{(\#tab:ch045-Deserted-Fish)Model without subjects compared to one with subjects.}
\centering
\begin{tabular}[t]{lrrrrr}
\toprule
Model & elpd\_diff & se\_diff & elpd\_loo & p\_loo & se\_p\_loo\\
\midrule
With Subjects & 0.00 & 0.00 & -925.1 & 75.57 & 5.432\\
Without Subjects & -75.96 & 19.13 & -1001.1 & 19.22 & 1.902\\
\bottomrule
\end{tabular}
\end{table}


One concern comes up when it comes to LOOCV and multilevel models. What does it mean to leave _one_ out? Should one subject be left out? One age group? Just one observation? With more levels in a model, more careful considerations must be taken when it comes to estimating predictive performance.
