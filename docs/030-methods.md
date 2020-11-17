


# Background {#methods}



## Fitting the psychometric function using GLMs {#glms}


Psychometric functions are commonly fit using generalized linear models which allows for the linear model to be related to the response variable via a link function, which for psychometric functions comes from the family of S-shaped curves called a sigmoid.


Commonly GLMs are fit using maximum likelihood estimation (MLE). The outcome of a single experiment can be represented as the result of a Bernoulli trial. The psychometric function, $F(x; \theta)$, determines the probability that the outcome is 1:


\begin{align*}
Y &\sim \textrm{Bernoulli}(\pi) \\
\pi &= P(Y=1 \vert x; \theta) = F(x; \theta)
\end{align*}


If $P(Y=1 | x; \theta) = F(x;\theta)$, then $P(Y = 0 | x; \theta) = 1 - F(x;\theta)$, and hence the probability of an outcome is:


\begin{equation}
  P(Y=y | x; \theta) = F(x;\theta)^y(1-F(x;\theta))^{1-y}
  (\#eq:bernproby)
\end{equation}


The likelihood $\mathcal{L}$ of observing a set of independent and identically distributed data given the parameterization $\theta$ is determined by taking the product of the probabilities for each datum:


\begin{equation}
  \begin{split}
    \mathcal{L} &= \prod_{i}^{N} P(y_i | x_i; \theta) \\
    &= \prod_{i}^{N}F(x_i;\theta)^{y_i}(1-F(x_i;\theta))^{1-y_i}
  \end{split}
  (\#eq:bernlik)
\end{equation}


For some $\theta$, $\mathcal{L}$ achieves a maximum value, so maximum likelihood estimation determines the parameters that maximizes the likelihood of the observed data. Equation \@ref(eq:bernlik) is commonly expressed in terms of its logarithm as a function of $\theta$:


\begin{equation}
  \ln \mathcal{L}(\theta | y, x) = \sum_{i}^{N} y_i \ln\left(F(x_i;\theta)\right) + (1-y_i) \ln\left(1 - F(x_i;\theta))\right)
  (\#eq:bernloglik)
\end{equation}


The classical approach is to differentiate \@ref(eq:bernloglik) with respect to $\theta$, set the equation equal to $0$, and solve for $\theta$:


\begin{equation}
  \frac{d}{d\theta} \ln \mathcal{L}(\theta) = 0
  (\#eq:ddthetaloglik)
\end{equation}


However, no closed form expression exists for the solution to \@ref(eq:ddthetaloglik), and so numerical root finding methods such as gradient descent are used to iteratively find the maximum likelihood solution. The likelihood function, $\mathcal{L}(\theta | y)$, also has a connection to Bayes' Theorem:


\begin{equation}
  \mathcal{L}(\theta | y) = \frac{P(y | \theta) P(\theta)}{P(y)}
  (\#eq:bayeslik)
\end{equation}


In \@ref(eq:bayeslik), $P(y | \theta)$ is the likelihood of the data given $\theta$, $P(\theta)$ is the prior distribution for the parameter $\theta$, and $P(y)$ is the probability of the data averaged over the parameter space. When the prior distribution is uniform over the parameter space, then the Bayesian _maximum a posteriori_ (MAP) estimate coincides with the maximum likelihood estimate.


There are common situations where MLE fails such as complete separation in the data. This is when the positive class can be separated from the negative class by a set of linear predictors (shown in figure \@ref(fig:ch030-comp-sep)).


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{030-methods_files/figure-latex/ch030-comp-sep-1} 

}

\caption{Example of complete separation in the data. All of the `0`-responses can be separated from the `1`-responses by some value of `x` between `-1` and `0`.}(\#fig:ch030-comp-sep)
\end{figure}


For a slope-intercept model, the MLE for the slope is infinity, and the location is undefined. Figure \@ref(fig:ch030-mle-comp-sep) displays a grid of log-likelihoods for a range of scale (inverse slope) and location parameters. The log-likelihood increases to zero as the scale decreases to zero (slope increases to infinity). Numerical root finding methods will converge after a finite number of iterations -- usually by a stopping condition such as the difference in log-likelihoods between steps.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{030-methods_files/figure-latex/ch030-mle-comp-sep-1} 

}

\caption{Grid of log-likelihoods for the completely separable data. The log-likelihood increases to zero as the scale decreases to zero (slope increases to infinity). For smaller slopes, the MLE for the location is `0.5` -- the median of the inner-most datum from each class. At larger slopes, the MLE for the location matters little.}(\#fig:ch030-mle-comp-sep)
\end{figure}


When the separable data from above is fit using `R`'s `glm` function, there is a warning about fitted probabilities being $0$ or $1$, indicating that the slope is very steep.



```r
fit <- glm(y ~ x, family = binomial("logit"))
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
coefficients(fit)
#> (Intercept)           x 
#>       22.89       45.97
```


The coefficients of the linear predictor are in slope-intercept form ($23 + 46 x$). Rearranging the coefficients into location-scale form yields:


\begin{equation}
  \theta = \frac{x - (-0.5)}{1/46}
  (\#eq:rglmmle)
\end{equation}


However, this is not the true maximum likelihood estimate. Table \@ref(tab:ch030-Quality-Surreal-Street) shows that the log-likelihood is still increasing as the scale decreases to zero. The change between $-2.05\times 10^{-10}$ and $-3.86\times10^{-22}$ is very large on a relative scale, but computers cannot tell the difference on an absolute scale. The precision of most modern computers is around $10^{-15}$, and anything smaller is treated as numerically zero.


\begin{table}[!h]

\caption{(\#tab:ch030-Quality-Surreal-Street)Log-likelihood estimates for different scale parameters. *Indicates the MLE solution from R.}
\centering
\begin{tabular}[t]{ll}
\toprule
Scale & Log-Likelihood\\
\midrule
1/10 & -1.34e-02\\
1/46* & -2.05e-10\\
1/100 & -3.86e-22\\
1/1000 & -1.42e-217\\
\bottomrule
\end{tabular}
\end{table}


Using a weakly-informative prior for the slope instead of the non-informative uniform distribution will allow for a proper MAP estimate, but care should be taken to consider what prior is appropriate, or why the data is separable in the first place. If an experiment can be improved to avoid the situation, that should be the first action.


## Multilevel modeling


In classical regression, a simple single-level slope-intercept model can be specified as:


\begin{equation}
y_i = \alpha + \beta x_i + \epsilon_i
(\#eq:single-level-fixed)
\end{equation}


The slope and intercept is fixed for all observations in the data set. If there is a categorical variable with $J$ levels, then a varying-slope varying-intercept (or simply varying effects) model can be specified as:


\begin{equation}
y_i = \alpha_{j[i]} + \beta_{j[i]} x_i + \epsilon_i
(\#eq:single-level-fixed-varying)
\end{equation}


where $j[i]$ indexes the group for observation $i$. In a multilevel model, the coefficients are modeled by a separate regression:


\setstretch{1.0}
\begin{equation}
\begin{split}
y_i &= \alpha_{j[i]} + \beta_{j[i]} x_i \\
\alpha_j &\sim \mathcal{N}(a_0 + a_1 u_j, \sigma_{\alpha}^2) \\
\beta_j &\sim \mathcal{N}(b_0 + b_1 u_j, \sigma_{\beta}^2) \\
\end{split}
(\#eq:multilevel-varying)
\end{equation}
\setstretch{2.0}


In \@ref(eq:multilevel-varying), $x$ is a population level predictor, and $u$ is a group level predictor. Equations \@ref(eq:single-level-fixed) and \@ref(eq:single-level-fixed-varying) represent the two extremes of excluding a categorical variable from a model (complete pooling) and fitting a regression coefficient for each level in the categorical variable (no pooling). Multilevel modeling provides a compromise between these two extremes, resulting in partial pooling estimates.


For a simple intercept-only model, the partial pooling estimate of the intercept, $\hat{\alpha}_j$ is a weighted average of the mean of the of the observations in the group (no pooling estimate, $\bar{y}_j$) and the mean over all groups (complete pooling estimate, $\bar{y}$):


$$
\hat{\alpha}_j \approx \frac{\frac{n_j}{\sigma_y^2} \bar{y}_j + \frac{1}{\sigma_\alpha^2} \bar{y}}{\frac{n_j}{\sigma_y^2} + \frac{1}{\sigma_\alpha^2}}
$$


where $n_j$ is the number of observations in group level $j$, $\sigma_y^2$ is the within-group variance, and $\sigma_\alpha^2$ is the variance between the group level averages. As the between-group variance goes to infinity (or as $n_j \rightarrow \infty$), the partial pooling estimates approach the no pooling estimates. When there are fewer samples, the partial pooling estimate is closer to the overall average. In this way, partial pooling reflects the relative information contained within each group. For an in-depth introduction to multilevel modeling, see @gelman2006data.


## Hamiltonian Monte Carlo and NUTS


We will be using `Stan` for model fitting throughout this paper. `Stan` allows for MCMC sampling of Bayesian models using a variant of Hamiltonian Monte Carlo called the No-U-Turn sampler (NUTS). HMC can be though of as a physics simulation: a massless "particle" is imparted with a random direction and some amount of kinetic energy in a probability field, and is stopped after a number of steps, $L$, called leapfrog steps. The stopping point is the new proposal sample. The NUTS algorithm removes the need for leapfrog steps by stopping automatically when the particle begins to double back and retrace its steps [@hoffman2014no]. This sampling scheme has a much higher rate of accepted samples, and also comes with many built-in diagnostic tools that let us know when the sampler is having trouble efficiently exploring the posterior.


The NUTS algorithm samples in two phases: a warm-up phase and a sampling phase. During the warm-up phase, the sampler is automatically tuning three internal parameters that can significantly affect the sampling efficiency. The sum of the potential energy and the kinetic energy of the system is called the Hamiltonian, and is conserved along the trajectory of the particle [@stanref]. The path that the particle takes is a discrete approximation to the actual path where the position of the particle is updated in small steps called leapfrog steps (see @leimkuhler2004simulating for a detailed explanation of the leapfrog algorithm). A divergent transition happens when the simulated trajectory is far from the true trajectory as measured by the Hamiltonian.


## Non-centered parameterization


Because HMC is a physics simulation, complicated geometry or posteriors with steep slopes can be difficult to traverse if the step size is too course. The solution is to explore a simpler geometry, and then transform the sample into the target distribution. Reparameterization is especially important for hierarchical models. For `Stan`, sampling from a standard normal or uniform distribution is very easy, and so the non-centered parameterization can alleviate divergent transitions. Here we present three reparameterizations that we use in the next chapter.


**Non-centered Gaussian distribution**


\setstretch{1.0}
\begin{equation}
  \begin{split}
    X &\sim \mathcal{N}(\mu, \sigma^2)
  \end{split}
\quad \Longrightarrow \quad
  \begin{split}
    Z &\sim \mathcal{N}(0, 1^2) \\
    X &= \mu + \sigma \cdot Z
  \end{split}
(\#eq:nc-normal)
\end{equation}
\setstretch{2.0}


**Non-centered Log-Normal distribution**


\setstretch{1.0}
\begin{equation}
  \begin{split}
    X &\sim \mathrm{Lognormal}(\mu, \sigma^2)
  \end{split}
\quad \Longrightarrow \quad
  \begin{split}
    Z &\sim \mathcal{N}(0, 1^2) \\
    X &= \exp\left(\mu + \sigma \cdot Z\right)
  \end{split}
(\#eq:nc-lognormal)
\end{equation}
\setstretch{2.0}


**Non-centered Cauchy distribution**


\setstretch{1.0}
\begin{equation}
  \begin{split}
    X &\sim \mathrm{Cauchy}(\mu, \tau)
  \end{split}
\quad \Longrightarrow \quad
  \begin{split}
    U &\sim \mathcal{U}\left(-\frac{\pi}{2}, \frac{\pi}{2}\right) \\
    X &= \mu + \tau \cdot \tan(U)
  \end{split}
(\#eq:nc-cauchy)
\end{equation}
\setstretch{2.0}


## Methods for model checking {#model-checking}


Below is the 8 Schools data [@gelman2013bayesian] which is a standard textbook example for introducing multilevel modeling. Here we use it to illustrate essential MCMC model checking tools.



```r
schools_dat <- list(
  J = 8,
  y = c(28,  8, -3,  7, -1,  1, 18, 12),
  sigma = c(15, 10, 16, 11,  9, 11, 10, 18)
)
```














**Trace Plots.** Trace plots have been used since the conception of MCMC to assess chain sampling efficiency and quality. They are visual aids that let the practitioner asses the qualitative health of the chains, looking for properties such as autocorrelation, heteroskedacity, non-stationarity, and convergence. Healthy chains are well-mixed and stationary. It's often better to run more chains during the model building process so that issues with mixing and convergence can be diagnosed sooner. One unhealthy chain can be indicative of a poorly specified model. The addition of more chains also contributes to the estimation of the split $\hat{R}$ statistic (discussed below). Figure \@ref(fig:ch030-Brave-Moose) shows what a set of healthy chains looks like -- the chains are nearly indistinguishable, fluctuate around the same mean, and do not show any long periods of being in the same location.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{030-methods_files/figure-latex/ch030-Brave-Moose-1} 

}

\caption{An example of healthy chains.}(\#fig:ch030-Brave-Moose)
\end{figure}


As the number of parameters in a model grows, it becomes exceedingly tedious to check the trace plots of all parameters, and so numerical summaries are helpful to flag potential issues within the model.


**R-hat Statistic.** The most common summary statistic for chain health is the potential scale reduction factor [@gelman1992inference] that measures the ratio of between chain variance and within chain variance. When the two have converged, the ratio is one. We've shared examples of healthy chains which would also have healthy $\hat{R}$ values, but it's valuable to also share an example of a bad model. Below is the 8 Schools example [@gelman2013bayesian] which is a classical example for introducing Stan and testing the operating characteristics of a model.


The initial starting parameters for this model are intentionally set to vary between $-10$ and $10$ -- in contrast to the default range of $(-2, 2)$ -- and with only a few samples drawn in order to artificially drive up the split $\hat{R}$ statistic. The model is provided as supplementary code in the [appendix](#code).



```r
fit_cp <- sampling(schools_cp, data = schools_dat, refresh = 0,
                   iter = 40, init_r = 10, seed = 671254821)
```


`Stan` warns about many different issues with this model, but the R-hat is the one of interest. The largest is $1.71$ which is incredibly large. Gelmen suggests using a threshold of $1.10$ to flag unhealthy chains.



\begin{center}\includegraphics[width=0.85\linewidth]{030-methods_files/figure-latex/ch030-Rocky-Test-1} \end{center}


These chains do not look good at all -- they have not converged to a stationary distribution. . The $\hat{R}$ values are listed in table \@ref(tab:ch030-Ninth-Finger).


\begin{table}[!h]

\caption{(\#tab:ch030-Ninth-Finger)Split R-hat values from the 8 Schools example.}
\centering
\begin{tabular}[t]{lr}
\toprule
Parameter & Rhat\\
\midrule
mu & 1.709\\
tau & 1.169\\
\bottomrule
\end{tabular}
\end{table}


To calculate the (non split) $\hat{R}$, first calculate the between-chain variance, and then the average chain variance. For $M$ independent Markov chains, $\{\theta_1, \ldots \theta_M\}$, with $N$ samples each, the between-chain variance is:


\setstretch{1.0}
$$
B = \frac{N}{M-1}\sum_{m=1}^{M}\left(\bar{\theta}_m - \bar{\theta}\right)^2
$$


where


$$
\bar{\theta}_m = \frac{1}{N}\sum_{n=1}^{N}\theta_{m}^{(n)}
$$


and


$$
\bar{\theta} = \frac{1}{M}\sum_{m=1}^{M}\bar{\theta}_m
$$


The within-chain variance, $W$, is the variance averaged over all the chains:


$$
W = \frac{1}{M}\sum_{m=1}^{M} s_{m}^2
$$


where


$$
s_{m}^2 = \frac{1}{N-1}\sum_{n=1}^{N}\left(\theta_{m}^{(n)} - \bar{\theta}_m\right)^2
$$


The weighted mixture of the within-chain and cross-chain variation is:


$$
\hat{var} = \frac{N-1}{N} W + \frac{1}{N} B
$$


and finally the $\hat{R}$ statistic is:


$$
\hat{R} = \sqrt{\frac{\hat{var}}{W}}
$$


Here is the calculation in `R`:



```r
param <- "mu"
theta <- p_cp[,,param]
N     <- nrow(theta)
M     <- ncol(theta)

theta_bar_m <- colMeans(theta)
theta_bar   <- mean(theta_bar_m)

B <- N / (M - 1) * sum((theta_bar_m - theta_bar)^2)
s_sq_m <- apply(theta, 2, var)

W <- mean(s_sq_m)
var_hat <- W * (N - 1) / N + B / N

(mu_Rhat <- sqrt(var_hat / W))
#> [1] 1.409
```
\setstretch{2.0}


The $\hat{R}$ statistic is smaller than the split $\hat{R}$ value provided by `Stan`. This is a consequence of steadily increasing or decreasing chains. The split value does what it sounds like, and splits the samples from the chains in half -- effectively doubling the number of chains and halving the number of samples per chain. In this way, the measure is more robust in detecting unhealthy chains. This also highlights the utility in using both visual and statistical tools to evaluate models. Here is the calculation of the split $\hat{R}$:



```r
param <- "mu"
theta_tmp <- p_cp[,,param]
theta <- cbind(theta_tmp[1:10,], theta_tmp[11:20,])
N     <- nrow(theta)
M     <- ncol(theta)

theta_bar_m <- colMeans(theta)
theta_bar   <- mean(theta_bar_m)

B <- N / (M - 1) * sum((theta_bar_m - theta_bar)^2)
s_sq_m <- apply(theta, 2, var)

W <- mean(s_sq_m)
var_hat <- W * (N - 1) / N + B / N

(mu_Rhat <- sqrt(var_hat / W))
#> [1] 1.709
```


We've successfully replicated the calculation of the split $\hat{R}$. @vehtari2020rank propose an improved rank-normalized $\hat{R}$ for assessing the convergence of MCMC chains, and also suggest using a threshold of $1.01$.


**Effective Sample Size.** Samples from Markov Chains are typically autocorrelated, which can increase uncertainty of posterior estimates. The solution is generally to reparameterize the model to avoid steep log-posterior densities. When the HMC algorithm is exploring difficult geometry, it can get stuck in regions of high densities, which means that there is more correlation between successive samples. Equation \@ref(eq:schools-ncp) shows the centered (left) and non-centered (right) parameterization of the 8-Schools model, and the benefit of reparameterization is conveyed by the ratio of effective sample size to actual sample size in figure \@ref(fig:ch030-Timely-Nitrogen).


\begin{equation}
  \begin{split}
    \sigma &\sim \mathcal{U}(0, \infty) \\
    \mu &\sim \mathcal{N}(0, 10) \\
    \tau &\sim \mathrm{HalfCauchy}(0, 10) \\
    \theta &\sim \mathcal{N}(\mu, \tau) \\
    y &\sim \mathcal{N}(\theta, \sigma)
  \end{split}
\quad \Longrightarrow \quad
  \begin{split}
    \sigma &\sim \mathcal{U}(0, \infty) \\
    \mu &\sim \mathcal{N}(0, 10) \\
    \tau &\sim \mathrm{HalfCauchy}(0, 10) \\
    \eta &\sim \mathcal{N}(0, 1) \\
    \theta &= \mu + \tau \cdot \eta \\
    y &\sim \mathcal{N}(\theta, \sigma)
  \end{split}
(\#eq:schools-ncp)
\end{equation}





\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{030-methods_files/figure-latex/ch030-Timely-Nitrogen-1} 

}

\caption{Ratio of N\_eff to actual sample size. Low ratios imply high autocorrelation which can be alleviated by reparameterizing the model or by thinning.}(\#fig:ch030-Timely-Nitrogen)
\end{figure}


As the strength of autocorrelation generally decreases at larger lags, a simple prescription to decrease autocorrelation between samples and increase the effective sample size is to use thinning. Thinning means saving every $k^{th}$ sample and throwing the rest away. If one desired to have 2000 posterior draws, it could be done in two of many possible ways


- Generate 2000 draws after warmup and save all of them
- Generate 10,000 draws after warmup and save every $5^{th}$ sample. 


Both will produce 2000 samples, but the method using thinning will have less autocorrelation and a higher effective number of samples. Though it should be noted that generating 10,000 draws and saving all of them will have a higher number of effective samples than the second method with thinning, so thinning should only be favored to save memory.


**Divergent Transitions.** Unlike the previous tools for algorithmic faithfulness which can be used for any MCMC sampler, information about divergent transitions is intrinsic to Hamiltonian Monte Carlo. Recall that the HMC and NUTS algorithm can be imagined as a physics simulation of a particle in a potential energy field, and a random momentum is imparted on the particle. The sum of the potential energy and the kinetic energy of the system is called the Hamiltonian, and is conserved along the trajectory of the particle [@stanref]. The path that the particle takes is a discrete approximation to the actual path where the position of the particle is updated in small steps called leapfrog steps (see @leimkuhler2004simulating for a detailed explanation of the leapfrog algorithm). A divergent transition happens when the simulated trajectory is far from the true trajectory as measured by the Hamiltonian.


A few divergent transitions is not indicative of a poorly performing model, and often divergent transitions can be mitigated by reducing the step size and increasing the adapt delta parameter. On the other hand, a bad model may never be improved just by tweaking some parameters. This is the folk theorem of statistical computing - if there is a problem with the sampling, blame the model, not the algorithm.


Divergent transitions are never saved in the posterior samples, but they are saved internally to the `Stan` fit object and can be compared against good samples. Sometimes this can give insight into which parameters and which regions of the posterior the divergent transitions are coming from.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{030-methods_files/figure-latex/ch030-Hot-Locomotive-1} 

}

\caption{Divergent transitions highlighted for some parameters from the centered parameterization of the eight schools example.}(\#fig:ch030-Hot-Locomotive)
\end{figure}


From figure \@ref(fig:ch030-Hot-Locomotive) we can see that most of the divergent transitions occur when the variance term $\tau$ is close to zero. This is common for multilevel models, and illustrates why non-centered parameterization is so important. We discuss centered and non-centered parameterization in the next chapter.


## Estimating predictive performance


All models are wrong, but some are useful. This quote is from George Box, and it is a popular quote that statisticians like to throw around. All models are wrong because it is nearly impossible to account for the minutiae of every process that contributes to an observed phenomenon, and often trying to results in poorer performing models. Also it is never truly possible to prove that a model is correct. At best the scientific method can falsify certain hypotheses, but it cannot ever determine if a model is universally correct. That does not matter. What does matter is if the model is useful and can make accurate predictions.


Why is predictive performance so important? Consider five points of data (figure \@ref(fig:ch030-Moving-Moose)). They have been simulated from some polynomial equation of degree less than five, but with no more information other than that, how can the best polynomial model be selected?


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{030-methods_files/figure-latex/ch030-Moving-Moose-1} 

}

\caption{Five points from a polynomial model.}(\#fig:ch030-Moving-Moose)
\end{figure}

One thing to try is fit a handful of linear models, check the parameter's p-values, the $R^2$ statistic, and perform other goodness of fit tests, but there is a problem. As the degree of the polynomial fit increases, the $R^2$ statistic will always increase. In fact with five data points, a fourth degree polynomial will fit the data perfectly (figure \@ref(fig:ch030-Olive-Screwdriver)).


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{030-methods_files/figure-latex/ch030-Olive-Screwdriver-1} 

}

\caption{Data points with various polynomial regression lines.}(\#fig:ch030-Olive-Screwdriver)
\end{figure}


If a $6^{th}$ point were to be added -- a new observation -- which of the models would be expected to predict best? Can it be estimated which model will predict best before testing with new data? One guess is that the quadratic or cubic model will do well because because the linear model is potentially _underfit_ to the data and the quartic is _overfit_ to the data. Figure \@ref(fig:ch030-Cold-Fish) shows the new data point from the polynomial model. Now the linear and cubic models are trending in the wrong direction. The quadratic and quartic models are both trending down, so perhaps they may be the correct form for the model.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{030-methods_files/figure-latex/ch030-Cold-Fish-1} 

}

\caption{The fitted polynomial models with a new observation.}(\#fig:ch030-Cold-Fish)
\end{figure}


Figure \@ref(fig:ch030-Strawberry-Swallow) shows the 80% and 95% prediction intervals for a new observation given $x = 1$ as well as the true outcome as a dashed line at $y = 1.527$. The linear model has the smallest prediction interval (PI), but completely misses the target. The remaining three models all include the observed value in their 95% PIs, but the quadratic model has the smallest PI of the three. The actual data generating polynomial is


\begin{align*}
y &\sim \mathcal{N}(\mu, 1^2) \\
\mu &= -0.5(x - 2)^2 + 2
\end{align*}





\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{030-methods_files/figure-latex/ch030-Strawberry-Swallow-1} 

}

\caption{95\% Prediction intervals for the four polynomial models, as well as the true value (dashed line).}(\#fig:ch030-Strawberry-Swallow)
\end{figure}


The best fit to the observed data is the quartic model, but it is too variable and doesn't capture the regular features of the data, so it does poorly for the out-of-sample prediction. The linear model suffers as well by being less biased and too inflexible to capture the structure of the data. The quadratic and cubic are in the middle, but the quadratic does well and makes fewer assumptions about the data. The quadratic model is just complex enough to predict well while making fewer assumptions. Information criteria is a way of weighing the prediction quality of a model against its complexity, and is arguably a better system for model selection/comparison than other goodness-of-fit statistics such as $R^2$ or p-values.


A technique to evaluate predictive performance is cross validation, where the data is split into training data and testing data. The model is fit to the training data, and then predictions are made with the testing data and compared to the observed values. This can often give a good estimate for out-of-sample prediction error. Cross validation can be extended into k-fold cross validation. The idea is to fold the data into $k$ disjoint partitions, and predict partition $i$ using the rest of the data to train on. The prediction error of the $k$-folds can then be averaged over to get an estimate for out-of-sample prediction error.


Taking $k$-fold CV to the limit by letting $k$ equal the number of observations results in leave-one-out cross validation (LOOCV), where for each observation in the data, the model is fit to the remaining data and predicted for the left out observation. $k$-fold cross validation requires fitting the model $k$ times, which can be computationally expensive for complex Bayesian models. Thankfully there is a way to approximate LOOCV without having to refit the model many times.


**Estimating cross validation error via Pareto-Smoothed-Importance Sampling**. LOOCV and many other evaluation tools such as the widely applicable information criterion (WAIC) rest on the log-pointwise-predictive-density (lppd), which measures deviance from some "true" probability distribution. Typically we don't have the analytic form of the predictive posterior density, so instead we use $S$ MCMC draws to approximate the lppd [@vehtari2017practical]:


\begin{equation}
\mathrm{lppd}(y, \Theta) = \sum_i \log \frac{1}{S} \sum_s p(y_i | \Theta_s)
(\#eq:lppd)
\end{equation}


To estimate LOOCV, the relative "importance" of each observation must be computed. Certain observations have more influence on the posterior distribution, and so have more impact on the posterior if they are removed. By omitting a sample, the relative importance weight can be measured by the lppd. This omitted calculation is known as the out-of-sample lppd. For each omitted $y_i$,


$$
\mathrm{lppd}_{CV} = \sum_i \frac{1}{S} \sum_s \log p(y_{i} | \theta_{-i,s})
$$


The method of using weights to estimate the cross-validation is called Pareto-Smoothed Importance Sampling Cross-Validation (PSIS). Pareto-smoothing is a technique for making the importance weight more reliable. Each sample $s$ is re-weighted by the inverse of the probability of the omitted observation:


$$
r(\theta_s) = \frac{1}{p(y_i \vert \theta_s)}
$$


Then the importance sampling estimate of the out-of-sample lppd is calculated as:


$$
\mathrm{lppd}_{IS} = \sum_{i}\log \frac{\sum_{s} r(\theta_s) p(y_i \vert \theta_s)}{\sum_{s} r(\theta_s)}
$$


However, the importance weights can have a heavy right tail, and so they can be stabilized by using the Pareto distribution [@vehtari2015pareto]. The distribution of weights theoretically follow a Pareto distribution, so the larger weights can be used to estimate the generalized Pareto distribution


$$
p(r; \mu, \sigma, k) = \frac{1}{\sigma} \left(1 + \frac{k (r - \mu)}{\sigma}\right)^{-(1/k + 1)}
$$


where $\mu$ is the location, $\sigma$ is the scale, and $k$ is the shape. Then the estimated distribution is used to smooth the weights. A side-effect of using PSIS is that the estimated value of $k$ can be used as a diagnostic tool for a particular observation. For $k>0.5$, the Pareto distribution will have infinite variance, and a really heavy tail. If the tail is very heavy, then the smoothed weights are harder to trust. In theory and in practice, PSIS works well as long as $k < 0.7$ [@vehtari2015pareto].


There is an `R` package called `loo` that can compute the expected log-pointwise-posterior-density (ELPD) using PSIS-LOO, as well as the estimated number of effective parameters and LOO information criterion [@R-loo]. For the part of the researcher, the log-likelihood of the observations must be computed. This can be calculated in the `generated quantities` block of a `Stan` program, and it is standard practice to name the log-likelihood as `log_lik` in the model. An example of calculating the log-likelihood for the eight schools data in `Stan` is:


\setstretch{1.0}
```
generated quantities {
  vector[J] log_lik;
  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
  }
}
```
\setstretch{2.0}


Models can be compared simply using `loo::loo_compare`. It estimates the ELPD and its standard error, then calculates the relative differences between all the models. The model with the highest ELPD is predicted to have the best out-of-sample predictions. The comparison of four polynomial models from the earlier example is shown below.







```r
comp <- loo_compare(linear, quadratic, cubic, quartic)
```


\begin{table}[!h]
\centering
\begin{tabular}{lrrrr}
\toprule
Model & elpd\_diff & se\_diff & p\_loo & looic\\
\midrule
Cubic & 0.000 & 0.000 & 2.617 & 18.57\\
Quartic & -0.374 & 0.475 & 3.710 & 19.32\\
Quadratic & -2.657 & 1.799 & 3.860 & 23.89\\
Linear & -2.855 & 1.693 & 2.631 & 24.28\\
\bottomrule
\end{tabular}
\end{table}


This comparison is unreliable since there are only five data points to estimate the predictive performance. This assertion is backed by the difference in ELPD and the standard error of the differences -- the standard error is as large or larger than the difference.


## A modern principled bayesian modeling workflow


A principled workflow is a method of employing domain expertise and statistical knowledge to iteratively build a statistical model that satisfies the constraints and goals set forth by the researcher. Many other workflow and model checking techniques are given without context for when they are appropriate, and according to @betancourt2020, this leaves "practitioners to piece together their own model building workflows from potentially incomplete or even inconsistent heuristics." For any given problem, there is not, nor should there be, a default set of steps to take to get from data exploration to predictive inferences. Rather, consideration must be given to domain expertise and the questions that one is trying to answer with the statistical model.


Because everyone asks different questions, the value of a model is not in how well it ticks the boxes of goodness-of-fit checks, but in how consistent it is with domain expertise and its ability to answer the unique set of questions. Betancourt suggests answering four questions to evaluate a model by, summarized in table \@ref(tab:ch030-Confidential-Proton).


\begin{table}[!h]

\caption{(\#tab:ch030-Confidential-Proton)Questions for model evaluation.}
\centering
\begin{tabular}[t]{>{\raggedright\arraybackslash}p{1.75in}>{\raggedright\arraybackslash}p{3.25in}}
\toprule
Evaluation & Question\\
\midrule
\cellcolor{gray!6}{1. Domain Expertise Consistency} & \cellcolor{gray!6}{Is our model consistent with our domain expertise?}\\
2. Computational Faithfulness & Will our computational tools be sufficient to accurately fit our posteriors?\\
\cellcolor{gray!6}{3. Inferential Adequacy} & \cellcolor{gray!6}{Will our inferences provide enough information to answer our questions?}\\
4. Model Adequacy & Is our model rich enough to capture the relevant structure of the true data generating process?\\
\bottomrule
\end{tabular}
\end{table}


Much work is done before seeing the data or building a model. This includes talking with experts to gain domain knowledge or to elicit priors. A benefit of modeling in a Bayesian framework is that all prior knowledge may be incorporated into the model to be used to estimate the posterior distribution. The same prior knowledge may also be used to check the posterior to ensure that predictions remain within physical or expert-given constraints.


In this section we describe a simulation-based, principled workflow proposed by @betancourt2020 and broadly adopted by many members of the Bayesian community. The workflow broadly consists of specifying the likelihood and priors, performing prior predictive checks, fitting a model, and performing posterior predictive checks. The steps of the workflow are divided into three phases: 1) pre-model, pre-data, 2) post-model, pre-data, and 3) post-model, post-data. Tables \@ref(tab:ch030-Reborn-Space), \@ref(tab:ch030-Freaky-Sledgehammer), and \@ref(tab:ch030-Bleeding-Liquid-Dagger) list the steps of each phase.

<!-- Table \@ref(tab:ch030-workflow-steps) lists the detailed steps broken up into three phases.

\begin{table}[!h]

\caption{(\#tab:ch030-workflow-steps)Principled workflow}
\centering
\begin{tabular}[t]{ll}
\toprule
Phase & Step\\
\midrule
 & conceptual analysis\\
\cmidrule{2-2}
 & define observational space\\
\cmidrule{2-2}
\multirow[t]{-3}{*}{\raggedright\arraybackslash Pre-Model, Pre-Data} & construct summary statistics\\
\cmidrule{1-2}
 & develop model\\
\cmidrule{2-2}
 & construct summary functions\\
\cmidrule{2-2}
 & simulate Bayesian ensemble\\
\cmidrule{2-2}
 & prior checks\\
\cmidrule{2-2}
 & configure algorithm\\
\cmidrule{2-2}
 & fit simulated ensemble\\
\cmidrule{2-2}
 & algorithmic calibration\\
\cmidrule{2-2}
\multirow[t]{-8}{*}{\raggedright\arraybackslash Post-Model, Pre-Data} & inferential calibration\\
\cmidrule{1-2}
 & fit observed data\\
\cmidrule{2-2}
 & diagnose posterior fit\\
\cmidrule{2-2}
 & posterior retrodictive checks\\
\cmidrule{2-2}
\multirow[t]{-4}{*}{\raggedright\arraybackslash Post-Model, Post-Data} & celebrate\\
\bottomrule
\end{tabular}
\end{table}
-->


\begin{table}[!h]

\caption{(\#tab:ch030-Reborn-Space)Pre-Model, Pre-Data steps.}
\centering
\begin{tabular}[t]{>{\raggedright\arraybackslash}p{1.75in}>{\raggedright\arraybackslash}p{3.25in}}
\toprule
Step & Description\\
\midrule
\cellcolor{gray!6}{Conceptual Analysis} & \cellcolor{gray!6}{Write down the inferential goals and consider how the variables of interest interact with the environment and how those interactions work to generate observations.}\\
Define Observational Space & What are the possible values that the observed data can take on? The observational space can help inform the statistical model such as in count data.\\
\cellcolor{gray!6}{Construct Summary Statistics} & \cellcolor{gray!6}{What measurements and estimates can be used to help ensure that the inferential goals are met? Prior predictive checks and posterior retrodictive checks are founded on summary statistics that answer the questions of domain expertise consistency and model adequacy.}\\
\bottomrule
\end{tabular}
\end{table}


\begin{table}[!h]

\caption{(\#tab:ch030-Freaky-Sledgehammer)Post-Model, Pre-Data steps.}
\centering
\begin{tabular}[t]{>{\raggedright\arraybackslash}p{1.75in}>{\raggedright\arraybackslash}p{3.25in}}
\toprule
Step & Description\\
\midrule
\cellcolor{gray!6}{Develop Model} & \cellcolor{gray!6}{Build an observational model that is consistent with the conceptual analysis and observational space, and then specify the complementary prior model.}\\
Construct Summary Functions & Use the developed model to construct explicit summary functions that can be used in prior predictive checks and posterior retrodictive checks.\\
\cellcolor{gray!6}{Simulate Bayesian Ensemble} & \cellcolor{gray!6}{Since the model is a data generating model, it can be used to simulate observations from the prior predictive distribution without yet having seen any data.}\\
Prior Checks & Check that the prior predictive distribution is consistent with domain expertise using the summary functions developed in the previous step.\\
\cellcolor{gray!6}{Configure Algorithm} & \cellcolor{gray!6}{Having simulated data, the next step is to fit the data generating model to the generated data. There are many different MCMC samplers with their own configurable parameters, so here is where those settings are tweaked.}\\
\addlinespace
Fit Simulated Ensemble & Fit the simulated data to the model using the algorithm configured in the previous step.\\
\cellcolor{gray!6}{Algorithmic Calibration} & \cellcolor{gray!6}{How well did the algorithm do in fitting the simulated data? This step helps to answer the question regarding computational faithfulness. A model may be well specified, but if the algorithm used is unreliable then the posterior distribution is also unreliable, and this can lead to poor inferences. Methods for checking models is discussed in (\#model-checking).}\\
Inferential Calibration & Are there any pathological behaviors in the model such as overfitting or non-identifiability? This step helps to answer the question of inferential adequacy.\\
\bottomrule
\end{tabular}
\end{table}


\begin{table}[!h]

\caption{(\#tab:ch030-Bleeding-Liquid-Dagger)Post-Model, Post-Data steps.}
\centering
\begin{tabular}[t]{>{\raggedright\arraybackslash}p{1.75in}>{\raggedright\arraybackslash}p{3.25in}}
\toprule
Step & Description\\
\midrule
\cellcolor{gray!6}{Fit Observed Data} & \cellcolor{gray!6}{After performing the prior predictive checks and being satisfied with the model, the next step is to fit the model to the observed data.}\\
Diagnose Posterior Fit & Did the model fit well? Can a poorly performing algorithm be fixed by tweaking the algorithmic configuration, or is there a problem with the model itself where it is not rich enough to capture the structure of the observed data? Utilize the diagnostic tools available for the algorithm to check the computational faithfulness.\\
\cellcolor{gray!6}{Posterior Retrodictive Checks} & \cellcolor{gray!6}{Do the posterior retrodictions match the observed data well, or are there still apparent discrepancies between what is expected and what is predicted by the model? It is important that any changes to the model going forward are motivated by domain expertise so as to mitigate the risk of overfitting.}\\
Celebrate & After going through the tedious process of iteratively developing a model, it is okay to celebrate before moving on to answer the research questions.\\
\bottomrule
\end{tabular}
\end{table}


These steps are not meant to be followed in a strictly linear fashion. If a conceptual misunderstanding is discovered at any step in the process, then it is recommended to go back to an earlier step and start over. The workflow is a process of model expansion, and multiple iterations are required to get to a final model (or collection of models). Similarly if the model fails prior predictive checks, then one may need to return to the model development step. A full diagram of the workflow is displayed in figure \@ref(fig:ch030-workflow-diagram).


\begin{figure}

{\centering \includegraphics[width=1\linewidth]{figures/workflow-diagram} 

}

\caption{Diagram is copywrited material of Michael Betancourt and used under the CC BY-NC 4.0 license. Image created with Lucid app.}(\#fig:ch030-workflow-diagram)
\end{figure}
