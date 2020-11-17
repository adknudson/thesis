


# Modeling Psychometric Quantities {#psych-quant}

<!--
It is now time to define priors for the model, while still not having looked at the data. The priors should be motivated by domain expertise and *prior knowledge*, not the data. There are also many choices when it comes to selecting a psychometric (sigmoid) function. Common choices are logistic, Gaussian, and Weibull.
-->

\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{040-psychometric-quantities_files/figure-latex/ch040-pf-assortment-1} 

}

\caption{Assortment of psychometric functions.}(\#fig:ch040-pf-assortment)
\end{figure}


The Weibull psychometric function is more common when it comes to 2-alternative forced choice (2-AFC) psychometric experiments where the independent variable is a stimulus intensity (non-negative) and the goal is signal detection. The data in this paper includes both positive and negative SOA values, so the Weibull is not a natural choice. Our first choice is the logistic function as it is the canonical choice for Binomial count data. The data in this study are exchangeable, meaning that the label of a positive response can be swapped with the label of a negative response and the inferences would remain the same. Since there is no natural ordering, it makes more sense for the psychometric function to be symmetric, e.g. the logistic function and Gaussian CDF. We use symmetric loosely to mean that the probability density function (PDF) has zero skewness. In practice, there is little difference in inferences between the _logit_ and _probit_ links, but computationally the logit link is more efficient.


It is appropriate to provide additional background to GLMs and their role in working with psychometric functions. A GLM allows the linear model to be related to the outcome variable via a _link_ function. An example of this is the logit link -- the inverse of the logistic function. The logistic function, $F$, takes $x \in \mathbb{R}$ and constrains the output to be in $(0, 1)$.


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


The motivation for this background is to show that a model for the psychometric function can be specified using a linear predictor, $\theta$. Given a simple slope-intercept model, the linear predictor would typically be written as:


\begin{equation}
  \theta = \alpha + \beta x
  (\#eq:linearform1)
\end{equation}


This isn't the only possible form; it could be written in the slope-location parameterization:


\begin{equation}
  \theta = \beta(x - a)
  (\#eq:linearform2)
\end{equation}


Both parameterizations will describe the same geometry, so why should it matter which form is chosen? The interpretation of the parameters change between the two models, but the reason becomes clear when we consider how the linear model relates back to the physical properties that the psychometric model describes. Take equation \@ref(eq:linearform1), substitute it in to \@ref(eq:logistic), and then take the logit of both sides:


\begin{equation}
  \mathrm{logit}(\pi) = \alpha+\beta x
  (\#eq:pfform1)
\end{equation}


Now recall that the PSS is defined as the SOA value such that the response probability, $\pi$, is $0.5$. Substituting $\pi = 0.5$ into \@ref(eq:pfform1) and solving for $x$ yields:


$$pss = -\frac{\alpha}{\beta}$$


Similarly, the JND is defined as the difference between the SOA value at the 84% level and the PSS. Substituting $\pi = 0.84$ into \@ref(eq:pfform1), solving for $x$, and subtracting off the pss yields:


\begin{equation}
  jnd = \frac{\mathrm{logit}(0.84)}{\beta}
  (\#eq:jnd1)
\end{equation}


From the conceptual analysis, it is easy to define priors for the PSS and JND, but then how does one set the priors for $\alpha$ and $\beta$? Let's say the prior for the just noticeable difference is $jnd \sim \pi_j$. Then the prior for $\beta$ would be


$$\beta \sim \frac{\mathrm{logit}(0.84)}{\pi_j}$$


The log-normal distribution has a nice property where its multiplicative inverse is still a log-normal distribution. If we let $\pi_j = \mathrm{Lognormal}(\mu, \sigma^2)$, then $\beta$ would be distributed as


$$
\beta \sim \mathrm{Lognormal}(-\mu + \ln(\mathrm{logit}(0.84)), \sigma^2)
$$


This is acceptable as the slope must always be positive for this psychometric data, and a log-normal distribution constrains the support to positive real numbers. Next suppose that the prior distribution for the PSS is $pss \sim \pi_p$. Then the prior for $\alpha$ is:


$$\alpha \sim -\pi_p \cdot \beta$$


If $\pi_p$ is set to a log-normal distribution as well, then $\pi_p \cdot \beta$ would also be log-normal, but there is still the problem of the negative sign. If $\alpha$ is always negative, then the PSS will also always be negative, which is certainly not always true. Furthermore, we don't want to _a priori_ put more weight on positive PSS values compared to negative ones.


Let's now consider using equation \@ref(eq:linearform2) and repeat the above process.


\begin{equation}
  \mathrm{logit}(\pi) = \beta(x - a)
  (\#eq:pfform2)
\end{equation}


The just noticeable difference is still given by \@ref(eq:jnd1), and so the same method for choosing a prior can be used. However, the PSS is now given by:


$$pss = \alpha$$


This is a fortunate consequence of using \@ref(eq:linearform2) because now the JND only depends on $\beta$ and the PSS only depends on $\alpha$. Additionally $\alpha$ can be interpreted as the PSS of the estimated psychometric function. Also thrown in is the ability to set a prior for $\alpha$ that is symmetric around $0$ such as a Gaussian distribution.


This also highlights the benefit of using a modeling language like `Stan` over others. For fitting GLMs in `R`, there are a handful of functions that utilize MLE like `stats::glm` and others that use Bayesian methods like `rstanarm::stan_glm` and `arm::bayesglm` [@R-rstanarm; @R-arm]. Each of these functions requires the linear predictor to be in the form of \@ref(eq:linearform1). The `stan_glm` function uses Stan in the back-end to fit a model, but is limited to priors from the Student-t family of distributions. By writing the model directly in `Stan`, the linear model can be parameterized in any way and with any prior distribution, and so allows for much more expressive modeling.
