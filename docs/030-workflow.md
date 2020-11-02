

# Principled Bayesian Workflow {#workflow}

_The meat, the cheese, the entire sandwich_

Leading up to now, I haven't discussed what is a principled Bayesian workflow, nor what multilevel modeling is. I was hoping to build up the suspense. Well I hope you're now ready for the answer. A principled Bayesian workflow is a method of employing domain expertise and statistical knowledge to iteratively build a statistical model that satisfies the constraints and goals set forth by the researcher. Oh, and Bayesian techniques are used in exchange for classical ones. Maybe not worth the suspense, but the simple idea spawns a creative and descriptive way to analyze data.

What about the multilevel aspect? While I get into that more in the following sections, the concept is simple. Multilevel models should be the default. The alternatives are models with complete pooling, or models with no pooling. Pooling vs. no pooling is a fancy way of saying that all the data is modeled as a whole, or the smallest component (group) is modeled individually. The former implies that the variation between groups is zero (all groups are the same), and the latter implies that the variation between groups is infinite (no groups are the same). Multilevel models assume that the truth is somewhere in the middle of zero and infinity. That's not a difficult thing to posit.

Hierarchical models are a specific kind of multilevel model where one or more groups are nested within a larger one. In the case of the psychometric data, there are three age groups, and within each age group are individual subjects. Multilevel modeling provides a way to quantify and apportion the variation within the data to each level in the model. For an in-depth introduction to multilevel modeling, see @gelman2006data.

There are many great resources out there for following along with an analysis of some data or problem, and much more is the abundance of tips, tricks, techniques, and testimonies to good modeling practices. The problem is that many of these prescriptions are given without context for when they are appropriate to be taken. According to @betancourt2020, this leaves "practitioners to piece together their own model building workflows from potentially incomplete or even inconsistent heuristics." The concept of a principled workflow is that for any given problem, there is not, nor should there be, a default set of steps to take to get from data exploration to predictive inferences. Rather great consideration must be given to domain expertise and the questions that one is trying to answer with the data.

Since everyone asks different questions, the value of a model is not in how well it ticks the boxes of goodness-of-fit checks, but in how consistent it is with domain expertise and its ability to answer the unique set of questions. Betancourt suggests answering four questions to evaluate a model by:

1. Domain Expertise Consistency - Is our model consistent with our domain expertise?
2. Computational Faithfulness - Will our computational tools be sufficient to accurately fit our posteriors?
3. Inferential Adequacy - Will our inferences provide enough information to answer our questions?
4. Model Adequacy - Is our model rich enough to capture the relevant structure of the true data generating process?

Like any good Bayesian^[The opposite of a Frequentist.], much work is done before seeing the data or building a model. This may include talking with experts to gain domain knowledge or to _elicit priors_. Experts may know something about a particular measure, perhaps the mean or variability of the data from years of research, and different experts may provide different estimates of a measure. The benefit of modeling in a Bayesian framework is that all prior knowledge may be incorporated into the model to be used to estimate the _posterior distribution_. The same prior knowledge may also be used to check the posterior to ensure that predictions remain within physical or expert-given constraints. Consistency is key.

The computational tool I will be using to estimate the posterior is a probabilistic programming language (PPL) called Stan [@R-rstan] within the R programming language. Stan uses the No U-Turn Sampler (NUTS) version of Hamiltonian Monte Carlo (HMC). For a gentle introduction to Bayesian statistics and sampling methods, see @bolstad2016introduction, and for an in-depth review of HMC see @betancourt2017conceptual.

Why do we need a sampler at all? Bayesian statistics and modeling stems from Bayes theorem (Equation \@ref(eq:bayesthm)). The prior $P(\theta)$ is some distribution over the parameter space and the likelihood $P(X | \theta)$ is the probability of an outcome in the sample space given a value in the parameter space. To keep things simple, we generally say that the posterior is proportional to the prior times the likelihood. Why proportional? The posterior distribution is a probability distribution, which means that the sum or integral over the parameter space must evaluate to one. Because of this constraint, the denominator in \@ref(eq:bayesthm) acts as a scale factor to ensure that the posterior is valid. Often it happens that the integral in the denominator is complex or of a high dimension. In the former situation, the integral may not be possible to evaluate, and in the latter there may not be enough computational resources in the world to perform a simple grid approximation.


\begin{equation}
  P(\theta | X) = \frac{P(X | \theta)\cdot P(\theta)}{\sum_i P(X | \theta_i)} =   \frac{P(X | \theta)\cdot P(\theta)}{\int_\Omega P(X | \theta)d\theta}
  (\#eq:bayesthm)
\end{equation}


The solution is to use Markov Chain Monte Carlo (MCMC). The idea is that we can _draw samples_ from the posterior distribution in a way that samples proportionally to the density. This sampling is a form of approximation to the area under the curve (i.e. an approximation to the denominator in \@ref(eq:bayesthm)). Rejection sampling [@gilks1992adaptive] and slice sampling [@neal2003slice] are basic methods for sampling from a target distribution, however they can often be inefficient^[Efficiency of a sampler is related to the proportion of proposal samples that get accepted.]. NUTS is a much more complex algorithm that can be compared to a physics simulation. A massless "particle" is flicked in a random direction with some amount of kinetic energy in a probability field, and is stopped randomly. The stopping point is the new proposal sample. The No U-Turn part means that when the algorithm detects that the particle is turning around, it will stop so as not to return to the starting position. This sampling scheme has a much higher rate of accepted samples, and also comes with many built-in diagnostic tools that let us know when the sampler is having trouble efficiently exploring the posterior. I'll talk more about these diagnostic tools throughout the remaining sections and in [chapter 4](#model-checking).

The question of inferential adequacy depends on the set of questions that we are seeking to answer with the data from the psychometric experiment. The broad objective is to determine if there are any significant differences between age groups when it comes to temporal sensitivity, perceptual synchrony, and temporal recalibration, and if the task influences the results as well. The specific goals are to estimate and compare the PSS an JND across all age groups, conditions, and tasks, and determine the affect of recalibration between age groups.

For the last question, model adequacy, I will be following a set of steps proposed in @betancourt2020. The purpose of laying out these steps is not to again blindly check them off, but to force the analyst to carefully consider each point and make an _informed_ decision whether the step is necessary or to craft the specifics of how the step should be completed. The steps are listed in table \@ref(tab:ch030-workflow-steps). These steps are also not meant to be followed linearly. If at any point it is discovered that there is an issue in conceptual understanding or model adequacy or something else, then it is encouraged to go back to a previous step and start with a new understanding.

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>(\#tab:ch030-workflow-steps)Principled workflow</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> Part </th>
   <th style="text-align:left;"> Step </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;vertical-align: top !important;" rowspan="3"> Pre-Model, Pre-Data </td>
   <td style="text-align:left;"> conceptual analysis </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> define observational space </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> construct summary statistics </td>
  </tr>
  <tr>
   <td style="text-align:left;vertical-align: top !important;" rowspan="8"> Post-Model, Pre-Data </td>
   <td style="text-align:left;"> develop model </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> construct summary functions </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> simulate Bayesian ensemble </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> prior checks </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> configure algorithm </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> fit simulated ensemble </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> algorithmic calibration </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> inferential calibration </td>
  </tr>
  <tr>
   <td style="text-align:left;vertical-align: top !important;" rowspan="4"> Post-Model, Post-Data </td>
   <td style="text-align:left;"> fit observed data </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> diagnose posterior fit </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> posterior retrodictive checks </td>
  </tr>
  <tr>
   
   <td style="text-align:left;"> celebrate </td>
  </tr>
</tbody>
</table>

I'll talk about each step in the first iteration, but may choose to omit steps in subsequent iterations if there are no changes. For the purposes of building a model and being concise, I will focus around the audiovisual TOJ task in this chapter, but the final model will apply similarly to the visual and duration tasks. For the sensorimotor task, the model will be modified to accept Bernoulli data as apposed to aggregated Binomial counts (described more in the next section).

## Iteration 1 (journey of a thousand miles) {#iter1}

**Pre-Model, Pre-Data**

I begin the modeling process by modeling the experiment according to the description of how it occurred and how the data were collected. This first part consists of conceptual analysis, defining the observational space, and constructing summary statistics that can help us to identify issues in the model specification.

_Conceptual Analysis_

In section \@ref(toj-task) I discussed the experimental setup and data collection. To reiterate, subjects are presented with two stimuli separated by some temporal delay, and they are asked to respond as to their perception of the temporal order. There are 45 subjects with 15 each in the young, middle, and older age groups. As the SOA becomes larger in the positive direction, subjects are expected to give more "positive" responses, and as the SOA becomes larger in the negative direction, more "negative" responses are expected. By the way the experiment and responses are constructed, there is no expectation to see a reversal of this trend unless there was an issue with the subject's understanding of the directions given to them or an error in the recording device.

After the first experimental block the subjects go through a recalibration period, and repeat the experiment again. The interest is in seeing if the recalibration has an effect on temporal sensitivity and perceptual synchrony, and if the effect is different for each age group.

_Define Observational Space_

The response that subjects give during a TOJ task is recorded as a zero or a one (see section \@ref(toj-task)), and their relative performance is determined by the SOA value. Let $y$ represent the binary outcome of a trial and let $x$ be the SOA value.


\begin{align*}
y_i &\in \lbrace 0, 1\rbrace \\
x_i &\in \mathbb{R}
\end{align*}


If the SOA values are fixed like in the audiovisual task, then the responses can be aggregated into binomial counts, $k$.


$$
k_i, n_i \in \mathbb{Z}_0^+, k_i \le n_i
$$


In the above expression, $\mathbb{Z}_0^+$ represents the set of non-negative integers. Notice that the number of trials $n$ has an index variable $i$. This is because the number of trials per SOA is not fixed between blocks. In the pre-adaptation block, there are five trials per SOA compared to three in the post-adaptation block. So if observation 32 is recorded during a "pre" block, $n_{32} = 5$, and if observation 1156 is during a "post" block, $n_{1156} = 3$. Of course this is assuming that each subject completed all trials in the block, but the flexibility of the indexing can manage even if they didn't.

Then there are also three categorical variables -- age group, subject ID, and trial (block). The first two are treated as factor variables^[Factor variables also go by the name index variable or categorical variable]. Rather than using one-hot encoding or dummy variables, the age levels are left as categories and a coefficient is fit for each level. Among the benefits of this approach is the ease of interpretation and ease of working with the data programmatically. This is especially true at the subject level. If a dummy variables was used for all 45 subjects, we would have 44 different dummy variables to work with times the number of coefficients that make estimates at the subject level. The number of parameters in the model grows rapidly as the model complexity grows.

Age groups and individual subjects can be indexed in the same way that number of trials is indexed. $S_i$ refers to the subject in record $i$, and similarly $G_i$ refers to the age group of that subject. Observation 63 is for record ID av-post1-M-f-HG, so then $S_{63}$ is M-f-HG and $G_{63}$ is middle_age. Under the hood of R, these factor levels are represented as integers (e.g. middle age group level is stored internally as the number 2).



```r
(x <- factor(c("a", "a", "b", "c")))
#> [1] a a b c
#> Levels: a b c
storage.mode(x)
#> [1] "integer"
```

This data storage representation can later be exploited for the Stan model.

The pre- and post-adaptation categories are treated as a binary indicator referred to as $trt$ (short for treatment) since there are only two levels in the category. In this setup, a value of 1 indicates a post-adaptation block. I chose this encoding over the reverse because the pre-adaptation block is like the baseline performance, and it seemed more appropriate to interpret the post-adaptation block as turning on some effect. Using a binary indicator in a regression setting may not be the best practice as I discuss in section \@ref(iter2).

In the Stan modeling language, data for a binomial model with subject and age group levels and treatment is specified as

\setstretch{1.0}
```
data {
  int N;        // Number of observations
  int N_S;      // Number of subject levels
  int N_G;      // Number of age group levels
  int N_T;      // Number of treatment/control groups
  int n[N];     // Trials per SOA
  int k[N];     // binomial counts
  vector[N] x;  // SOA values
  int S[N];     // Subject identifier
  int G[N];     // Age group identifier
  int trt[N];   // Treatment indicator
}
```
\setstretch{2.0}

In Stan (and unlike in R), data types must be statically declared. While sometimes a nuisance, this requirement aids in something called _type inference_, and also lets Stan optimize certain parts of the model. 

_Construct Summary Statistics_

In order to effectively challenge the validity of the model, a set of summary statistics are constructed that help answer the questions of domain expertise consistency and model adequacy. We are studying the affects of age and temporal recalibration through the PSS and JND (see section \@ref(psycho-experiments)), so it is natural to define summary statistics around these quantities to verify model consistency. Additionally the PSS and JND can be computed regardless of the model parameterization or chosen psychometric function.

By the experimental setup and recording process, it is impossible that a properly conducted block would result in a JND less than 0 (i.e. the psychometric function is always non-decreasing), so that can be a lower limit for its threshold. On the other end it is unlikely that it will be beyond the limits of the SOA values, but even more concrete, it seems unlikely (though not impossible) that the just noticeable difference would be more than a second.

The lower bound on the JND can be further refined if we draw information from other sources. Some studies show that we cannot perceive time differences below 30 ms, and others show that an input lag as small as 100ms can impair a person's typing ability. Then according to these studies, a time delay of 100ms is enough to notice, and so a just noticeable difference should be much less than one second -- much closer to 100ms. I'll continue to use one second as an extreme estimate indicator, but will incorporate this knowledge when it comes to selecting priors.

As for the point of subjective simultaneity, it can be either positive or negative, with the belief that larger values are more rare. Some studies suggest that for audio-visual TOJ tasks, the separation between stimuli need to be as little as 20 milliseconds for subjects to be able to determine which modality came first [@vatakis2007influence]. Other studies suggest that our brains can detect temporal differences as small as 30 milliseconds. If these values are to be believed then we should be skeptical of PSS estimates larger than say 150 milliseconds in absolute value, just to be safe.

A histogram of computed PSS and JND values will suffice for summary statistics. We can estimate the proportion of values that fall outside of our limits defined above, and use them as indications of problems with the model fitting or conceptual understanding.

**Post-Model, Pre-Data**

It is now time to define priors for the model, while still not having looked at the [distribution of] data. The priors should be motivated by domain expertise and *prior knowledge*, not the data. There are also many choices when it comes to selecting a psychometric (sigmoid) function. Common ones are logistic, Gaussian, and Weibull.


<div class="figure" style="text-align: center">
<img src="030-workflow_files/figure-html/ch031-pf-assortment-1.png" alt="Assortment of psychometric functions." width="70%" />
<p class="caption">(\#fig:ch031-pf-assortment)Assortment of psychometric functions.</p>
</div>


The Weibull psychometric function is more common when it comes to 2-AFC psychometric experiments where the independent variable is a stimulus intensity (non-negative) and the goal is signal detection. The data in this paper includes both positive and negative SOA values, so the Weibull is not a natural choice. In fact, because this is essentially a model for logistic regression, my first choice is the logistic function as it is the canonical choice for Binomial data. Additionally, the data in this study are reversible. The label of a positive response can be swapped with the label of a negative response and the inferences should remain the same. Since there is no natural ordering, it makes more sense for the psychometric function to be symmetric, e.g. the logistic and Gaussian. I use symmetric loosely to mean that probability density function (PDF) is symmetric about its middle. More specifically, the distribution has zero skewness.

In practice, there is little difference in inference between the _logit_ and _probit_ links, but computationally the logit link is more efficient. I am also more familiar with working on the log-odds scale compared to the probit scale, so I make the decision to go forward with the logistic function. In [chapter 4](#model-checking) I will show how even with a mis-specified link function, we can still achieve accurate predictions.

_Develop Model_

Before moving on to specifying priors, I think it is appropriate to provide a little more background into generalized linear models (GLMs) and their role in working with psychometric functions. A GLM allows the linear model to be related to the outcome variable via a _link_ function. An example of this is the logit link - the inverse of the logistic function. The logistic function, $F$, takes $x \in \mathbb{R}$ and constrains the output to be in $(0, 1)$.


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


The purpose of all this setup is to show that a model for the psychometric function can be specified using a linear predictor, $\theta$. Given a simple slope-intercept model, one would typically write the linear predictor as


\begin{equation}
  \theta = \alpha + \beta x
  (\#eq:linearform1)
\end{equation}


This isn't the only acceptable form; it could be written in the centered parameterization


\begin{equation}
  \theta = \beta(x - a)
  (\#eq:linearform2)
\end{equation}


Both parameterizations will describe the same geometry, so why should it matter which form is chosen? Clearly the interpretation of the parameters change between the two models, but the reason becomes clear when you consider how the linear model relates back to the physical properties that the psychometric model describes. Take equation \@ref(eq:linearform1), substitute it in to \@ref(eq:logistic), and then take the logit of both sides


\begin{equation}
  \mathrm{logit}(\pi) = \alpha+\beta x
  (\#eq:pfform1)
\end{equation}


Now recall that the PSS is defined as the SOA values such that the response probability, $\pi$, is $0.5$. Substituting $\pi = 0.5$ into \@ref(eq:pfform1) and solving for $x$ yields

$$
pss = -\frac{\alpha}{\beta}
$$

Similarly, the JND is defined as the difference between the SOA value at the 84% level and the PSS. Substituting $\pi = 0.84$ into \@ref(eq:pfform1), solving for $x$, and subtracting off the pss yields


\begin{equation}
  jnd = \frac{\mathrm{logit}(0.84)}{\beta}
  (\#eq:jnd1)
\end{equation}


From the conceptual analysis, it is easy to define priors for the PSS and JND, but then how does one set the priors for $\alpha$ and $\beta$? Let's say the prior for the just noticeable difference is $jnd \sim \pi_j$. Then the prior for $\beta$ would be

$$
\beta \sim \frac{\mathrm{logit}(0.84)}{\pi_j}
$$

The log-normal distribution has a nice property where its multiplicative inverse is still a log-normal distribution. We could let $\pi_j = \mathrm{Lognormal}(\mu, \sigma^2)$ and then $\beta$ would be distributed as

$$
\beta \sim \mathrm{Lognormal}(-\mu + \ln(\mathrm{logit}(0.84)), \sigma^2)
$$

This is acceptable, as it was determined last chapter that the slope must always be positive, and a log-normal distribution constrains the support to postive real numbers. Next suppose that the prior distribution for the PSS is $pss \sim \pi_p$. Then the prior for $\alpha$ is 

$$
\alpha \sim -\pi_p \cdot \beta
$$

If $\pi_p$ is set to a log-normal distribution as well, then $\pi_p \cdot \beta$ would also be log-normal, but there is still the problem of the negative sign. If $\alpha$ is always negative, then the PSS will also always be negative, which is certainly not always true. Furthermore, I don't want to _a priori_ put more weight on positive PSS values compared to negative ones, for which a lognormal distribution would not do.

Let's now go back and consider using equation \@ref(eq:linearform2) and repeat the above process.


\begin{equation}
  \mathrm{logit}(\pi) = \beta(x - a)
  (\#eq:pfform2)
\end{equation}


The just noticeable difference is still given by \@ref(eq:jnd1) and so the same method for choosing a prior can be used, but the PSS is now given by

$$
pss = \alpha
$$

This is a fortunate consequence of using \@ref(eq:linearform2) because now the JND only depends on $\beta$ and the PSS only depends on $\alpha$, and now $\alpha$ can literally be interpreted as the PSS of the estimated psychometric function! Also thrown in is the ability to set a prior for $\alpha$ that is symmetric around $0$ like a Gaussian distribution.

This also brings me to point out the first benefit of using a modeling language like Stan over others. For fitting GLMs in R, there are a handful of functions that utilize MLE like `stats::glm` and others that use Bayesian methods like `rstanarm::stan_glm` and `arm::bayesglm` [@R-rstanarm; @R-arm]. Each of these functions requires the linear predictor to be in the form of \@ref(eq:linearform1). The `stan_glm` function actually uses Stan in the backend to fit a model, but is limited to priors from the Student t family of distributions. By writing the model directly in Stan, the linear model can be parameterized in any way and with any prior distribution, and so allows for much more expressive modeling - a key aspect of this principled workflow.

For the first iteration of this model, I am going to start with the simplest model that captures the structure of the data without including information about age group, treatment, or subject. Here is a simple model that draws information from the conceptual analysis. 


\begin{align*}
  k_i &\sim \mathrm{Binomial}(n_i, p_i) \\
  \mathrm{logit}(p_i) &= \beta ( x_i - \alpha )
\end{align*}


Since I am using the linear model from \@ref(eq:linearform2), setting the priors for $\alpha$ and $\beta$ is relatively straightforward. The PSS can be positive or negative without any expected bias towards either, so a symmetric distribution like the Gaussian is a fine choice for $\alpha$ without having any other knowledge about the distribution of PSS values. Since I said earlier that a PSS value more than 150ms in absolute value is unlikely, I can define a Gaussian prior such that $P(|pss| > 0.150) \approx 0.01$. Since the prior does not need to be exact, the following mean and variance suffice

$$
pss \sim \mathcal{N}(0, 0.06^2) \Longleftrightarrow \alpha \sim \mathcal{N}(0, 0.06^2)
$$

For the just noticeable difference, I will continue to use the log-normal distribution because it is constrained to positive values and has the nice reciprocal property. The JND is expected to be close to 100ms and extremely unlikely to exceed 1 second. This implies a prior such that the mean is around 100ms and the bulk of the distribution is below 1 second - i.e. $E[X] \approx 0.100$ and $P(X < 1) \approx 0.99$. This requires solving a system of nonlinear equations in two variables

$$
\begin{cases}
E[X] = 0.100 = \exp\left(\mu + \sigma^2 / 2\right) \\
P(X < 1) = 0.99 = 0.5 + 0.5 \cdot \mathrm{erf}\left[\frac{\ln (1) - \mu}{\sqrt{2} \cdot \sigma}\right]
\end{cases}
$$

This nonlinear system can be solved using Stan's algebraic solver.


```stan
functions {
  vector system(vector y, vector theta, real[] x_r, int[] x_i) {
    vector[2] z;
    z[1] = exp(y[1] + y[2]^2 / 2) - theta[1];
    z[2] = 0.5 + 0.5 * erf(-y[1] / (sqrt(2) * y[2])) - theta[2];
    return z;
  }
}
transformed data {
  vector[2] y_guess = [1, 1]';
  real x_r[0];
  int x_i[0];
}
transformed parameters {
  vector[2] theta = [0.100, 0.99]';
  vector[2] y;
  y = algebra_solver(system, y_guess, theta, x_r, x_i);
}
```



```r
fit <- sampling(prior_jnd, iter=1, warmup=0, chains=1, refresh=0,
                seed=31, algorithm="Fixed_param")
sol <- extract(fit)
sol$y
#>           
#> iterations   [,1]  [,2]
#>       [1,] -7.501 3.225
```

The solver has determined that $\mathrm{Lognormal}(-7.5, 3.2^2)$ is the appropriate prior. However, simulating some values from this distribution produces a lot of extremely small values ($<10^{-5}$) and a few extremely large values ($\approx 10^2$). This is because the expected value of a log-normal random variable depends on both the mean and standard deviation. If the median is used in place for the mean, then a more acceptable prior may be determined.






```r
fit <- sampling(prior_jnd_using_median, iter=1, warmup=0, chains=1, refresh=0,
                seed=31, algorithm="Fixed_param")
sol <- extract(fit)
sol$y
#>           
#> iterations   [,1]   [,2]
#>       [1,] -2.303 0.9898
```

Sampling from a log-normal distribution with these parameters and plotting the histogram shows no inconsistency with the domain expertise.

<img src="030-workflow_files/figure-html/ch031-Risky-Lion-1.png" width="70%" style="display: block; margin: auto;" />

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

\setstretch{1.0}

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
  alpha ~ normal(0, 0.05);
  beta ~ lognormal(3.0, 1.5);
  k ~ binomial_logit(n, p);
}
```
\setstretch{2.0}

Notice that the model block is nearly identical to the mathematical model!

_Construct Summary Functions_

Whew! that was a lot of work to define the priors for just two parameters. Thankfully going forward not as much work will need to be done to expand the model. The next step is to construct any relevant summary functions. Since the distribution of posterior PSS and JND values are needed for the summary statistics, it will be nice to have a function that can take in the posterior samples for $\alpha$ and $\beta$ and return the PSS and JND values. I'll define $Q$ as a more general function that takes in the two parameters and a probability, $\pi$, and returns the distribution of SOA values at $\pi$.


\begin{equation}
  Q(\pi; \alpha, \beta) = \frac{\mathrm{logit(\pi)}}{\beta} + \alpha
  (\#eq:summfun1)
\end{equation}


The function can be defined in R as


```r
Q <- function(p, a, b) qlogis(p) / b + a
```

With $Q$, the PSS and JND can be calculated as


\begin{align}
  pss &= Q(0.5) \\
  jnd &= Q(0.84) - Q(0.5)
\end{align}


_Simulate Bayesian Ensemble_

During this step, I simulate the Bayesian ensemble and later feed the prior values into the summary functions in order to verify that there are no other inconsistencies with domain knowledge. Since the model is fairly simple, I will simulate directly in R.



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

This step pertains to ensuring that prior estimates are consistent with domain expertise. I already did that in the model construction step by sampling values for the just noticeable difference. The first prior chosen was not producing JND estimates that were consistent with domain knowledge, so I adjusted accordingly. That check would normally be done during this step, and I would have had to return to the model development step.

Figure \@ref(fig:ch030-prior-pf-plot) shows the distribution of prior psychometric functions derived from the simulated ensemble. There are a few very steep and very shallow curves, but the majority fall within a range that appears likely.


<div class="figure" style="text-align: center">
<img src="030-workflow_files/figure-html/ch030-prior-pf-plot-1.png" alt="Prior distribution of psychometric functions using the priors for alpha and beta." width="70%" />
<p class="caption">(\#fig:ch030-prior-pf-plot)Prior distribution of psychometric functions using the priors for alpha and beta.</p>
</div>


Additionally most of the PSS values are within $\pm 0.1$ with room to allow for some larger values. Let's check the prior distribution of PSS and JND values.


<div class="figure" style="text-align: center">
<img src="030-workflow_files/figure-html/ch031-prior-pss-plot-1.png" alt="PSS prior distribution." width="70%" />
<p class="caption">(\#fig:ch031-prior-pss-plot)PSS prior distribution.</p>
</div>


<div class="figure" style="text-align: center">
<img src="030-workflow_files/figure-html/ch031-prior-jnd-plot-1.png" alt="JND prior distribution." width="70%" />
<p class="caption">(\#fig:ch031-prior-jnd-plot)JND prior distribution.</p>
</div>


I am satisfied with the prior coverage of the PSS and JND values, and there are only a few samples that go beyond the extremes that were specified in the summary statistics step.

_Configure Algorithm_

There are a few parameters that can be set for Stan. On the user side, the main parameters are the number of iterations, the number of warm-up iterations, the target acceptance rate, and the number of chains to run. The NUTS algorithm samples in two phases: a warm-up phase and a sampling phase. During the warm-up phase, the sampler is automatically tuning three internal parameters that can significantly affect the sampling efficiency. By default, the Stan function will use half the number of iterations for warm-up and the other half for actual sampling. The full details of Stan's HMC algorithm is described in the Stan reference manual. For now I am going to use the default algorithm parameters in Stan, and will tweak them later if and when issues arise.

_Fit Simulated Ensemble_

Nothing to say here. Only code.


```r
sim_dat <- with(av_dat, list(N = N, x = x, n = n, k = sim_k)) 
m031 <- sampling(m031_stan, data = sim_dat, 
                 chains = 4, cores = 4, refresh = 0)
```

_Algorithmic Calibration_

One benefit of using HMC over other samplers like Gibbs sampling is that HMC offers diagnostic tools for the health of chains and the ability to check for _divergent transitions_. Recall that the HMC and NUTS algorithm can be imagined as a physics simulation of a particle in a potential energy field, and a random momentum is imparted on the particle. The sum of the potential energy and the kinetic energy of the system is called the Hamiltonian, and is conserved along the trajectory of the particle (@stanref). The path that the particle takes is a discrete approximation to the actual path where the position of the particle is updated in small steps called _leapfrog steps_ (see @leimkuhler2004simulating for a detailed explanation of the leapfrog algorithm). A divergent transition happens when the simulated trajectory is far from the true trajectory as measured by the Hamiltonian.

To check the basic diagnostics of the model, I run the following code.

\setstretch{1.0}

```r
check_hmc_diagnostics(m031)
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
\setstretch{2.0}

There is no undesirable behavior from this model, so next I check the summary statistics of the estimated parameters. The $\hat{R}$ statistic is a comparison of the measure of variance within chains and between chains. When chains have converged to a stationary distribution, the variance within and between chains is the same, and the ratio is one. Values of $\hat{R} > 1.1$ are usually indicative of chains that have not converged to a common distribution. Lastly there is the effective sample size ($N_{\mathrm{eff}}$) which is a loose measure for the autocorrelation within the parameter samples. As autocorrelation generally decreases as the lag increases, one can achieve a higher $N_{\mathrm{eff}}$ by running a chain with more samples and then _thinning_ the samples, i.e. saving only every $n^{th}$ sample.


<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>(\#tab:ch031-Cloudy-Toupee)Summary statistics of the fitted Bayesian ensemble.</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> parameter </th>
   <th style="text-align:right;"> mean </th>
   <th style="text-align:right;"> se_mean </th>
   <th style="text-align:right;"> sd </th>
   <th style="text-align:right;"> 2.5% </th>
   <th style="text-align:right;"> 97.5% </th>
   <th style="text-align:right;"> n_eff </th>
   <th style="text-align:right;"> Rhat </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> alpha </td>
   <td style="text-align:right;"> 0.0061 </td>
   <td style="text-align:right;"> 0.0001 </td>
   <td style="text-align:right;"> 0.0035 </td>
   <td style="text-align:right;"> -0.0007 </td>
   <td style="text-align:right;"> 0.0129 </td>
   <td style="text-align:right;"> 3728 </td>
   <td style="text-align:right;"> 1.000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> beta </td>
   <td style="text-align:right;"> 10.7726 </td>
   <td style="text-align:right;"> 0.0054 </td>
   <td style="text-align:right;"> 0.2473 </td>
   <td style="text-align:right;"> 10.3054 </td>
   <td style="text-align:right;"> 11.2600 </td>
   <td style="text-align:right;"> 2073 </td>
   <td style="text-align:right;"> 1.001 </td>
  </tr>
</tbody>
</table>

Both the $\hat{R}$ and $N_{\mathrm{eff}}$ look fine for both $\alpha$ and $\beta$, thought it is slightly concerning that $\alpha$ is centered relatively far from zero. This could just be due to sampling variance, so I will continue on to the next step.

_Inferential Calibration_



**Post-Model, Post-Data**

_Fit Observed Data_

All of the work up until now has been done without peaking at the observed data. Satisfied with the model so far, I can now go ahead and run the data through.



```r
obs_dat <- with(av_dat, list(N = N, x = x, n = n, k = k)) 
m031 <- sampling(m031_stan, data = obs_dat, 
                 chains = 4, cores = 4, refresh = 0)
```


_Diagnose Posterior Fit_

Here I repeat the diagnostic checks that I used after fitting the simulated Bayesian ensemble. 

\setstretch{1.0}

```r
check_hmc_diagnostics(m031)
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
\setstretch{2.0}

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>(\#tab:ch031-Maroon-Oyster)Summary statistics of the fitted Bayesian ensemble.</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> parameter </th>
   <th style="text-align:right;"> mean </th>
   <th style="text-align:right;"> se_mean </th>
   <th style="text-align:right;"> sd </th>
   <th style="text-align:right;"> 2.5% </th>
   <th style="text-align:right;"> 97.5% </th>
   <th style="text-align:right;"> n_eff </th>
   <th style="text-align:right;"> Rhat </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> alpha </td>
   <td style="text-align:right;"> 0.0373 </td>
   <td style="text-align:right;"> 0.0001 </td>
   <td style="text-align:right;"> 0.0040 </td>
   <td style="text-align:right;"> 0.0293 </td>
   <td style="text-align:right;"> 0.0453 </td>
   <td style="text-align:right;"> 4035 </td>
   <td style="text-align:right;"> 0.9992 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> beta </td>
   <td style="text-align:right;"> 8.4236 </td>
   <td style="text-align:right;"> 0.0039 </td>
   <td style="text-align:right;"> 0.1857 </td>
   <td style="text-align:right;"> 8.0771 </td>
   <td style="text-align:right;"> 8.7992 </td>
   <td style="text-align:right;"> 2311 </td>
   <td style="text-align:right;"> 1.0017 </td>
  </tr>
</tbody>
</table>

No indications of an ill-behaved posterior fit! Let's also check the posterior distribution of $\alpha$ and $\beta$ against the prior density (\@ref(fig:ch031-m031-posterior-alpha-beta)).


<div class="figure" style="text-align: center">
<img src="030-workflow_files/figure-html/ch031-m031-posterior-alpha-beta-1.png" alt="Comparison of posterior distributions for alpha and beta to their respective prior distributions." width="70%" />
<p class="caption">(\#fig:ch031-m031-posterior-alpha-beta)Comparison of posterior distributions for alpha and beta to their respective prior distributions.</p>
</div>

The posterior distributions for $\alpha$ and $\beta$ are well within the range determined by domain knowledge, and highly concentrated due to both the large amount of data and the fact that this is a completely pooled model - no stratification. As expected, the prior for the JND could have been tighter with more weight below half a second compared to the one second limit used, but this is not prior information, so it is not prudent to change the prior in this manner after having seen the posterior. As a rule of thumb, priors should only be updated as motivated by domain expertise and not by posterior distributions.

_Posterior Retrodictive Checks_

Finally it is time to run the posterior samples through the summary functions and then perform _retrodictive_ checks. A retrodiction is using the posterior model to predict and compare to the observed data. This is simply done by drawing samples from the posterior and feeding in the observational data. This may be repeated to gain a retrodictive distribution.



```r
posterior_pss <- Q(0.5, p031$alpha, p031$beta)
posterior_jnd <- Q(0.84, p031$alpha, p031$beta) - posterior_pss
```


<div class="figure" style="text-align: center">
<img src="030-workflow_files/figure-html/ch031-posterior-pss-jnd-plot-1.png" alt="Posterior distribution of the PSS and JND." width="70%" />
<p class="caption">(\#fig:ch031-posterior-pss-jnd-plot)Posterior distribution of the PSS and JND.</p>
</div>

Neither of the posterior estimates for the PSS or JND exceed the extreme cutoffs set in the earlier steps, so I can be confident that the model is consistent with domain expertise. Let's also take a second to appreciate how simple it is to visualize and summarize the distribution of values for these measures. Using classical techniques like MLE might require using bootstrap methods to estimate the distribution of parameter values, or one might approximate the parameter distributions using the mean and standard error of the mean to simulate new values. Since we have the entire posterior distribution we can calculate the distribution of transformed parameters by working directly with the posterior samples and be sure that the intervals are credible.

Next is to actually do the posterior retrodictions. I will do this in two steps to better show how the distribution of posterior psychometric functions relates to the observed data, and then compare the observed data to the retrodictions. Figure \@ref(fig:ch031-posterior-pf-plot) shows the result of the first step.

<div class="figure" style="text-align: center">
<img src="030-workflow_files/figure-html/ch031-posterior-pf-plot-1.png" alt="Posterior distribution of psychometric functions using pooled observations." width="70%" />
<p class="caption">(\#fig:ch031-posterior-pf-plot)Posterior distribution of psychometric functions using pooled observations.</p>
</div>

Next I sample parameter values from the posterior distribution and use them to simulate a new data set. In the next iteration I will show how I can get Stan to automatically produce retrodictions for me in the model fitting step. The results of the posterior retrodictions are shown in figure \@ref(fig:ch031-obs-vs-retro-plot).



```r
alpha <- sample(p031$alpha, n_obs, replace = TRUE)
beta  <- sample(p031$beta, n_obs, replace = TRUE)
logodds <- beta * (av_dat$x - alpha)
probs <- logistic(logodds)
sim_k <- rbinom(n_obs, av_dat$n, probs)
```


<div class="figure" style="text-align: center">
<img src="030-workflow_files/figure-html/ch031-obs-vs-retro-plot-1.png" alt="Observed data compared to the posterior retrodictions. The data is post-stratified by block for easier visualization." width="70%" />
<p class="caption">(\#fig:ch031-obs-vs-retro-plot)Observed data compared to the posterior retrodictions. The data is post-stratified by block for easier visualization.</p>
</div>

I want to make it clear exactly what the first iteration of this model tells us. It is the average distribution of underlying psychometric functions across all subjects and blocks. It cannot tell us what the differences are between pre- and post-adaptation blocks are, or even what the variation between subjects is. As such, it is only useful in determining if the average value for the PSS is different from 0 or if the average JND is different from some other predetermined level. This model is still useful given the right question, but this model cannot answer questions about group-level effects.

Figure \@ref(fig:ch031-obs-vs-retro-plot) shows that the model captures the broad structure of the observed data, but is perhaps a bit under-dispersed in the tail ends of the SOA values. Besides this one issue, I am satisfied with the first iteration of this model and am ready to proceed to the next iteration.

## Iteration 2 (electric boogaloo) {#iter2}

In this iteration I will be adding in the treatment and age groups into the model. There are no changes with the conceptual understanding of the experiment, and nothing to change with the observational space. As such I will be skipping the first three steps and go straight to the model development step. As I build the model, the number of changes from one iteration to the next should go to zero as the model _expands_ to become only as complex as necessary to answer the research questions.

**Post-Model, Pre-Data**

_Develop Model_

To start, let's add in the treatment indicator and put off consideration of adding in the age group levels. In classical statistics, it is added as an indicator variable (zero or one) for both the slope and intercept (varying slopes, varying intercepts model). Let $trt$ be $0$ if it is the pre-adaptation block and $1$ if the observation comes from the post-adaptation block.

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

This might seem like a natural way to introduce an indicator variable, but it comes with serious implications. This model implies that there is more uncertainty about the post-adaptation block compared to the baseline block, and this is not necessarily true. 

\begin{align*}
\mathrm{Var}(\theta_{post}) &= \mathrm{Var}((\alpha + \alpha_{trt}) + (\beta + \beta_{trt}) \times x) \\
&= \mathrm{Var}(\alpha) + \mathrm{Var}(\alpha_{trt}) + x^2 \mathrm{Var}(\beta) + x^2\mathrm{Var}(\beta_{trt})
\end{align*}


On the other hand, the variance of $\theta_{pre}$ is

$$
\mathrm{Var}(\theta_{pre}) = \mathrm{Var}(\alpha) + x^2 \mathrm{Var}(\beta) \le \mathrm{Var}(\theta_{post})
$$

Furthermore, the intercept, $\alpha$, is no longer the average response probability at $x=0$ for the entire data set, but is instead exclusively the average for the pre-adaptation block. This may not matter in certain analyses, but one nice property of multilevel models is the separation of population level estimates and group level estimates (fixed vs. mixed effects).

So instead the treatment variable is introduced into the linear model as a factor variable. This essentially means that each level in the treatment gets its own parameter estimate, and this also makes it easier to set priors when there are many levels in a group (such as for the subject level). The linear model, using equation \@ref(eq:linearform2), with the treatment is written as


\begin{equation}
  \theta = (\beta + \beta_{trt[i]}) \left[x_i - (\alpha + \alpha_{trt[i]})\right]
  (\#eq:linearmodel2)
\end{equation}


As I add in more predictors and groups, equation \@ref(eq:linearmodel2) will start to be more difficult to read. What I can do is break up the slope and intercept parameters and write the linear model as


\begin{align*}
\mu_\alpha &= \alpha + \alpha_{trt[i]} \\
\mu_\beta &= \beta + \beta_{trt[i]} \\
\theta &= \mu_\beta (x - \mu_\alpha)
\end{align*}


In this way the combined parameters can be considered separately from the linear parameterization. Which leads me to consider the priors for $\alpha_{trt}$ and $\beta_{trt}$. The way that we can turn an normal model with categorical predictors into a multilevel model is by allowing the priors to borrow information from other groups. This is accomplished by putting priors on priors. It is easier to write down the model first before explaining how it works.


\begin{align*}
k_i &\sim \mathrm{Binomial}(n_i, p_i) \\
\mu_\alpha &= \alpha + \alpha_{trt[i]} \\
\mu_\beta &= \beta + \beta_{trt[i]} \\
\mathrm{logit}(p_i) &= \mu_\beta (x_i - \mu_\alpha) \\
\alpha &\sim \mathcal{N}(0, 0.06^2) \\
\alpha_{trt} &\sim \mathcal{N}(0, \sigma_{trt}^2) \\
\sigma_{trt} &\sim \textrm{to be defined}
\end{align*}


In the above model, $\alpha$ gets a fixed prior (the same as in the first iteration), and $\alpha_{trt}$ gets a Gaussian prior with an adaptive variance term that is allowed to be learned from the data. This notation is compact, but $\alpha_{trt}$ is actually two parameters - one each for pre- and post-adaptation block - but they both share the same variance term $\sigma_{trt}$. This produces a _regularizing_ effect where both treatment estimates are shrunk towards the mean, $\alpha$.

I'll discuss selecting a prior for the variance term shortly, but now I want to discuss setting the prior for the slope terms. Instead of modeling $\beta$ with a log-normal prior, I can sample from a normal distribution and take the exponential of it to produce a log-normal distribution. I.e.

$$
X \sim \mathcal{N}(3, 1^2) \\
Y = \exp\left\lbrace X \right\rbrace \Longleftrightarrow Y \sim \mathrm{Lognormal(3, 1^2)}
$$

The motivation behind this transformation is that it is now easier to include new slope variables as an additive affect. If both $\beta$ and $\beta_{trt}$ are specified with Gaussian priors, then the exponential of the sum will be a log-normal distribution! So now the model is


\begin{align*}
k_i &\sim \mathrm{Binomial}(n_i, p_i) \\
\mu_\alpha &= \alpha + \alpha_{trt[i]} \\
\mu_\beta &= \beta + \beta_{trt[i]} \\
\mathrm{logit}(p_i) &= \exp(\mu_\beta) (x_i - \mu_\alpha) \\
\alpha &\sim \mathcal{N}(0, 0.06^2) \\
\alpha_{trt} &\sim \mathcal{N}(0, \sigma_{trt}^2) \\
\beta &\sim \mathcal{N}(3, 1^2) \\
\beta_{trt} &\sim \mathcal{N}(0, \gamma_{trt}^2) \\
\sigma_{trt} &\sim \textrm{to be defined} \\
\gamma_{trt} &\sim \textrm{to be defined}
\end{align*}


Deciding on priors for the variance term requires some careful consideration. In one sense, the variance term is the within group variance, but with non-linear models like logistic regression, the logit link can have undesirable or unpredictable floor and ceiling effects. @gelman2006prior recommends that for multilevel models with groups with less than say 5 levels to use a half Cauchy prior with a larger scale parameter. This weakly informative prior still has a regularizing affect and dissuades larger variance estimates. Even though the treatment group only has two levels, there is still value in specifying an adaptive prior for them, and there is also a lot of data for each treatment so partial pooling won't make a difference anyway.


\begin{align*}
\sigma_{trt} &\sim \mathrm{HalfCauchy}(0, 25) \\
\gamma_{trt} &\sim \mathrm{HalfCauchy}(0, 25)
\end{align*}


Finally I can add in the age group level effects and specify the variance terms.


\begin{align*}
k_i &\sim \mathrm{Binomial}(n_i, p_i) \\
\mu_\alpha &= \alpha + \alpha_{trt[i]} + \alpha_{G[i]} \\
\mu_\beta &= \beta + \beta_{trt[i]} + \beta_{G[i]} \\
\mathrm{logit}(p_i) &= \exp(\mu_\beta) (x_i - \mu_\alpha) \\
\alpha &\sim \mathcal{N}(0, 0.06^2) \\
\alpha_{trt} &\sim \mathcal{N}(0, \sigma_{trt}^2) \\
\alpha_{G} &\sim \mathcal{N}(0, \tau_{G}^2)\\
\beta &\sim \mathcal{N}(3, 1^2) \\
\beta_{trt} &\sim \mathcal{N}(0, \gamma_{trt}^2) \\
\beta_{G} &\sim \mathcal{N}(0, \nu_{G}^2) \\
\sigma_{trt} &\sim \mathrm{HalfCauchy}(0, 25) \\
\gamma_{trt} &\sim \mathrm{HalfCauchy}(0, 25) \\
\tau_{G} &\sim \mathrm{HalfCauchy}(0, 25) \\
\nu_{G} &\sim \mathrm{HalfCauchy}(0, 25)
\end{align*}


Here is the corresponding Stan code that also computes the posterior retrodictions and JND and PSS estimates.


\setstretch{1.0}

```stan
data {
  int N;
  int N_G;
  int N_T;
  int n[N];
  int k[N];
  vector[N] x;
  int G[N];
  int trt[N];
}
parameters {
  real a;
  real<lower=machine_precision()> sd_aG;
  real<lower=machine_precision()> sd_aT;
  real aG[N_G];
  real aT[N_T];

  real b;
  real<lower=machine_precision()> sd_bG;
  real<lower=machine_precision()> sd_bT;
  real bG[N_G];
  real bT[N_T];
}
model {
  vector[N] theta;

  a  ~ normal(0, 0.06);
  aG ~ normal(0, sd_aG);
  aT ~ normal(0, sd_aT);
  sd_aG ~ cauchy(0, 25);
  sd_aT ~ cauchy(0, 25);

  b  ~ normal(3.0, 1.0);
  bG ~ normal(0, sd_bG);
  bT ~ normal(0, sd_bT);
  sd_bG ~ cauchy(0, 25);
  sd_bT ~ cauchy(0, 25);

  for (i in 1:N) {
    real mu_a = a + aT[trt[i]] + aG[G[i]];
    real mu_b = b + bT[trt[i]] + bG[G[i]];
    theta[i] = exp(mu_b) * (x[i] - mu_a);
  }

  k ~ binomial_logit(n, theta);
}
generated quantities {
  matrix[N_G, N_T] pss;
  matrix[N_G, N_T] jnd;
  vector[N] k_pred;

  for (i in 1:N_G) {
    for (j in 1:N_T) {
      real mu_b = exp(b + bT[j] + bG[i]);
      real mu_a = a + aT[j] + aG[i];
      pss[i, j] = mu_a;
      jnd[i, j] = logit(0.84) / mu_b;
    }
  }
  
  for (i in 1:N) {
    real mu_a = a + aT[trt[i]] + aG[G[i]];
    real mu_b = b + bT[trt[i]] + bG[G[i]];
    real p = inv_logit(exp(mu_b) * (x[i] - mu_a));
    k_pred[i] = binomial_rng(n[i], p);
  }
}
```
\setstretch{2.0}

**Post-Model, Post-Data**

_Fit Observed Data_

_Diagnose Posterior Fit_

_Posterior Retrodictive Checks_

## Iteration 3 (the one for me){#iter3}

**Pre-Model, Pre-Data**

_Conceptual Analysis_

_Define Observational Space_

_Construct Summary Statistics_

**Post-Model, Pre-Data**

_Develop Model_

_Construct Summary Functions_

_Simulate Bayesian Ensemble_

_Prior Checks_

_Configure Algorithm_

_Fit Simulated Ensemble_

_Algorithmic Calibration_

_Inferential Calibration_

**Post-Model, Post-Data**

_Fit Observed Data_

_Diagnose Posterior Fit_

_Posterior Retrodictive Checks_

## Iteration 4 (what's one more) {#iter4}

**Pre-Model, Pre-Data**

_Conceptual Analysis_

_Define Observational Space_

_Construct Summary Statistics_

**Post-Model, Pre-Data**

_Develop Model_

_Construct Summary Functions_

_Simulate Bayesian Ensemble_

_Prior Checks_

_Configure Algorithm_

_Fit Simulated Ensemble_

_Algorithmic Calibration_

_Inferential Calibration_

**Post-Model, Post-Data**

_Fit Observed Data_

_Diagnose Posterior Fit_

_Posterior Retrodictive Checks_

## Iteration 5 (final_final_draft_2.pdf) {#iter5}

**Pre-Model, Pre-Data**

_Conceptual Analysis_

_Define Observational Space_

_Construct Summary Statistics_

**Post-Model, Pre-Data**

_Develop Model_

_Construct Summary Functions_

_Simulate Bayesian Ensemble_

_Prior Checks_

_Configure Algorithm_

_Fit Simulated Ensemble_

_Algorithmic Calibration_

_Inferential Calibration_

**Post-Model, Post-Data**

_Fit Observed Data_

_Diagnose Posterior Fit_

_Posterior Retrodictive Checks_

## Celebrate

_celebrate_

