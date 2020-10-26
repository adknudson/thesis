


# Principled Bayesian Workflow {#workflow}

There are many great resources out there^[citation needed] for following along with an analysis of some data or problem, and much more is the abundance of tips, tricks, techniques, and testimonies to good modeling practices. The problem is that many of these prescriptions are given without context for when they are appropriate to be taken. According to @betancourt2020, this leaves "practitioners to piece together their own model building workflows from potentially incomplete or even inconsistent heuristics." The concept of a principled workflow is that for any given problem, there is not, nor should there be, a default set of steps to take to get from data exploration to predictive inferences. Rather great consideration must be given to domain expertise and the questions that one is trying to answer with the data.

Since everyone asks different questions, the value of a model is not in how well it ticks the boxes of goodness-of-fit checks, but in consistent it is with domain expertise and its ability to answer the unique set of questions. Betancourt suggests answering four questions to evaluate a model by:

1. Domain Expertise Consistency - Is our model consistent with our domain expertise?
2. Computational Faithfulness - Will our computational tools be sufficient to accurately fit our posteriors?
3. Inferential Adequacy - Will our inferences provide enough information to answer our questions?
4. Model Adequacy - Is our model rich enough to capture the relevant structure of the true data generating process?

<br />

- Scope out your problem
- Specify likelihood and priors
- check the model with fake data
- fit the model to the real data
- check diagnostics
- graph fit estimates
- check predictive posterior
- compare models

## Iteration 1 (The Journey of a Thousand Miles Begins with a Single Step) {#iter1}

### Pre-Model, Pre-Data

We begin the modeling process by modeling the experiment according to the description of how it occurred and how the data were collected. This first part consists of conceptual analysis, defining the observational space, and constructing summary statistics that can help us to identify issues in the model specification.

#### Conceptual analysis {#iter1-concept}

In section \@ref(toj-task) we discussed the experimental setup and data collection. To reiterate, subjects are presented with two stimuli separated by some temporal delay, and they are asked to respond as to their perception of the temporal order. There are 45 subjects with 15 each in the young, middle, and older age groups. As the SOA becomes larger in the positive direction, we expect subjects to give more "positive" responses, and as the SOA becomes larger in the negative direction, we expect more "negative" responses. By the way the experiment and responses are constructed, we would not expect to see a reversal of this trend unless there was an issue with the subject's understanding of the directions given to them or an error in the recording device.

We also know that after the first experimental block the subjects go through a recalibration period, and repeat the experiment again. We are interested in seeing if the recalibration has an effect on temporal sensitivity and perceptual synchrony, and if the effect is different for each age group.

#### Define observational space {#iter1-obs-space}

The response that subjects give to a TOJ task is recorded as a zero or a one (see section \@ref(toj-task)), and their relative performance is determined by the SOA value. Let $y$ represent the binary outcome of a trial and let $x$ be the SOA value.

\begin{align*}
y_i &\in \lbrace 0, 1\rbrace \\
x_i &\in \mathbb{R}
\end{align*}

If the SOA values are fixed like in the audiovisual task, then the responses can be aggregated into binomial counts, $k$.

$$
k_i, n_i \in \mathbb{Z}_0^+, k_i \le n_i
$$

In the above equation, $\mathbb{Z}_0^+$ represents the set of non-negative integers. Notice that the number of trials $n$ has an index variable $i$. This is because the number of trials per SOA is not fixed between blocks. In the pre-adaptation block, there are five trials per SOA compared to three in the post-adaptation block. So if observation 32 is recorder during a "pre" block, $n_{32} = 5$, and if observation 1156 is during a "post" block, $n_{1156} = 3$.

Then we also have the three categorical variables -- age group, subject ID, and adaptation. For the first two, we treat them as factor variables^[Factor variables also go by the name index variable or categorical variable]. Rather than using one-hot encoding or dummy variables, we leave the age levels as categories and fit a coefficient for each level. Among the benefits of this approach is the ease of interpretation and ease of working with the data programmatically. This is especially true at the subject level. If we used dummy variables for all 45 subjects, we would have 44 different dummy variables to work with, times the number of coefficients that make estimates at the subject level. In the final iteration of our model, this can be as many as $44 \times 4 = 176$ dummy variables for the subject level!

Age groups and individual subjects can be indexed in the same way that we index the number of trials. $S_i$ refers to the subject in record $i$, and similarly $G_i$ refers to the age group of that subject. Observation 63 is for record ID av-post1-M-f-HG, so then $S_{63}$ is M-f-HG and $G_{63}$ is middle_age. Under the hood of R, these factor levels are represented as integers (e.g. middle age group level is stored internally as the number 2).

We treat the pre- and post-adaptation categories as a binary indicator referred to as $trt$ (short for treatment) since there are only two levels in the category. In this setup, a value of 1 indicates a post-adaptation block. We chose this encoding over the reverse because the pre-adaptation block is like the baseline performance, and it seemed more appropriate to interpret the post-adaptation block as turning on some effect. Using a binary indicator in a regression setting may not be the best practice as we discuss in section \@ref(iter2-model-dev).

We will be using the Stan probabilistic programming language to estimate the model for our data. In the Stan modeling language, data for a binomial model with subject and age group levels and treatment is specified as

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

However, in most of this paper we will be using the `rethinking` package [@R-rethinking] which is a high level wrapper around `rstan`. For most of the model fitting and analyses, it is sufficient. For more complex routines and for finer control, we will utilize `rstan` directly.

#### Construct summary statistics {#iter1-sum-stats}

In order to effectively challenge the validity of our model, we construct a set of summary statistics that help answer the questions of domain expertise consistency and model adequacy. We are studying the affects of age and temporal recalibration through the PSS and JND (see section \@ref(psycho-experiments)), so it is natural to define summary statistics around these quantities to verify model consistency. Additionally the PSS and JND can be computed regardless of the model parameterization or chosen psychometric function.

By the experimental setup and recording process, it is impossible that a properly conducted block would result in a JND less than 0 (i.e. the psychometric function is always non-decreasing), so that can be a lower limit for its threshold. On the other end it is unlikely that it will be beyond the limits of the SOA values, but even more concrete, it seems unlikely (though not impossible) that the just noticeable difference would be more than a second.

<!-- As for the point of subjective simultaneity, it can be either positive or negative, with the belief that larger values are more rare. Some studies suggest that for audio-visual TOJ tasks, the separation between stimuli need to be as little as 20 milliseconds for subjects to be able to determine which modality came first [@vatakis2007influence]. Other studies suggest that our brains can detect temporal differences as small as 30 milliseconds. If we take these studies to heart, then we should be skeptical of PSS estimates larger than say 150 milliseconds in absolute value, just to be safe. -->

A histogram of computed PSS and JND values will suffice for summary statistics. We can estimate the proportion of values that fall outside of our limits defined above, and use them as indications of problems with the model fitting or our conceptual understanding.

<!-- We can further refine the lower bound on the JND if we draw information from other sources. Some studies show that we cannot perceive time differences below 30 ms, and others show that an input lag as small as 100ms can impair a person's typing ability. -->

### Post-Model, Pre-Data

We will now define priors for our model, still not having looked at the data. The priors should be motivated by domain expertise and *prior knowledge*, not the data.

#### Model development {#iter1-model-dev}

Talk here about linear parameterization and connection to PSS and JND.

Talk here about how I select priors for the intercept (PSS) and the slope (JND). Choose a standard deviation for intercept so that $\approx 95\%$ of the values are between $\pm 0.1$

\begin{align*}
\alpha &\sim \mathcal{N}(0, 0.05) \\
\beta &\sim \mathrm{Lognormal}(3.96, 1.2)
\end{align*}

If the expected JND is 0.100 (100 ms) and is distributed log-normally, then $\mathrm{logit}(0.84)/jnd$ is also log-normally distributed with mean $\mathrm{logit}(0.84) - log(0.1) \approx 3.96$.

Choose a standard deviation value so that $\approx 99\%$ of the JND values are less than 1.

The distribution of prior psychometric functions now looks like

<div class="figure" style="text-align: center">
<img src="050-bayesian-workflow_files/figure-html/ch050-prior-pf-plot-1.png" alt="Prior distribution of psychometric functions using the priors for slope and intercept." width="70%" />
<p class="caption">(\#fig:ch050-prior-pf-plot)Prior distribution of psychometric functions using the priors for slope and intercept.</p>
</div>


Notice that the family of psychometric functions covers the broad range of possible slopes and intercepts, though the prior distribution appears to put more weight on steeper slopes (smaller JNDs). There is also too much possibility that the PF is nearly flat. We can reduce the mean-log and sd-log of the slope parameter and get a much more uniform-looking distribution of prior psychometric curves.


\begin{align*}
\alpha &\sim \mathcal{N}(0, 0.05) \\
\beta  z&\sim \mathrm{Lognormal}(3.0, 1.5)
\end{align*}


<div class="figure" style="text-align: center">
<img src="050-bayesian-workflow_files/figure-html/ch050-prior-pf-plot-2-1.png" alt="Second prior distribution of psychometric functions using the priors for slope and intercept." width="70%" />
<p class="caption">(\#fig:ch050-prior-pf-plot-2)Second prior distribution of psychometric functions using the priors for slope and intercept.</p>
</div>

This prior distribution is much more reasonable. There is good prior coverage of both very steep slopes and very shallow slopes, but not so wide that nearly flat or nearly vertical slopes are likely. Also notice how the spread around $y=0.5$ remains the same independent of the slope values. This is because of how the model is parameterized. If instead we parameterized the linear predictor as 

$$
\mathrm{logit}(\pi) = \alpha^* + \beta^* \times x
$$

then the PSS would depend on both $\alpha^*$ and $\beta^*$

$$
\mathrm{PSS}^* = -\frac{\alpha^*}{\beta^*}
$$

while the JND would remain the same

$$
\mathrm{JND}^* = \mathrm{logit}(0.84)/\beta^*
$$

The problem is that it is much harder to define priors for the slope and intercept when they are so closely coupled, and the interpretation of the parameters becomes more difficult as well.

We can now extend the Stan program to include the parameters and model.

\setstretch{1.0}
```
parameters {
  real alpha;          // Intercept (PSS)
  real<lower=0> beta;  // Slope (logit(0.84) / JND)
}
model {
  alpha ~ normal(0, 0.05);      // Prior for intercept
  beta ~ lognormal(3.0, 1.5);   // Prior for slope
  vector[N] p;                  // Binomial probability
  for (i in 1:N) {
    p[i] = beta * (x[i] - alpha);
  }
  k ~ binomial_logit(n, p); // Observational model
}
```
\setstretch{2.0}

#### Construct summary functions {#iter1-summary-funs}

NA

#### Simulate bayesian ensemble {#iter1-sim}

What is the purpose of this step? To make sure that the generating model coupled with the summary stats/functions yields prior estimates that are consistent with domain expertise (see \@ref(iter1-prior-check)).

\setstretch{1.0}

```stan
data {
  int<lower=0> N;
  int n[N];
  vector[N] x;
}
generated quantities {
  real alpha = normal_rng(0, 0.05);
  real beta = lognormal_rng(3.0, 1.5);
  vector[N] theta = inv_logit( beta * (x - alpha) );
  int y_sim[N] = binomial_rng(n, theta);
  real pss = alpha;
  real jnd = logit(0.84) / beta;
}
```
\setstretch{2.0}





#### Prior checks {#iter1-prior-check}

> If the prior predictive checks indicate conflict between the model and our domain expertise then we have to return to step four [(model development)] and refine our model.



```
#>    95%    99%  99.9%   100% 
#>  1.099  2.904  6.681 20.538
```

<img src="050-bayesian-workflow_files/figure-html/ch050-Long Supersonic Cosmic-1.png" width="70%" style="display: block; margin: auto;" /><img src="050-bayesian-workflow_files/figure-html/ch050-Long Supersonic Cosmic-2.png" width="70%" style="display: block; margin: auto;" />


We're satisfied with the prior coverage of the PSS and JND, so now we can move on to fitting the model to the simulated data.

#### Configure algorithm {#iter1-config-algo}

As a default, we will be using the `rethinking` package [@rethinking].

#### Fit simulated ensemble {#iter1-fit-sim}




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
generated quantities {
  real pss = alpha;
  real jnd = logit(0.84) / beta;
}

```
\setstretch{2.0}






#### Algorithmic calibration {#iter1-algo-calibration}

Did the algorithm perform correctly? What kind of diagnostics exist for this algorithm?


\setstretch{1.0}

```
#> 
#> Divergences:
#> 0 of 5000 iterations ended with a divergence.
#> 
#> Tree depth:
#> 0 of 5000 iterations saturated the maximum tree depth of 10.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```
\setstretch{2.0}



```
#>          mean se_mean     sd    2.5%   97.5% n_eff  Rhat
#> alpha -0.0110   1e-04 0.0043 -0.0194 -0.0025  4762 1.000
#> beta   7.8900   3e-03 0.1759  7.5517  8.2420  3385 1.002
#> pss   -0.0110   1e-04 0.0043 -0.0194 -0.0025  4762 1.000
#> jnd    0.2103   1e-04 0.0047  0.2012  0.2196  3369 1.002
```

- Using HMC
  - $\hat{R}$
  - Divergences
  - Effective sample size
  - Tail effective sample size
  - Bulk effective sample size
  - Bayesian fraction of missing information

Is there anything we can tune during the fitting process that can alleviate algorithmic issues? Or is it a case of Folk Theorem, and we need to adjust the model?

#### Inferential calibration {#iter1-inferential-calibration}

Non-identifiable model??

> In either case we might have to return to Step One to consider an improved experimental design or tempered scientific goals. Sometimes we may only need to return to Step Four to incorporate additional domain expertise to improve our inferences.

### Post-Model, Post-Data

#### Fit observation {#iter1-fit-obs}


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
generated quantities {
  real pss = alpha;
  real jnd = logit(0.84) / beta;
  int y_post_pred[N];
  for (i in 1:N)
    y_post_pred[i] = binomial_rng(n[i], inv_logit(beta * (x[i] - alpha)));
}
```
\setstretch{2.0}





#### Diagnose posterior fit {#iter1-diagnose-post}

> If any diagnostics indicate poor performance then not only is our computational method suspect but also our model might not be rich enough to capture the relevant details of the observed data. At the very least we should return to Step Eight and enhance our computational method.


\setstretch{1.0}

```
#> 
#> Divergences:
#> 0 of 5000 iterations ended with a divergence.
#> 
#> Tree depth:
#> 0 of 5000 iterations saturated the maximum tree depth of 10.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```


```
#>         mean se_mean     sd   2.5%  97.5% n_eff   Rhat
#> alpha 0.0372   1e-04 0.0041 0.0292 0.0453  5217 0.9995
#> beta  8.4235   3e-03 0.1832 8.0731 8.7861  3814 0.9994
#> pss   0.0372   1e-04 0.0041 0.0292 0.0453  5217 0.9995
#> jnd   0.1970   1e-04 0.0043 0.1887 0.2054  3828 0.9994
```
\setstretch{2.0}


#### Posterior retrodictive checks {#iter1-post-retro}

Need an example of using summary stats on posterior retrodictions




<img src="050-bayesian-workflow_files/figure-html/ch050-Angry Pineapple-1.png" width="70%" style="display: block; margin: auto;" />

The posterior retrodictions do well to cover the observed data, but we don't actually have a model that can answer the questions that we are seeking to answer. At best, this model can only inform of of the population average PSS and JND across both pre- and post-adaptation. This first iteration does serve as a useful foundation for building a more complex model, and for practicing visualization techniques that will be more important in the next few iterations.

While we are here, let's also take a look at the underlying performance (psychometric) function as well as the density estimates of the PSS and JND.

<img src="050-bayesian-workflow_files/figure-html/ch050-Swift Strong Xylophone-1.png" width="70%" style="display: block; margin: auto;" />

We can see that the distribution of psychometric curves never wanders too far off from the mean. In this sense, we can be fairly confident that the PSS is positive.

<img src="050-bayesian-workflow_files/figure-html/ch050-Hollow Mustard-1.png" width="70%" style="display: block; margin: auto;" />

