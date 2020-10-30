

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

\begin{table}[!h]

\caption{(\#tab:ch030-workflow-steps)Principled workflow}
\centering
\begin{tabular}[t]{ll}
\toprule
Part & Step\\
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

I'll talk about each step in the first iteration, but may choose to omit steps in subsequent iterations if there are no changes. For the purposes of building a model and being concise, I will focus around the audiovisual TOJ task in this chapter, but the final model will apply similarly to the visual and duration tasks. For the sensorimotor task, the model will be modified to accept Bernoulli data as apposed to aggregated Binomial counts (described more in the next section).

## Iteration 1 (journey of a thousand miles) {#iter1}

**pre-model, pre-data**

I begin the modeling process by modeling the experiment according to the description of how it occurred and how the data were collected. This first part consists of conceptual analysis, defining the observational space, and constructing summary statistics that can help us to identify issues in the model specification.

_conceptual analysis_

In section \@ref(toj-task) I discussed the experimental setup and data collection. To reiterate, subjects are presented with two stimuli separated by some temporal delay, and they are asked to respond as to their perception of the temporal order. There are 45 subjects with 15 each in the young, middle, and older age groups. As the SOA becomes larger in the positive direction, subjects are expected to give more "positive" responses, and as the SOA becomes larger in the negative direction, more "negative" responses are expected. By the way the experiment and responses are constructed, there is no expectation to see a reversal of this trend unless there was an issue with the subject's understanding of the directions given to them or an error in the recording device.

After the first experimental block the subjects go through a recalibration period, and repeat the experiment again. The interest is in seeing if the recalibration has an effect on temporal sensitivity and perceptual synchrony, and if the effect is different for each age group.

_define observational space_

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

_construct summary statistics_

In order to effectively challenge the validity of the model, a set of summary statistics are constructed that help answer the questions of domain expertise consistency and model adequacy. We are studying the affects of age and temporal recalibration through the PSS and JND (see section \@ref(psycho-experiments)), so it is natural to define summary statistics around these quantities to verify model consistency. Additionally the PSS and JND can be computed regardless of the model parameterization or chosen psychometric function.

By the experimental setup and recording process, it is impossible that a properly conducted block would result in a JND less than 0 (i.e. the psychometric function is always non-decreasing), so that can be a lower limit for its threshold. On the other end it is unlikely that it will be beyond the limits of the SOA values, but even more concrete, it seems unlikely (though not impossible) that the just noticeable difference would be more than a second.

The lower bound on the JND can be further refined if we draw information from other sources. Some studies show that we cannot perceive time differences below 30 ms, and others show that an input lag as small as 100ms can impair a person's typing ability. Then according to these studies, a time delay of 100ms is enough to notice, and so a just noticeable difference should be much less than one second -- much closer to 100ms. I'll continue to use one second as an extreme estimate indicator, but will incorporate this knowledge when it comes to selecting priors.

As for the point of subjective simultaneity, it can be either positive or negative, with the belief that larger values are more rare. Some studies suggest that for audio-visual TOJ tasks, the separation between stimuli need to be as little as 20 milliseconds for subjects to be able to determine which modality came first [@vatakis2007influence]. Other studies suggest that our brains can detect temporal differences as small as 30 milliseconds. If these values are to be believed then we should be skeptical of PSS estimates larger than say 150 milliseconds in absolute value, just to be safe.

A histogram of computed PSS and JND values will suffice for summary statistics. We can estimate the proportion of values that fall outside of our limits defined above, and use them as indications of problems with the model fitting or conceptual understanding.

**post-model, pre-data**

It is now time to define priors for the model, while still not having looked at the [distribution of] data. The priors should be motivated by domain expertise and *prior knowledge*, not the data. There are also many choices when it comes to selecting a psychometric (sigmoid) function. Common ones are logistic, Gaussian, and Weibull.



\begin{center}\includegraphics[width=0.7\linewidth]{030-workflow_files/figure-latex/ch031-Eastern Needless Autopsy-1} \end{center}


The Weibull psychometric function is more common when it comes to 2-AFC psychometric experiments where the independent variable is a stimulus intensity (non-negative) and the goal is signal detection. The data in this paper includes both positive and negative SOA values, so the Weibull is not a natural choice. In fact, because this is essentially a model for logistic regression, my first choice is the logistic function as it is the canonical choice for Binomial data. Additionally, the data in this study are reversible. The label of a positive response can be swapped with the label of a negative response and the inferences should remain the same. Since there is no natural ordering, it makes more sense for the psychometric function to be symmetric, e.g. the logistic and Gaussian. I use symmetric loosely to mean that probability density function (PDF) is symmetric about its middle. More specifically, the distribution has zero skewness.

In practice, there is little difference in inference between the _logit_ and _probit_ links, but computationally the logit link is more efficient. I am also more familiar with working on the log-odds scale compared to the probit scale, so I make the decision to go forward with the logistic function. In [chapter 4](#model-checking) I will show how even with a mis-specified link function, we can still achieve accurate predictions.

_develop model_



_construct summary functions_

_simulate Bayesian ensemble_

_prior checks_

_configure algorithm_

_fit simulated ensemble_

_algorithmic calibration_

_inferential calibration_

**post-model, post-data**

_fit observed data_

_diagnose posterior fit_

_posterior retrodictive checks_

## Iteration 2 (electric boogaloo) {#iter2}

**pre-model, pre-data**

_conceptual analysis_

_define observational space_

_construct summary statistics_

**post-model, pre-data**

_develop model_

_construct summary functions_

_simulate Bayesian ensemble_

_prior checks_

_configure algorithm_

_fit simulated ensemble_

_algorithmic calibration_

_inferential calibration_

**post-model, post-data**

_fit observed data_

_diagnose posterior fit_

_posterior retrodictive checks_

## Iteration 3 (the one for me){#iter3}

**pre-model, pre-data**

_conceptual analysis_

_define observational space_

_construct summary statistics_

**post-model, pre-data**

_develop model_

_construct summary functions_

_simulate Bayesian ensemble_

_prior checks_

_configure algorithm_

_fit simulated ensemble_

_algorithmic calibration_

_inferential calibration_

**post-model, post-data**

_fit observed data_

_diagnose posterior fit_

_posterior retrodictive checks_

## Iteration 4 (what's one more) {#iter4}

**pre-model, pre-data**

_conceptual analysis_

_define observational space_

_construct summary statistics_

**post-model, pre-data**

_develop model_

_construct summary functions_

_simulate Bayesian ensemble_

_prior checks_

_configure algorithm_

_fit simulated ensemble_

_algorithmic calibration_

_inferential calibration_

**post-model, post-data**

_fit observed data_

_diagnose posterior fit_

_posterior retrodictive checks_

## Iteration 5 (final_final_draft_2.pdf) {#iter5}

**pre-model, pre-data**

_conceptual analysis_

_define observational space_

_construct summary statistics_

**post-model, pre-data**

_develop model_

_construct summary functions_

_simulate Bayesian ensemble_

_prior checks_

_configure algorithm_

_fit simulated ensemble_

_algorithmic calibration_

_inferential calibration_

**post-model, post-data**

_fit observed data_

_diagnose posterior fit_

_posterior retrodictive checks_

## Celebrate

_celebrate_

