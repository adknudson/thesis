---
title: "Contributions to Modern Bayesian Multilevel Modeling"
author: "Alexander Knudson"
advisor: "A.G. Schissler"
date: "August, 2020"
site: bookdown::bookdown_site
bibliography: [bibliography.bib]
biblio-style: apalike
link-citations: yes
github-repo: adkudson/thesis
---

# Introduction

- Soft intro to modern computing, analysis
  - advance in CPU+programming leads to evolved statistical methods
    - ML/DL, high dimensional analysis, big data, Bayesian techniques
  - multidisciplinary techniques; numerical methods, probability theory, statistics, computer science, visualizations, etc
    - root finding, change of variables, Gaussian quadrature, Hermite polynomials, Monte Carlo simulation, floating point arithmetic
  - Organization and reproducibility are crucial in a data analysis setting
    - pre-planning, modularity, workflows, versioning, virtual environments, DRY programming, code that is easy to read
  - Clean data is important for good modeling
    - garbage in leads to garbage out



With the advances in computational power and high-level programming languages like Python, R, and Julia, statistical methods have evolved to be more flexible and expressive. 



- Overview of classical modeling methods
  - classical approaches to data analysis usually adhere to the flexibility-interpretability trade-off
  - generally inflexible (parametric) to be more interpretable and computationally easier
  - sometimes a model is too flexible (non-parametric) and loses crucial inferential power
  - sometimes our assumptions about the data are invalid
    - normality, independence, heteroskedacity, etc.
  - often limited when it comes to statistical summaries and confidence intervals



- Solutions or alternatives when classical models fail
  - Bayesian inference is a powerful, descriptive, and flexible modeling framework
  - Bayes theorem is a simple model of incorporating prior information and data to produce a posterior probability or distribution
  - $P(\theta | X) \propto P(X | \theta) * P(\theta)$ or $posterior \propto prior \times likelihood$
    - The prior is some distribution over the parameter space
    - The likelihood is the probability of an outcome in the sample space given a value in the parameter space
    - The posterior is the likelihood of values in the parameter space after observing values from the sample space
  - Bayesian statistics, when described without math, actually feels natural to most people
    - you hear hoof beats, you think horses, not zebras [unless you're in Africa, but that's prior information ;)]
  - The catch is that the model is not complete as written above
  - There is actually a denominator in Bayes' Theorem
    - $P(\theta | X) = \frac{P(X | \theta)\cdot P(\theta)}{\sum_i P(X | \theta_i)} = \frac{P(X | \theta)\cdot P(\theta)}{\int_\Omega P(X | \theta)d\theta}$
    - In general, the denominator is not known, or is not not easy (or possible) to calculate, but it always evaluates to a constant (hence the "proportional to")
    - The denominator acts as a scaling value that forces $P(\theta|X)$ to be a probability distribution (i.e. area under PDF is equal to 1)
    - There are simulation-based techniques that let one approximate the posterior distribution without needing to know the analytic solution to the denominator



I have organized this thesis as follows. In [Chapter 2](#motivating-data) I introduce the data set that drives the narrative and that motivates the adoption of Bayesian multilevel modeling. In [Chapter 3](#background) there is a review of common approaches approaches to modeling with psychometric data, and the benefits and drawbacks of such techniques. [Chapter 4](#bayesian-modeling) introduces Bayesian hierarchical modeling and programming frameworks for Bayesian inference. In [Chapter 5](#workflow) I describe and work through a principled Bayesian workflow for multilevel modeling. [Chapter 6](#model-checking) goes into more depth on checking the model goodness of fit and model diagnostics in a Bayesian setting. Finally in [Chapter 7](#predictive-inference) I demonstrate how to use the Bayesian model from the principled workflow for predictive inference, and use posterior predictive distributions to plot and compare models.
