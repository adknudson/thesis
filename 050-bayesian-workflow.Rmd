# Principled Bayesian Workflow {#workflow}

There are many great resources out there^[citation needed] for following along with an analysis of some data or problem, and much more is the abundance of tips, tricks, techniques, and testimonies to good modeling practices. The problem is that many of these prescriptions are given without context for when they are appropriate to be taken. According to @betancourt2020, this leaves "practitioners to piece together their own model building workflows from potentially incomplete or even inconsistent heuristics." The concept of a principled workflow is that for any given problem, there is not, nor should there be, a default set of steps to take to get from data exploration to predictive inferences. Rather great consideration must be given to domain expertise and the questions that one is trying to answer with the data.

Since everyone asks different questions, the value of a model is not in how well it ticks the boxes of goodness-of-fit checks, but in consistent it is with domain expertise and its ability to answer the unique set of questions. Betancourt suggests answering four questions to evaluate a model by:

1. Domain Expertise Consistency - Is our model consistent with our domain expertise?
2. Computational Faithfulness - Will our computational tools be sufficient to accurately fit our posteriors?
3. Inferential Adequacy - Will our inferences provide enough information to answer our questions?
4. Model Adequacy - Is our model rich enough to capture the relevant structure of the true data generating process?



<br />



<br />

- Scope out your problem
- Specify likelihood and priors
- check the model with fake data
- fit the model to the real data
- check diagnostics
- graph fit estimates
- check predictive posterior
- compare models


## Pre-Model, Pre-Data

### Conceptual analysis



### Define observational space

### Construct summary statistics

## Post-Model, Pre-Data

### model development

### Construct summary functions

### simulate bayesian ensemble

### Prior checks

### configure algorithm

### fit simulated ensemble

### algorithmic calibration

### inferential calibration

## Post-Model, Post-Data

### fit observation

### diagnose posterior fit

### Posterior retrodictive checks

### Celebrate
