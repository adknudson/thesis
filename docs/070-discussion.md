# Discussion {#discussion}

## Project History

### Data Cleaning

- Data doesn't always come in a nice tidy format
- Metadata came in the form of directory names
- Additional metadata (clinical data) came from a separate excel file
  - Ages were provided, but not age groups
- Actual data were recorded into a Matlab file
- Not every task had the same column data
- Not every directory/file name was spelled correctly
  - Adaptation, adapat, 
- Had to create a unique identifier for each subject
  - subjects with the same initials
  - [task, trial,] age group, gender, and initials
- Some subjects with two-letter initials, others with three, and yet others that were singly unique
  - JM, JM_F
  - DTF
  - IV_23_, DD_21_
- Aggregating Bernoulli trials into binomial counts
  - Sufficient statistics
- Tidying the data
  - Tidy Data by Hadley Wickham
  - each row is an individual observation

### Developing a model

- I knew I wanted to apply Bayesian modeling techniques to the data
- Tried using classical GLM fit first to get a baseline understanding of the data
  - The fact that some estimates for certain subjects failed due to complete separation reinforced my enthusiasm to employ non-classical techniques
- First model was derived from the psychometric function from Bayesian Cognitive Modeling
  - Used nested loops to iterate over subjects and SOA values
  - Data was messy and the model was hard to build off of
- Tried `bayes_glm` and `rstanarm`
- Moved on to multilevel modeling
  - "Statistical Rethinking" by McElreath was my first introduction to Bayesian multilevel modeling
    - His `rethinking` package is too good to be just a book package
  - Tried writing my own package that generates a Stan program based on R formula syntax
    - didn't fully understand the concepts of no-pooling, complete pooling, and partial pooling
    - didn't understand fixed/random effects models
    - Was shot down quickly by some kind folks on Reddit who told me that `brms` and `rstanarm` already did what I was trying to do
      - `brms` and `rstanarm` actually did it better, but I'm not bitter. At all...
    - The fossilized remains of my attempt can be viewed on github
  - Tried using `lme4`, `rstanarm`, and `brms` for random effects modeling
    - was still limited by the lack of control over parameterization and link functions
    - Couldn't easily include a lapse rate parameter into these packages
- Realized that parameterization can have a significant affect on the efficiency of a model and the inferential power of the estimated parameters
  - When fitting a classical model, there is little difference in estimating `a + bx` vs. `d(x - c)` since the latter can just be expanded as `-cd + dx` which is essentially the same as the first parameterization, but there is a practical difference in the interpretation of the parameters
    - The second parameterization implies that there is a dependence among the parameters that can be factored out
    - In the context of psychometric functions, there is a stronger connection between PSS and `c` and the JND and `d`
    - This reparameterization made it easier to specify priors and also increased the model efficiency
- Visualization and making shiny plots
  - For the end of STAT 629 - Intro to Bayesian Statistics with Dr. Schissler, I created an interactive graph that lets you select a subject, plot a sample of their estimated psychometric functions, and plot the marginal distributions of SOA values given a probability and probability given an SOA value
- Still issues with parameterization
  - Treating the condition as an indicator/dummy variable has implications on the variance of predictions
  - Using an indicator variable also introduced an interaction effect into the model that I almost did not account for
  - Switched to index/factor variables helped with model fitting, and interaction effects between factors is handled by creating a new factor that is essentially the cross-product of other factor variables
    - E.g. x=[a, b, c], y=[i, j] --> xy=[ai, aj, bi, bj, ci, cj]
- One more bout of reparameterization
  - To us, $Z \sim N(0, 1^2);\quad X = 3 + 2Z$ is the same as saying $X \sim N(3, 2^2)$, but to a computer, the process of sampling from $X$ can be more difficult than sampling from $Z$
  - Hierarchical models can benefit greatly from non-centered parameterization
  - Results in more efficient exploration of the posterior, higher number of effected samples, and fewer divergent transitions
- Side quest: helped explain logistic regression to a fellow who is taking a statistical psychology course in New Zealand
  - He was interested in fitting a 2-AFC model in Julia, which did not provide a link function for such psychometric experiments
  - In the end of our discussion, I wrote my own N-AFC link function that can be dropped in to the `GLM.jl` package

## Model selection is not always the goal

- Building a model motivated by a set of principles and domain expertise should be the preferred way
- Model comparison is important, especially in terms of predictive inference
- One model doesn't fit all
  - Different models help to answer different questions
    - Mean effect, individual effect, predictive density, conditional densities, etc

