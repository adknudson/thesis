# (APPENDIX) Appendix {-}


# Supplementary Code {#code}


**Eight Schools Model**


\setstretch{1.0}
```
data {
  int<lower=0> J;         // number of schools 
  real y[J];              // estimated treatment effects
  real<lower=0> sigma[J]; // standard error of effect estimates 
}
parameters {
  real mu;                // population treatment effect
  real<lower=0> tau;      // standard deviation in treatment effects
  vector[J] eta;          // unscaled deviation from mu by school
}
transformed parameters {
  vector[J] theta = mu + tau * eta; // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(y | theta, sigma); // log-likelihood
}
generated quantities {
  vector[J] log_lik;

  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
  }
}
```
\setstretch{2.0}


\clearpage
**Model with Lapse and Subject-level Parameters**


\setstretch{1.0}

```
data {
  int N;        // Number of observations
  int N_G;      // Number of age groups
  int N_T;      // Number of treatments
  int N_S;      // Number of subjects
  int n[N];     // Number of Bernoulli trials
  int k[N];     // Number of "positive" responses
  vector[N] x;  // SOA values
  int G[N];     // Age group index variable
  int trt[N];   // Treatment index variable
  int S[N];     // Subject index variable
}
parameters {
  real a_raw;
  real<lower=machine_precision(),upper=pi()/2> aGT_unif;
  real<lower=machine_precision(),upper=pi()/2> aS_unif;
  matrix[N_G, N_T] aGT_raw;
  vector[N_S] aS_raw;

  real b_raw;
  real<lower=machine_precision(),upper=pi()/2> bGT_unif;
  real<lower=machine_precision(),upper=pi()/2> bS_unif;
  matrix[N_G, N_T] bGT_raw;
  vector[N_S] bS_raw;

  vector[N_G] lG;
}
transformed parameters {
  real a;
  matrix[N_G, N_T] aGT;
  vector[N_S] aS;
  real sd_aGT;
  real sd_aS;

  real b;
  matrix[N_G, N_T] bGT;
  vector[N_S] bS;
  real sd_bGT;
  real sd_bS;

  a = a_raw * 0.06;
  sd_aGT = tan(aGT_unif);
  sd_aS  = tan(aS_unif);
  aS = aS_raw * sd_aS;

  b = 3.0 + b_raw;
  sd_bGT = 2 * tan(bGT_unif);
  sd_bS  = 2 * tan(bS_unif);
  bS = bS_raw * sd_bS;

  for (i in 1:N_G) {
    for (j in 1:N_T) {
      aGT[i, j] = aGT_raw[i, j] * sd_aGT;
      bGT[i, j] = bGT_raw[i, j] * sd_bGT;
    }
  }
}
model {
  vector[N] p;

  a_raw ~ std_normal();
  b_raw ~ std_normal();
  lG ~ beta(4, 96);

  aS_raw ~ std_normal();
  bS_raw ~ std_normal();
  to_vector(aGT_raw) ~ std_normal();
  to_vector(bGT_raw) ~ std_normal();

  for (i in 1:N) {
    real alpha = a + aGT[G[i], trt[i]] + aS[S[i]];
    real beta = b + bGT[G[i], trt[i]] + bS[S[i]];
    real lambda = lG[G[i]];
    p[i] = lambda + (1 - 2*lambda) * inv_logit(exp(beta) * (x[i] - alpha));
  }

  k ~ binomial(n, p);
}
generated quantities {
  vector[N] log_lik;
  vector[N] k_pred;

  for (i in 1:N) {
    real alpha = a + aGT[G[i], trt[i]] + aS[S[i]];
    real beta  = b + bGT[G[i], trt[i]] + bS[S[i]];
    real lambda = lG[G[i]];

    real p = lambda + (1 - 2*lambda) * inv_logit(exp(beta) * (x[i] - alpha));

    log_lik[i] = binomial_lpmf(k[i] | n[i], p);
    k_pred[i]  = binomial_rng(n[i], p);
  }
}
```
\setstretch{2.0}


\clearpage
**Stan Algebraic Solver**


\setstretch{1.0}
```
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
\setstretch{2.0}


# Developing a Model {#model-dev}


Our final modeling strategy is an evolution from other attempts. The development proceeded through multiple iterations described in [chapter 4](#application), but doesn't tell the full story. We learn more from others when they share what didn't work along with the final path that did work. There is knowledge to be gained in failed experiments, because then there is one more way to not do something, just like a failing outcome reduces the variance of the Beta distribution.


In the first attempt at modeling, we used a classical GLM to get a baseline understanding of the data, but the fact that some estimates for certain subjects failed due to complete separation reinforced the our adoption of non-classical techniques. Our first Bayesian model was derived from @lee2014bayesian which used nested loops to iterate over subjects and SOA values. The data were required to be stored in a complicated way that made it difficult to comprehend and extend.


We moved on to using `arm::bayesglm` to remove convergence issues, but we were met with other limitations such as linear parameterization and lack of hierarchical modeling. The book Statistical Rethinking [@mcelreath2020statistical] offers a great first introduction to Bayesian multilevel modeling. McElreath's `rethinking` package accompanies the book, and offers a compact yet expressive syntax for models that get translated into a Stan model. A model with age group and block can be written using `rethinking::ulam` as


\setstretch{1.0}

```r
rethinking::ulam(alist(
  k ~ binomial_logit(n, p),
  p = exp(b + bG[G] + bT[trt]) * (x - (a + aG[G] + aT[trt])),
  a ~ normal(0, 0.06),
  aG[G] ~ normal(0, sd_aG),
  aT[trt] ~ normal(0, sd_aT),
  b ~ normal(3, 1),
  bG[G] ~ normal(0, sd_bG),
  bT[trt] ~ normal(0, sd_bT),
  c(sd_aG, sd_aT, sd_bG, sd_bT) ~ half_cauchy(0, 5)
), data = df, chains = 4, cores = 4, log_lik = TRUE)
```
\setstretch{2.0}


While learning about multilevel models, we tried writing a package that generates a `Stan` program based on `R` formula syntax. At the time the concepts of no-pooling, complete pooling, and partial pooling were vaguely understood, and the package was plagued by the same lack of flexibility that `rstanarm` and `brms` have. Then it was discovered that `brms` and `rstanarm` already did what we were trying to do, but programming experience was invaluable.


We also tried using `lme4`, `rstanarm`, and `brms`, and learned more about the concepts of fixed and random effects. We noticed that parameterization can have a significant affect on the efficiency of a model and the inferential power of the estimated parameters. When fitting a classical model, there is little difference in estimating `a + bx` vs. `d(x - c)` since the latter can just be expanded as `-cd + dx` which is essentially the same as the first parameterization, but there is a practical difference in the interpretation of the parameters. The second parameterization implies that there is a dependence among the parameters that can be factored out. In the context of psychometric functions, there is a stronger connection between PSS and `c` and the JND and `d`. This parameterization made it easier to specify priors and also increased the model efficiency. Of the modeling tools mentioned, only `rethinking` and `Stan` allow for arbitrary parameterization.


We finally arrived at a model that worked well, but learned that using a binary indicator variable for the treatment comes with the assumption of higher uncertainty for one of the conditions. The linear model that we arrived at is displayed in equation \@ref(eq:badlinearmodel).


\begin{equation}
  \theta = \exp(\beta + \beta_G +(\beta_T + \beta_{TG})\times trt) \left[x - (\alpha + \alpha_G + (\alpha_T + \alpha_{TG})\times trt)\right]
  (\#eq:badlinearmodel)
\end{equation}


Using an indicator variable in this fashion also introduced an interaction effect into the model that we almost did not account for after switching to using a factor variable. Interaction effects between factors is handled by creating a new factor that is essentially the cross-product of other factor variables. E.g. for factor variables $x$ and $y$


\setstretch{1.0}
$$
x = \begin{bmatrix}
a \\
b \\
c
\end{bmatrix}, y =  \begin{bmatrix}
i \\
j
\end{bmatrix}\Longrightarrow x\times y = 
\begin{bmatrix}
ai & aj \\
bi & bj \\
ci & cj
\end{bmatrix}
$$
\setstretch{2.0}


The final round of reparameterization came in the form of adopting non-centered parameterization for more efficient models. To us, $Z \sim N(0, 1^2);\quad X = 3 + 2Z$ is the same as $X \sim N(3, 2^2)$, but to a computer the process of sampling from $X$ can be more difficult than sampling from $Z$ (discussed in [chapter 2](#methods)).


# Reproducible Results {#reproduce}





Data doesn't always come in a nice tidy format, and we had to turn the raw experimental data into a clean data set that is ready for modeling. Sometimes the process is quick and straight forward, but other times, like with this psychometric data, it takes more effort and clever techniques. There is academic value in describing the steps taken to reduce the headache later.


To begin, there is a strong push in recent years for reproducible data science. Scientific methods and results should be able to be replicated by other researchers, and part of that includes being able to replicate the process that takes the raw data and produces the tidy data that is ready for analysis. Tidy data is described by @wickham2014tidy and can be summed up by three principles


\setstretch{1.0}
1. Each variable forms a column
2. Each observation forms a row
3. Each type of observational unit forms a table
\setstretch{2.0}


One problem is having data in a spread sheet, modifying it, and then having no way of recovering the original data. Spread sheets are a convenient way to organize, transform, and lightly analyze data, but problems can quickly arise unless there is a good backup/snapshot system in place. Mutability in computer science is the property of a data structure where its contents can be modified in place. Immutability means that the object cannot be modified without first making a copy.  Data is immutable, or at least that is the mindset that researchers must adopt in order to have truly reproducible workflows. The raw data that is collected or produced by a measurement device should never be modified without first being copied, even if for trivial reasons such as correcting a spelling mistake. If a change is made to the raw data, it should be carefully documented and reversible.


To begin the data cleaning journey, we introduce the directory system that we had been given to work with. Each task is separated into its own folder, and within each folder is a subdirectory of age groups.


\begin{figure}

{\centering \includegraphics[width=0.3\linewidth]{figures/data_dir} 

}

\caption{Top-level data directory structure.}(\#fig:ch230-Lama-Everyday)
\end{figure}


Within each age group subdirectory are the subdirectories for each subject named by their initials which then contain the experimental data in Matlab files.


\begin{figure}

{\centering \includegraphics[width=0.35\linewidth]{figures/data_subdir} 

}

\caption{Subdirectory structure.}(\#fig:ch230-Third-Needless-Antique)
\end{figure}


The data appears manageable, and there is information contained in the directory structure such as task, age group, and initials, and file name contains information about the experimental block. There is also an excel file that we were later given that contains more subject information like age and sex, though that information is not used in the model. The columns of the Matlab file depend on the task, but generally they contain an SOA value and a response, but no column or row name information -- that was provided by the researcher who collected the data. 


We then created a table of metadata -- information extracted from the directory structure and file names combined with the the subject data and the file path. Regular expressions can be used to extract patterns from a string. With a list of all Matlab files within the `RecalibrationData` folder, we tried to extract the task, age group, initials, and block using the regular expression:


```
"^(\\w+)/(\\w+)/(\\w+)/[A-Z]{2,3}_*[A-Z]*(adapt[0-9]|baseline[0-9]*).*"
```


The `^(\\w+)/` matches any word characters at the start and before the next slash. Since the directory is structured as `Task/AgeGroup/Subject/file.mat`, the regular expression should match three words between slashes. The file name generally follows the pattern of `Initials__block#__MAT.mat`, so `[A-Z]{2,3}_*[A-Z]*` should match the initials, and `(adapt[0-9]|baseline[0-9]*)` should match the block (baseline or adapt). This method works for $536$ of the $580$ individual records. For the ones it failed, it was generally do to misspellings or irregular capitalizing of "baseline" and "adapt".


Since there is only a handful of irregular block names, they can be dealt with by a separate regular expression that properly extracts the block information. Other challenges in cleaning the data include the handling of subjects with the same initials. This becomes a problem when filtering by a subject's initials is not guaranteed to return a unique subject. Furthermore there are two middle age subjects with the same initials of "JM", so one was also identified with their sex "JM_F". The solution is to create a unique identifier (labeled as SID) that is a combination of age group, sex, and initials. For an experiment identifier (labeled as RID), the task and block were prepended to the SID. Each of these IDs uniquely identify the subjects and their experimental records making it easier to filter and search.


\setstretch{1.0}

```r
glimpse(features, width = 60)
#> Rows: 580
#> Columns: 8
#> $ rid       <fct> av-post1-M-f-CC, av-post1-M-f-DB, av-...
#> $ sid       <fct> M-f-CC, M-f-DB, M-f-HG, M-f-JM, M-f-M...
#> $ path      <chr> "Audiovisual/MiddleAge/CC/CCadapt1__M...
#> $ task      <chr> "audiovisual", "audiovisual", "audiov...
#> $ trial     <fct> post1, post1, post1, post1, post1, po...
#> $ age_group <fct> middle_age, middle_age, middle_age, m...
#> $ age       <dbl> 39, 44, 41, 48, 49, 43, 47, 49, 49, 4...
#> $ sex       <fct> F, F, F, F, F, F, F, F, F, M, M, M, M...
```
\setstretch{2.0}


Then with the table of clean metadata, the task is simply to loop through each row, read the Matlab file given by `path`, add the unique ID as a column, and then join the experimental data with the metadata to create a data set that is ready for model fitting and data exploration. The full code used to generate the clean data is not yet available online, but can be shared with the committee.


The benefit of writing a script to generate the data is that others can look over the code and verify that it is doing what it is intended to do, and we can go back to any step within the process to make changes if the need comes up. Another tool that contributed to the reproducibility is the version control management software, Git. With Git we can take a snapshot of the changes made, and revert if necessary. This thesis is also hosted on Github, and the entire history of development can be viewed there.
