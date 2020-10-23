


## Iteration 4 (What's one more?) {#iter4}

We now have a model that works well for the audiovisual data, but there are still two other data sets that we can apply the model to. Additionally there is one more modification that we can make to the model that reflects a real world problem - lapses in judgment.

### Conceptual analysis {#iter4-conceptual}

A lapse in judgment can happen for any reason, and is assumed to be random and independent of other lapses. They can come in the form of the subject accidentally blinking during the presentation of a visual stimulus, or unintentionally pressing the wrong button to respond. Whatever the case is, lapses can have a significant affect on the resulting psychometric function.

### Construct summary statistics {#iter4-summary-stats}

We will continue to use the posterior density of the PSS and JND as summary statistics, but the way we calculate them will change as a result of the change in the model in the next section.

### Model Development {#iter4-model-dev}

Lapses can be modeled as occurring independently at some fixed rate. Fundamentally this means that the underlying performance function, $F$, is bounded by some lower and upper lapse rate. This manifests as a scaling and translation of $F$. For a given lower and upper lapse rate $\lambda$ and $\gamma$, the performance function $\Psi$ is 

$$
\Psi(x; \alpha, \beta, \lambda, \gamma) = \lambda + (1 - \lambda - \gamma) F(x; \alpha, \beta)
$$


\begin{figure}

{\centering \includegraphics[width=0.7\linewidth]{053-bayesian-workflow_files/figure-latex/ch053-plot-pf-with-lapse-1} 

}

\caption{Psychometric function with lower and upper performance bounds.}(\#fig:ch053-plot-pf-with-lapse)
\end{figure}


In certain psychometric experiments, $\lambda$ is interpreted as the lower performance bound or the guessing rate. For example, in certain 2-alternative forced choice (2-AFC) tasks, subjects are asked to respond which of two masses is heavier, and the correctness of their response is recorded. When the masses are the same, the subject can do no better than random guessing. In this task, the lower performance bound is assumed to be 50% as their guess is split between two choices. As the absolute difference in mass grows, the subject's correctness rate increases, though lapses can still happen. In this scenario, $\lambda$ is fixed at $0.5$ and the lapse rate $\gamma$ is a parameter in the model.

Our data does not explicitly record correctness, so we do not give $\lambda$ the interpretation of a guessing rate. Since we are recording proportion of positive responses, we instead treat $\lambda$ and $\gamma$ as lapse rates for negative and positive SOAs. But why should we treat the lapse rates separately? A lapse in judgment can occur independently of the SOA, so $\lambda$ and $\gamma$ should be the same no matter what. With this assumption in mind, we throw away $\gamma$ and assume that the lower and upper performance bounds are restricted by the same amount. I.e.


$$
\Psi(x; \alpha, \beta, \lambda) = \lambda + (1 - 2\lambda) F(x; \alpha, \beta)
$$


While we're throwing in lapse rates, let's also ask the question if different age groups have different lapse rates. To answer this (or rather have our model answer this), we include the new parameter $\lambda_{G[i]}$ into the model so that we get an estimated lapse rate for each age group.

We assume that lapses in judgment are rare, and we know that the rate (or probability of a lapse) is bounded in the interval $[0, 1]$. Because of this, we put a $\mathrm{Beta(4, 96)}$ prior on $\lambda$ which *a priori* puts 99% of the weight below $0.1$ and an expected lapse rate of $0.04$.

We could also set up our model so that information about the lapse rate is shared between age groups (i.e. hierarchical), but we'll leave that as an exercise for the reader.

### Fit the model {#iter4-fit-obs}










```
#> 
#> Divergences:
#> 95 of 20000 iterations ended with a divergence (0.475%).
#> Try increasing 'adapt_delta' to remove the divergences.
#> 
#> Tree depth:
#> 1 of 20000 iterations saturated the maximum tree depth of 10 (0.005%).
#> Try increasing 'max_treedepth' to avoid saturation.
#> 
#> Energy:
#> E-BFMI indicated no pathological behavior.
```

### Posterior retrodictive checks {#iter4-post-retro}





\begin{center}\includegraphics[width=0.7\linewidth]{053-bayesian-workflow_files/figure-latex/ch053-Furious Ninth Xylophone-1} \end{center}






\begin{center}\includegraphics[width=0.7\linewidth]{053-bayesian-workflow_files/figure-latex/ch053-Bulldozer Cold-1} \end{center}



\begin{center}\includegraphics[width=0.7\linewidth]{053-bayesian-workflow_files/figure-latex/ch053-Discarded Firecracker-1} \end{center}
