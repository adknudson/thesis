


# Psychometric Results {#results}

## Affect of Adaptation across Age Groups

### On Perceptual Synchrony

**Audiovisual TOJ Task**

<img src="060-results_files/figure-html/ch060-Eastern-Cat-1.png" width="85%" style="display: block; margin: auto;" />


**Visual TOJ Task**

<img src="060-results_files/figure-html/ch060-Gruesome-Waffle-1.png" width="85%" style="display: block; margin: auto;" />

**Duration TOJ Task**

<img src="060-results_files/figure-html/ch060-Stormy-Frostbite-1.png" width="85%" style="display: block; margin: auto;" />

**Sensorimotor TOJ Task**

<img src="060-results_files/figure-html/ch060-Homeless-Anaconda-1.png" width="85%" style="display: block; margin: auto;" />

### On Temporal Sensitivity

**Audiovisual TOJ Task**

<img src="060-results_files/figure-html/ch060-Timely-Toupee-1.png" width="85%" style="display: block; margin: auto;" />


**Visual TOJ Task**

<img src="060-results_files/figure-html/ch060-Mercury-Rainbow-1.png" width="85%" style="display: block; margin: auto;" />

**Duration TOJ Task**

<img src="060-results_files/figure-html/ch060-Aimless-Planet-1.png" width="85%" style="display: block; margin: auto;" />

**Sensorimotor TOJ Task**

<img src="060-results_files/figure-html/ch060-Tombstone-Cold-1.png" width="85%" style="display: block; margin: auto;" />

## Lapse Rates across Age Groups


<div class="figure" style="text-align: center">
<img src="060-results_files/figure-html/ch060-Waffle-Hollow-1.png" alt="Process model of the result of a psychometric experiment with the assumption that lapses occur at random and at a fixed rate, and that the subject guesses randomly in the event of a lapse." width="85%" />
<p class="caption">(\#fig:ch060-Waffle-Hollow)Process model of the result of a psychometric experiment with the assumption that lapses occur at random and at a fixed rate, and that the subject guesses randomly in the event of a lapse.</p>
</div>

In the above figure, the outcome of one experiment can be represented as a directed acyclic graph (DAG) where at the start of the experiment, the subject either experiences a lapse in judgment with probability $\gamma$ or they do not experience a lapse in judgment. If there is no lapse, then they will give a positive response with probability $F(x)$. If there is a lapse in judgment, then it is assumed that they will respond randomly - e.g. a fifty-fifty chance of a positive response. In this model of an experiment, the probability of a positive response is the sum of the two paths.


\begin{align}
\mathrm{P}(\textrm{positive}) &= 
  \mathrm{P}(\textrm{lapse}) \cdot \mathrm{P}(\textrm{positive} | \textrm{lapse}) + 
  \mathrm{P}(\textrm{no lapse}) \cdot \mathrm{P}(\textrm{positive} | \textrm{no lapse}) \\
  &= \frac{1}{2} \gamma + (1 - \gamma) \cdot F(x)
\end{align}


If we then let $\gamma = 2\lambda$ then the probability of a positive response becomes

$$
\mathrm{P}(\textrm{positive}) = \lambda + (1 - 2\lambda) \cdot F(x)
$$

This is the lapse model described in \@ref(eq:Psi)! But now there is a little bit more insight into what the parameter $\lambda$ is. If $\gamma$ is the true lapse rate, then $\lambda$ is half the lapse rate. This may sound strange at first, but remember that equation \@ref(eq:Psi) was motivated as a lower and upper bound to the psychometric function, and where the bounds are constrained by the same amount. Here the motivation is from a process model, yet the two lines of reasoning arrive at the same model. 

<img src="060-results_files/figure-html/ch060-Magenta-Finger-1.png" width="85%" style="display: block; margin: auto;" />

