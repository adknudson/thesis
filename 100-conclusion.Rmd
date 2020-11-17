# Discussion and Conclusion {#conclusion}


The results from the previous chapter provide insight into how future experiments can be designed to offer better inferences. In the visual TOJ task, the granularity in the SOA values near the PSS could be increased to get more reliable estimates of the slope and to avoid complete separation. Including a lapse rate helps, but can be unreliable if the range of SOA values is too narrow. For more difficult tasks like the sensorimotor TOJ task, larger SOA values are necessary so that the lapse rate can be accurately measured. 


Since the lapse rate for trained observers is estimated to be no more than $0.05$, an experiment would need about $20$ trials at a larger SOA in order to pick up on a single lapse. In this data, we have multiple subjects within an age group, so the repetition is spread out. As a consequence, it is possible to estimate an age group level lapse rate, but more difficult to estimate subject specific lapses. 


For multilevel modeling and partial pooling to have a significant benefit, five or more groups is recommended. The study could be expanded to have five or six age groups (20-30, 30-40, etc.). More age groups would also allow for finer tracking of trends in the distribution of PSS and JND values and the affect of temporal recalibration on them.


In the future we would like to explore a causal inference model for the psychometric function. The results drawn from the statistical model are simply associations between the predictor variables, and of course correlations do not imply causation. How do we move from association to cause-and-effect? Drawing the model as a directed acyclic graph and testing the implications of the model is a start. With a proper model, total effects of a certain variable on the outcome can be determined.


The model development was motivated by domain expertise consistency and the model's ability to answer domain-related research questions. The emphasis is on model comparison which is not necessarily model selection. Certain models are useful for answering different questions. We compared models that have the potential to answer questions pertaining to the age group level and compared their estimated predictive performance. Predictive performance is a reliable metric for model comparison because a model than can predict well likely captures the regular features of the observed data and the data generating model.


We have produced a novel statistical model for temporal order judgment data by following a principled workflow and fitting a series of Bayesian models efficiently using Hamiltonian Monte Carlo in the `R` programming language with `Stan`. We described methods for selecting priors for the slope and intercept parameters, and argued why the selected linear parameterization can have practical benefits on prior specification. Finally we motivated the inclusion of a lapse rate into the model for the psychometric function with an illustrative diagram of the result of a temporal order judgment experiment.
