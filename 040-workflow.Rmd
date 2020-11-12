# Model Development {#application}


Multilevel models should be the default. The alternatives are models with complete pooling, or models with no pooling. Pooling vs. no pooling considers modeling all the data as a whole, or each of the the smallest components individually. The former implies that the variation between groups is zero (all groups are the same), and the latter implies that the variation between groups is infinite (no groups are the same). Multilevel models assume that the truth is somewhere in the middle of zero and infinity.


Hierarchical models are a specific kind of multilevel model where one or more groups are nested within a larger one. In the case of the psychometric data, there are three age groups, and within each age group are individual subjects. Multilevel modeling provides a way to quantify and apportion the variation within the data to each level in the model. For an in-depth introduction to multilevel modeling, see @gelman2006data.


<!--
The question of inferential adequacy depends on the set of questions that we are seeking to answer with the data from the psychometric experiment. The broad objective is to determine if there are any significant differences between age groups when it comes to temporal sensitivity, perceptual synchrony, and temporal recalibration, and if the task influences the results as well. The specific goals are to estimate and compare the PSS an JND across all age groups, conditions, and tasks, and determine the affect of recalibration between age groups.


For the last question, model adequacy, I will be following a set of steps proposed in @betancourt2020. The purpose of laying out these steps is not to again blindly check them off, but to force the analyst to carefully consider each point and make an _informed_ decision whether the step is necessary or to craft the specifics of how the step should be completed. The steps are listed in table \@ref(tab:ch030-workflow-steps). These steps are also not meant to be followed linearly. If at any point it is discovered that there is an issue in conceptual understanding or model adequacy or something else, then it is encouraged to go back to a previous step and start with a new understanding.
-->


## Iteration 1 {#iter1}

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

_Celebrate_

## Iteration 2 {#iter2}

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

_Celebrate_

## Iteration 3 {#iter3}

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

_Celebrate_

## Iteration 4 {#iter4}

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

_Celebrate_

## Iteration 5 {#iter5}

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

_Celebrate_

