## Feature Selection / Filtering {#fs}




Often, data sets include a large number of features.
The technique of extracting a subset of relevant features is called "feature selection".

The objective of feature selection is to fit the sparse dependent of a model on a subset of available data features in the most suitable manner.
Feature selection can enhance the interpretability of the model, speed up the learning process and improve the learner performance.
Different approaches exist to identify the relevant features.
Two different approaches are emphasized in the literature:
one is called [Filtering](#fs-filtering) and the other approach is often referred to as feature subset selection or [wrapper methods](#fs-wrapper).

What are the differences [@guyon2003;@chandrashekar2014]?

* **Filtering**:
  An external algorithm computes a rank of the features (e.g. based on the correlation to the response).
  Then, features are subsetted by a certain criteria, e.g. an absolute number or a percentage of the number of variables.
  The selected features will then be used to fit a model (with optional hyperparameters selected by tuning).
  This calculation is usually cheaper than "feature subset selection" in terms of computation time.
  All filters are connected via package [mlr3filters](https://mlr3filters.mlr-org.com).
* **Wrapper Methods**:
  Here, no ranking of features is done.
  Instead, an optimization algorithm selects a subset of the features, evaluates the set by calculating the resampled predictive performance, and then
  proposes a new set of features (or terminates).
  A simple example is the sequential forward selection.
  This method is usually computationally very intensive as a lot of models are fitted.
  Also, strictly speaking, all these models would need to be tuned before the performance is estimated.
  This would require an additional nested level in a CV setting.
  After undertaken all of these steps, the final set of selected features is again fitted (with optional hyperparameters selected by tuning).
  Wrapper methods are implemented in the [mlr3fselect](https://mlr3fselect.mlr-org.com) package.
* **Embedded Methods**:
  Many learners internally select a subset of the features which they find helpful for prediction.
  These subsets can usually be queried, as the following example demonstrates:
  
  ```r
  task = tsk("iris")
  learner = lrn("classif.rpart")
  
  # ensure that the learner selects features
  stopifnot("selected_features" %in% learner$properties)
  
  # fit a simple classification tree
  learner = learner$train(task)
  
  # extract all features used in the classification tree:
  learner$selected_features()
  ```
  
  ```
  ## [1] "Petal.Length" "Petal.Width"
  ```

There are also [Ensemble filters](#fs-ensemble) built upon the idea of stacking single filter methods. These are not yet implemented.


### Filters {#fs-filter}

Filter methods assign an importance value to each feature.
Based on these values the features can be ranked.
Thereafter, we are able to select a feature subset.
There is a list of all implemented filter methods in the [Appendix](#list-filters).

### Calculating filter values {#fs-calc}

Currently, only classification and regression tasks are supported.

The first step it to create a new R object using the class of the desired filter method.
Each object of class `Filter` has a `.$calculate()` method which computes the filter values and ranks them in a descending order.


```r
library("mlr3filters")
filter = FilterJMIM$new()

task = tsk("iris")
filter$calculate(task)

as.data.table(filter)
```

```
##         feature  score
## 1:  Petal.Width 1.0000
## 2: Sepal.Length 0.6667
## 3: Petal.Length 0.3333
## 4:  Sepal.Width 0.0000
```

Some filters support changing specific hyperparameters.
This is similar to setting hyperparameters of a [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) using `.$param_set$values`:


```r
filter_cor = FilterCorrelation$new()
filter_cor$param_set
```

```
## <ParamSet>
##        id    class lower upper
## 1:    use ParamFct    NA    NA
## 2: method ParamFct    NA    NA
##                                                                  levels
## 1: everything,all.obs,complete.obs,na.or.complete,pairwise.complete.obs
## 2:                                             pearson,kendall,spearman
##       default value
## 1: everything      
## 2:    pearson
```

```r
# change parameter 'method'
filter_cor$param_set$values = list(method = "spearman")
filter_cor$param_set
```

```
## <ParamSet>
##        id    class lower upper
## 1:    use ParamFct    NA    NA
## 2: method ParamFct    NA    NA
##                                                                  levels
## 1: everything,all.obs,complete.obs,na.or.complete,pairwise.complete.obs
## 2:                                             pearson,kendall,spearman
##       default    value
## 1: everything         
## 2:    pearson spearman
```

Rather than taking the "long" R6 way to create a filter, there is also a built-in shorthand notation for filter creation:


```r
filter = flt("cmim")
filter
```

```
## <FilterCMIM:cmim>
## Task Types: classif, regr
## Task Properties: -
## Packages: praznik
## Feature types: integer, numeric, factor, ordered
```

### Variable Importance Filters {#fs-var-imp-filters}

All [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) with the property "importance" come with integrated feature selection methods.

You can find a list of all learners with this property in the [Appendix](#fs-filter-embedded-list).

For some learners the desired filter method needs to be set during learner creation.
For example, learner `classif.ranger` (in the package [mlr3learners](https://mlr3learners.mlr-org.com)) comes with multiple integrated methods.
See the help page of [`ranger::ranger`](https://www.rdocumentation.org/packages/ranger/topics/ranger).
To use method "impurity", you need to set the filter method during construction.


```r
library("mlr3learners")
lrn = lrn("classif.ranger", importance = "impurity")
```

Now you can use the [`mlr3filters::FilterImportance`](https://mlr3filters.mlr-org.com/reference/FilterImportance.html) class for algorithm-embedded methods to filter a [`Task`](https://mlr3.mlr-org.com/reference/Task.html).


```r
library("mlr3learners")

task = tsk("iris")
filter = flt("importance", learner = lrn)
filter$calculate(task)
head(as.data.table(filter), 3)
```

```
##         feature  score
## 1:  Petal.Width 45.020
## 2: Petal.Length 43.615
## 3: Sepal.Length  8.679
```

### Ensemble Methods {#fs-ensemble}

Work in progress.

### Wrapper Methods {#fs-wrapper}

Wrapper feature selection is supported via the [mlr3fselect](https://mlr3fselect.mlr-org.com) extension package.
At the heart of [mlr3fselect](https://mlr3fselect.mlr-org.com) are the R6 classes:

* [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html), [`FSelectInstanceMultiCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceMultiCrit.html): These two classes describe the feature selection problem and store the results.
* [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html): This class is the base class for implementations of feature selection algorithms.

### The `FSelectInstance` Classes {#fs-wrapper-optimization}

The following sub-section examines the feature selection on the [`Pima`](https://mlr3.mlr-org.com/reference/mlr_tasks_sonar.html) data set which is used to predict whether or not a patient has diabetes.


```r
task = tsk("pima")
print(task)
```

```
## <TaskClassif:pima> (768 x 9)
## * Target: diabetes
## * Properties: twoclass
## * Features (8):
##   - dbl (8): age, glucose, insulin, mass, pedigree, pregnant, pressure,
##     triceps
```
We use the classification tree from [rpart](https://cran.r-project.org/package=rpart).


```r
learner = lrn("classif.rpart")
```

Next, we need to specify how to evaluate the performance of the feature subsets.
For this, we need to choose a [`resampling strategy`](https://mlr3.mlr-org.com/reference/Resampling.html) and a [`performance measure`](https://mlr3.mlr-org.com/reference/Measure.html).


```r
hout = rsmp("holdout")
measure = msr("classif.ce")
```

Finally, one has to choose the available budget for the feature selection.
This is done by selecting one of the available [`Terminators`](https://bbotk.mlr-org.com/reference/Terminator.html):

* Terminate after a given time ([`TerminatorClockTime`](https://bbotk.mlr-org.com/reference/mlr_terminators_clock_time.html))
* Terminate after a given amount of iterations ([`TerminatorEvals`](https://bbotk.mlr-org.com/reference/mlr_terminators_evals.html))
* Terminate after a specific performance is reached ([`TerminatorPerfReached`](https://bbotk.mlr-org.com/reference/mlr_terminators_perf_reached.html))
* Terminate when feature selection does not improve ([`TerminatorStagnation`](https://bbotk.mlr-org.com/reference/mlr_terminators_stagnation.html))
* A combination of the above in an *ALL* or *ANY* fashion ([`TerminatorCombo`](https://bbotk.mlr-org.com/reference/mlr_terminators_combo.html))

For this short introduction, we specify a budget of 20 evaluations and then put everything together into a [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html):


```r
library("mlr3fselect")

evals20 = trm("evals", n_evals = 20)

instance = FSelectInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measure = measure,
  terminator = evals20
)
instance
```

```
## <FSelectInstanceSingleCrit>
## * State:  Not optimized
## * Objective: <ObjectiveFSelect:classif.rpart_on_pima>
## * Search Space:
## <ParamSet>
##          id    class lower upper      levels        default value
## 1:      age ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 2:  glucose ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 3:  insulin ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 4:     mass ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 5: pedigree ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 6: pregnant ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 7: pressure ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## 8:  triceps ParamLgl    NA    NA  TRUE,FALSE <NoDefault[3]>      
## * Terminator: <TerminatorEvals>
## * Terminated: FALSE
## * Archive:
## <ArchiveFSelect>
## Null data.table (0 rows and 0 cols)
```

To start the feature selection, we still need to select an algorithm which are defined via the [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html) class

### The `FSelector` Class

The following algorithms are currently implemented in [mlr3fselect](https://mlr3fselect.mlr-org.com):

* Random Search ([`FSelectorRandomSearch`](https://mlr3fselect.mlr-org.com/reference/FSelectorRandomSearch.html))
* Exhaustive Search ([`FSelectorExhaustiveSearch`](https://mlr3fselect.mlr-org.com/reference/FSelectorExhaustiveSearch.html))
* Sequential Search ([`FSelectorSequential`](https://mlr3fselect.mlr-org.com/reference/FSelectorSequential.html))
* Recursive Feature Elimination ([`FSelectorRFE`](https://mlr3fselect.mlr-org.com/reference/FSelectorRFE.html))
* Design Points ([`FSelectorDesignPoints`](https://mlr3fselect.mlr-org.com/reference/FSelectorDesignPoints.html))

In this example, we will use a simple random search.


```r
fselector = fs("random_search")
```

### Triggering the Tuning {#wrapper-selection-triggering}

To start the feature selection, we simply pass the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html) to the `$optimize()` method of the initialized [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html). The algorithm proceeds as follows

1. The [`FSelector`](https://mlr3fselect.mlr-org.com/reference/FSelector.html) proposes at least one feature subset and may propose multiple subsets to improve parallelization, which can be controlled via the setting `batch_size`).
2. For each feature subset, the given [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) is fitted on the [`Task`](https://mlr3.mlr-org.com/reference/Task.html) using the provided [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html).
   All evaluations are stored in the archive of the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html).
3. The [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) is queried if the budget is exhausted.
   If the budget is not exhausted, restart with 1) until it is.
4. Determine the feature subset with the best observed performance.
5. Store the best feature subset as the result in the instance object.
The best feature subset (`$result_feature_set`) and the corresponding measured performance (`$result_y`) can be accessed from the instance.


```r
fselector$optimize(instance)
```

```
## INFO  [10:45:09.137] Starting to optimize 8 parameter(s) with '<FSelectorRandomSearch>' and '<TerminatorEvals>' 
## INFO  [10:45:09.336] Evaluating 10 configuration(s) 
## INFO  [10:45:11.473] Result of batch 1: 
## INFO  [10:45:11.476]    age glucose insulin  mass pedigree pregnant pressure triceps classif.ce 
## INFO  [10:45:11.476]   TRUE    TRUE    TRUE  TRUE     TRUE    FALSE     TRUE    TRUE     0.2227 
## INFO  [10:45:11.476]   TRUE    TRUE    TRUE  TRUE     TRUE    FALSE     TRUE    TRUE     0.2227 
## INFO  [10:45:11.476]   TRUE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.2461 
## INFO  [10:45:11.476]  FALSE   FALSE   FALSE  TRUE    FALSE    FALSE    FALSE   FALSE     0.3086 
## INFO  [10:45:11.476]  FALSE    TRUE   FALSE  TRUE     TRUE    FALSE    FALSE   FALSE     0.2266 
## INFO  [10:45:11.476]   TRUE    TRUE    TRUE  TRUE     TRUE     TRUE    FALSE    TRUE     0.2188 
## INFO  [10:45:11.476]   TRUE   FALSE    TRUE  TRUE    FALSE    FALSE    FALSE    TRUE     0.2812 
## INFO  [10:45:11.476]  FALSE   FALSE    TRUE FALSE     TRUE     TRUE     TRUE   FALSE     0.3555 
## INFO  [10:45:11.476]  FALSE   FALSE   FALSE  TRUE     TRUE    FALSE     TRUE    TRUE     0.3477 
## INFO  [10:45:11.476]   TRUE   FALSE   FALSE  TRUE     TRUE     TRUE     TRUE    TRUE     0.3125 
## INFO  [10:45:11.476]                                 uhash 
## INFO  [10:45:11.476]  1fe18e5f-7a34-4148-b6c2-a42c58b928ff 
## INFO  [10:45:11.476]  d097d57a-97cc-4ddf-92ee-eaec8fd6405c 
## INFO  [10:45:11.476]  814df63d-c0de-496e-b955-a97f681e2fd8 
## INFO  [10:45:11.476]  ea42c156-d045-4bfb-bd46-6dc33af9c437 
## INFO  [10:45:11.476]  28f1dca0-c3bc-4f70-bf19-4e31b332245b 
## INFO  [10:45:11.476]  bfe49505-6036-4423-9b47-f283005449da 
## INFO  [10:45:11.476]  b8db71b4-70b4-43ed-82d0-cd389a60b064 
## INFO  [10:45:11.476]  ddb471a7-d2a3-4531-be99-5471cf745938 
## INFO  [10:45:11.476]  66d73848-727d-4464-8217-0b2433f22d84 
## INFO  [10:45:11.476]  6af55d70-19a4-4e43-bdb7-5c820a0c9934 
## INFO  [10:45:11.480] Evaluating 10 configuration(s) 
## INFO  [10:45:13.321] Result of batch 2: 
## INFO  [10:45:13.325]    age glucose insulin  mass pedigree pregnant pressure triceps classif.ce 
## INFO  [10:45:13.325]   TRUE   FALSE    TRUE FALSE     TRUE     TRUE    FALSE    TRUE     0.3359 
## INFO  [10:45:13.325]  FALSE   FALSE   FALSE FALSE     TRUE    FALSE    FALSE   FALSE     0.3555 
## INFO  [10:45:13.325]  FALSE    TRUE    TRUE  TRUE    FALSE     TRUE    FALSE    TRUE     0.2266 
## INFO  [10:45:13.325]   TRUE    TRUE    TRUE FALSE    FALSE    FALSE    FALSE   FALSE     0.2461 
## INFO  [10:45:13.325]  FALSE    TRUE    TRUE  TRUE     TRUE    FALSE    FALSE    TRUE     0.2500 
## INFO  [10:45:13.325]   TRUE   FALSE   FALSE  TRUE    FALSE     TRUE     TRUE    TRUE     0.3086 
## INFO  [10:45:13.325]  FALSE   FALSE    TRUE FALSE     TRUE    FALSE     TRUE    TRUE     0.3594 
## INFO  [10:45:13.325]   TRUE    TRUE   FALSE FALSE     TRUE    FALSE    FALSE    TRUE     0.2578 
## INFO  [10:45:13.325]  FALSE   FALSE    TRUE FALSE     TRUE     TRUE    FALSE   FALSE     0.3320 
## INFO  [10:45:13.325]   TRUE    TRUE    TRUE  TRUE     TRUE    FALSE     TRUE    TRUE     0.2227 
## INFO  [10:45:13.325]                                 uhash 
## INFO  [10:45:13.325]  6f77e9cc-86d7-4440-b520-653993a691e1 
## INFO  [10:45:13.325]  3b505cfd-618a-4626-81f7-c37a4cd787e9 
## INFO  [10:45:13.325]  8d7fb624-9afe-4c6a-b262-54cd474279c6 
## INFO  [10:45:13.325]  3a557463-aed6-432a-a994-1c9109246de2 
## INFO  [10:45:13.325]  636797da-04bb-4778-8977-c759b77d5523 
## INFO  [10:45:13.325]  4814e1bd-7d35-4324-a2df-d0a727c31be2 
## INFO  [10:45:13.325]  7e9cead2-3016-4b4d-85e6-293a642fc83f 
## INFO  [10:45:13.325]  683a6817-2aff-40e5-91ce-9bfb42303c31 
## INFO  [10:45:13.325]  726f50ab-3461-4b33-ba08-c3d4e83680d5 
## INFO  [10:45:13.325]  1c37e1cf-2b64-43e1-a114-a5fd948c177f 
## INFO  [10:45:13.332] Finished optimizing after 20 evaluation(s) 
## INFO  [10:45:13.333] Result: 
## INFO  [10:45:13.336]   age glucose insulin mass pedigree pregnant pressure triceps 
## INFO  [10:45:13.336]  TRUE    TRUE    TRUE TRUE     TRUE     TRUE    FALSE    TRUE 
## INFO  [10:45:13.336]                                        features  x_domain classif.ce 
## INFO  [10:45:13.336]  age,glucose,insulin,mass,pedigree,pregnant,... <list[8]>     0.2188
```

```
##     age glucose insulin mass pedigree pregnant pressure triceps
## 1: TRUE    TRUE    TRUE TRUE     TRUE     TRUE    FALSE    TRUE
##                                          features  x_domain classif.ce
## 1: age,glucose,insulin,mass,pedigree,pregnant,... <list[8]>     0.2188
```

```r
instance$result_feature_set
```

```
## [1] "age"      "glucose"  "insulin"  "mass"     "pedigree" "pregnant" "triceps"
```

```r
instance$result_y
```

```
## classif.ce 
##     0.2188
```
One can investigate all resamplings which were undertaken, as they are stored in the archive of the [`FSelectInstanceSingleCrit`](https://mlr3fselect.mlr-org.com/reference/FSelectInstanceSingleCrit.html) and can be accessed through `$data()` method:


```r
instance$archive$data()
```

```
##       age glucose insulin  mass pedigree pregnant pressure triceps classif.ce
##  1:  TRUE    TRUE    TRUE  TRUE     TRUE    FALSE     TRUE    TRUE     0.2227
##  2:  TRUE    TRUE    TRUE  TRUE     TRUE    FALSE     TRUE    TRUE     0.2227
##  3:  TRUE    TRUE    TRUE FALSE     TRUE     TRUE     TRUE    TRUE     0.2461
##  4: FALSE   FALSE   FALSE  TRUE    FALSE    FALSE    FALSE   FALSE     0.3086
##  5: FALSE    TRUE   FALSE  TRUE     TRUE    FALSE    FALSE   FALSE     0.2266
##  6:  TRUE    TRUE    TRUE  TRUE     TRUE     TRUE    FALSE    TRUE     0.2188
##  7:  TRUE   FALSE    TRUE  TRUE    FALSE    FALSE    FALSE    TRUE     0.2812
##  8: FALSE   FALSE    TRUE FALSE     TRUE     TRUE     TRUE   FALSE     0.3555
##  9: FALSE   FALSE   FALSE  TRUE     TRUE    FALSE     TRUE    TRUE     0.3477
## 10:  TRUE   FALSE   FALSE  TRUE     TRUE     TRUE     TRUE    TRUE     0.3125
## 11:  TRUE   FALSE    TRUE FALSE     TRUE     TRUE    FALSE    TRUE     0.3359
## 12: FALSE   FALSE   FALSE FALSE     TRUE    FALSE    FALSE   FALSE     0.3555
## 13: FALSE    TRUE    TRUE  TRUE    FALSE     TRUE    FALSE    TRUE     0.2266
## 14:  TRUE    TRUE    TRUE FALSE    FALSE    FALSE    FALSE   FALSE     0.2461
## 15: FALSE    TRUE    TRUE  TRUE     TRUE    FALSE    FALSE    TRUE     0.2500
## 16:  TRUE   FALSE   FALSE  TRUE    FALSE     TRUE     TRUE    TRUE     0.3086
## 17: FALSE   FALSE    TRUE FALSE     TRUE    FALSE     TRUE    TRUE     0.3594
## 18:  TRUE    TRUE   FALSE FALSE     TRUE    FALSE    FALSE    TRUE     0.2578
## 19: FALSE   FALSE    TRUE FALSE     TRUE     TRUE    FALSE   FALSE     0.3320
## 20:  TRUE    TRUE    TRUE  TRUE     TRUE    FALSE     TRUE    TRUE     0.2227
##                                    uhash  x_domain           timestamp batch_nr
##  1: 1fe18e5f-7a34-4148-b6c2-a42c58b928ff <list[8]> 2020-12-01 10:45:11        1
##  2: d097d57a-97cc-4ddf-92ee-eaec8fd6405c <list[8]> 2020-12-01 10:45:11        1
##  3: 814df63d-c0de-496e-b955-a97f681e2fd8 <list[8]> 2020-12-01 10:45:11        1
##  4: ea42c156-d045-4bfb-bd46-6dc33af9c437 <list[8]> 2020-12-01 10:45:11        1
##  5: 28f1dca0-c3bc-4f70-bf19-4e31b332245b <list[8]> 2020-12-01 10:45:11        1
##  6: bfe49505-6036-4423-9b47-f283005449da <list[8]> 2020-12-01 10:45:11        1
##  7: b8db71b4-70b4-43ed-82d0-cd389a60b064 <list[8]> 2020-12-01 10:45:11        1
##  8: ddb471a7-d2a3-4531-be99-5471cf745938 <list[8]> 2020-12-01 10:45:11        1
##  9: 66d73848-727d-4464-8217-0b2433f22d84 <list[8]> 2020-12-01 10:45:11        1
## 10: 6af55d70-19a4-4e43-bdb7-5c820a0c9934 <list[8]> 2020-12-01 10:45:11        1
## 11: 6f77e9cc-86d7-4440-b520-653993a691e1 <list[8]> 2020-12-01 10:45:13        2
## 12: 3b505cfd-618a-4626-81f7-c37a4cd787e9 <list[8]> 2020-12-01 10:45:13        2
## 13: 8d7fb624-9afe-4c6a-b262-54cd474279c6 <list[8]> 2020-12-01 10:45:13        2
## 14: 3a557463-aed6-432a-a994-1c9109246de2 <list[8]> 2020-12-01 10:45:13        2
## 15: 636797da-04bb-4778-8977-c759b77d5523 <list[8]> 2020-12-01 10:45:13        2
## 16: 4814e1bd-7d35-4324-a2df-d0a727c31be2 <list[8]> 2020-12-01 10:45:13        2
## 17: 7e9cead2-3016-4b4d-85e6-293a642fc83f <list[8]> 2020-12-01 10:45:13        2
## 18: 683a6817-2aff-40e5-91ce-9bfb42303c31 <list[8]> 2020-12-01 10:45:13        2
## 19: 726f50ab-3461-4b33-ba08-c3d4e83680d5 <list[8]> 2020-12-01 10:45:13        2
## 20: 1c37e1cf-2b64-43e1-a114-a5fd948c177f <list[8]> 2020-12-01 10:45:13        2
```

The associated resampling iterations can be accessed in the [`BenchmarkResult`](https://mlr3.mlr-org.com/reference/BenchmarkResult.html):


```r
instance$archive$benchmark_result$data
```

```
## <ResultData>
##   Public:
##     as_data_table: function (view = NULL, reassemble_learners = TRUE, convert_predictions = TRUE, 
##     clone: function (deep = FALSE) 
##     combine: function (rdata) 
##     data: list
##     initialize: function (data = NULL) 
##     iterations: function (view = NULL) 
##     learners: function (view = NULL, states = TRUE, reassemble = TRUE) 
##     logs: function (view = NULL, condition) 
##     prediction: function (view = NULL, predict_sets = "test") 
##     predictions: function (view = NULL, predict_sets = "test") 
##     resamplings: function (view = NULL) 
##     sweep: function () 
##     task_type: active binding
##     tasks: function (view = NULL, reassemble = TRUE) 
##     uhashes: function (view = NULL) 
##   Private:
##     deep_clone: function (name, value) 
##     get_view_index: function (view)
```

The `uhash` column links the resampling iterations to the evaluated feature subsets stored in `instance$archive$data()`. This allows e.g. to score the included [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html)s on a different measure.

Now the optimized feature subset can be used to subset the task and fit the model on all observations.


```r
task$select(instance$result_feature_set)
learner$train(task)
```

The trained model can now be used to make a prediction on external data.
Note that predicting on observations present in the `task`,  should be avoided.
The model has seen these observations already during feature selection and therefore results would be statistically biased.
Hence, the resulting performance measure would be over-optimistic.
Instead, to get statistically unbiased performance estimates for the current task, [nested resampling](#nested-resampling) is required.

### Automating the Feature Selection {#autofselect}

The [`AutoFSelector`](https://mlr3fselect.mlr-org.com/reference/AutoFSelector.html) wraps a learner and augments it with an automatic feature selection for a given task.
Because the [`AutoFSelector`](https://mlr3fselect.mlr-org.com/reference/AutoFSelector.html) itself inherits from the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) base class, it can be used like any other learner.
Analogously to the previous subsection, a new classification tree learner is created.
This classification tree learner automatically starts a feature selection on the given task using an inner resampling (holdout).
We create a terminator which allows 10 evaluations, and and uses a simple random search as feature selection algorithm:


```r
library("paradox")
library("mlr3fselect")

learner = lrn("classif.rpart")
terminator = trm("evals", n_evals = 10)
fselector = fs("random_search")

at = AutoFSelector$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  terminator = terminator,
  fselector = fselector
)
at
```

```
## <AutoFSelector:classif.rpart.fselector>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

We can now use the learner like any other learner, calling the `$train()` and `$predict()` method.
This time however, we pass it to [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) to compare the optimized feature subset to the complete feature set.
This way, the [`AutoFSelector`](https://mlr3fselect.mlr-org.com/reference/AutoFSelector.html) will do its resampling for feature selection on the training set of the respective split of the outer resampling.
The learner then undertakes predictions using the test set of the outer resampling.
This yields unbiased performance measures, as the observations in the test set have not been used during feature selection or fitting of the respective learner.
This is called [nested resampling](#nested-resampling).

To compare the optimized feature subset with the complete feature set, we can use [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html):


```r
grid = benchmark_grid(
  task = tsk("pima"),
  learner = list(at, lrn("classif.rpart")),
  resampling = rsmp("cv", folds = 3)
)

# avoid console output from mlrfselect
logger = lgr::get_logger("bbotk")
logger$set_threshold("warn")

bmr = benchmark(grid, store_models = TRUE)
bmr$aggregate(msrs(c("classif.ce", "time_train")))
```

```
##    nr      resample_result task_id              learner_id resampling_id iters
## 1:  1 <ResampleResult[21]>    pima classif.rpart.fselector            cv     3
## 2:  2 <ResampleResult[21]>    pima           classif.rpart            cv     3
##    classif.ce time_train
## 1:     0.2656          0
## 2:     0.2734          0
```

Note that we do not expect any significant differences since we only evaluated a small fraction of the possible feature subsets.
