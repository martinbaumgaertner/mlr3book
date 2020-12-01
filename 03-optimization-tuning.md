## Hyperparameter Tuning {#tuning}


Hyperparameters are second-order parameters of machine learning models that, while often not explicitly optimized during the model estimation process, can have an important impact on the outcome and predictive performance of a model.
Typically, hyperparameters are fixed before training a model.
However, because the output of a model can be sensitive to the specification of hyperparameters, it is often recommended to make an informed decision about which hyperparameter settings may yield better model performance.
In many cases, hyperparameter settings may be chosen _a priori_, but it can be advantageous to try different settings before fitting your model on the training data.
This process is often called model 'tuning'.

Hyperparameter tuning is supported via the [mlr3tuning](https://mlr3tuning.mlr-org.com) extension package.
Below you can find an illustration of the process:

<img src="images/tuning_process.svg" style="display: block; margin: auto;" />

At the heart of [mlr3tuning](https://mlr3tuning.mlr-org.com) are the R6 classes:

* [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html), [`TuningInstanceMultiCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceMultiCrit.html): These two classes describe the tuning problem and store the results.
* [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html): This class is the base class for implementations of tuning algorithms.

### The `TuningInstance*` Classes {#tuning-optimization}

The following sub-section examines the optimization of a simple classification tree on the [`Pima Indian Diabetes`](https://mlr3.mlr-org.com/reference/mlr_tasks_pima.html) data set.


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

We use the classification tree from [rpart](https://cran.r-project.org/package=rpart) and choose a subset of the hyperparameters we want to tune.
This is often referred to as the "tuning space".


```r
learner = lrn("classif.rpart")
learner$param_set
```

```
## <ParamSet>
##                 id    class lower upper      levels        default value
##  1:       minsplit ParamInt     1   Inf                         20      
##  2:      minbucket ParamInt     1   Inf             <NoDefault[3]>      
##  3:             cp ParamDbl     0     1                       0.01      
##  4:     maxcompete ParamInt     0   Inf                          4      
##  5:   maxsurrogate ParamInt     0   Inf                          5      
##  6:       maxdepth ParamInt     1    30                         30      
##  7:   usesurrogate ParamInt     0     2                          2      
##  8: surrogatestyle ParamInt     0     1                          0      
##  9:           xval ParamInt     0   Inf                         10     0
## 10:     keep_model ParamLgl    NA    NA  TRUE,FALSE          FALSE
```

Here, we opt to tune two parameters:

* The complexity `cp`
* The termination criterion `minsplit`

The tuning space needs to be bounded, therefore one has to set lower and upper bounds:


```r
library("paradox")
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))
tune_ps
```

```
## <ParamSet>
##          id    class lower upper levels        default value
## 1:       cp ParamDbl 0.001   0.1        <NoDefault[3]>      
## 2: minsplit ParamInt 1.000  10.0        <NoDefault[3]>
```

Next, we need to specify how to evaluate the performance.
For this, we need to choose a [`resampling strategy`](https://mlr3.mlr-org.com/reference/Resampling.html) and a [`performance measure`](https://mlr3.mlr-org.com/reference/Measure.html).


```r
hout = rsmp("holdout")
measure = msr("classif.ce")
```

Finally, one has to select the budget available, to solve this tuning instance.
This is done by selecting one of the available [`Terminators`](https://bbotk.mlr-org.com/reference/Terminator.html):

* Terminate after a given time ([`TerminatorClockTime`](https://bbotk.mlr-org.com/reference/mlr_terminators_clock_time.html))
* Terminate after a given amount of iterations ([`TerminatorEvals`](https://bbotk.mlr-org.com/reference/mlr_terminators_evals.html))
* Terminate after a specific performance is reached ([`TerminatorPerfReached`](https://bbotk.mlr-org.com/reference/mlr_terminators_perf_reached.html))
* Terminate when tuning does not improve ([`TerminatorStagnation`](https://bbotk.mlr-org.com/reference/mlr_terminators_stagnation.html))
* A combination of the above in an *ALL* or *ANY* fashion ([`TerminatorCombo`](https://bbotk.mlr-org.com/reference/mlr_terminators_combo.html))

For this short introduction, we specify a budget of 20 evaluations and then put everything together into a [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html):


```r
library("mlr3tuning")

evals20 = trm("evals", n_evals = 20)

instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measure = measure,
  search_space = tune_ps,
  terminator = evals20
)
instance
```

```
## <TuningInstanceSingleCrit>
## * State:  Not optimized
## * Objective: <ObjectiveTuning:classif.rpart_on_pima>
## * Search Space:
## <ParamSet>
##          id    class lower upper levels        default value
## 1:       cp ParamDbl 0.001   0.1        <NoDefault[3]>      
## 2: minsplit ParamInt 1.000  10.0        <NoDefault[3]>      
## * Terminator: <TerminatorEvals>
## * Terminated: FALSE
## * Archive:
## <ArchiveTuning>
## Null data.table (0 rows and 0 cols)
```

To start the tuning, we still need to select how the optimization should take place.
In other words, we need to choose the **optimization algorithm** via the [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) class.

### The `Tuner` Class

The following algorithms are currently implemented in [mlr3tuning](https://mlr3tuning.mlr-org.com):

* Grid Search ([`TunerGridSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_grid_search.html))
* Random Search ([`TunerRandomSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_random_search.html)) [@bergstra2012]
* Generalized Simulated Annealing ([`TunerGenSA`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_gensa.html))
* Non-Linear Optimization ([`TunerNLoptr`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_nloptr.html))

In this example, we will use a simple grid search with a grid resolution of 5.


```r
tuner = tnr("grid_search", resolution = 5)
```

Since we have only numeric parameters, [`TunerGridSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_grid_search.html) will create an equidistant grid between the respective upper and lower bounds.
As we have two hyperparameters with a resolution of 5, the two-dimensional grid consists of $5^2 = 25$ configurations.
Each configuration serves as a hyperparameter setting for the previously defined [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) and triggers a 3-fold cross validation on the task.
All configurations will be examined by the tuner (in a random order), until either all configurations are evaluated or the [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) signals that the budget is exhausted.

### Triggering the Tuning {#tuning-triggering}

To start the tuning, we simply pass the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) to the `$optimize()` method of the initialized [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html).
The tuner proceeds as follows:

1. The [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) proposes at least one hyperparameter configuration (the [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) may propose multiple points to improve parallelization, which can be controlled via the setting `batch_size`).
2. For each configuration, the given [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) is fitted on the [`Task`](https://mlr3.mlr-org.com/reference/Task.html) using the provided [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html).
   All evaluations are stored in the archive of the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html).
3. The [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) is queried if the budget is exhausted.
   If the budget is not exhausted, restart with 1) until it is.
4. Determine the configuration with the best observed performance.
5. Store the best configurations as result in the instance object.
   The best hyperparameter settings (`$result_learner_param_vals`) and the corresponding measured performance (`$result_y`) can be accessed from the instance.


```r
tuner$optimize(instance)
```

```
## INFO  [10:44:36.812] Starting to optimize 2 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [10:44:36.845] Evaluating 1 configuration(s) 
## INFO  [10:44:37.178] Result of batch 1: 
## INFO  [10:44:37.181]     cp minsplit classif.ce                                uhash 
## INFO  [10:44:37.181]  0.001        1     0.2734 24e7d138-3089-4446-b530-cd64a1b96a52 
## INFO  [10:44:37.184] Evaluating 1 configuration(s) 
## INFO  [10:44:37.379] Result of batch 2: 
## INFO  [10:44:37.381]       cp minsplit classif.ce                                uhash 
## INFO  [10:44:37.381]  0.07525        3     0.2305 444e102b-7e59-423c-a756-ba5e16e147ce 
## INFO  [10:44:37.384] Evaluating 1 configuration(s) 
## INFO  [10:44:37.484] Result of batch 3: 
## INFO  [10:44:37.486]      cp minsplit classif.ce                                uhash 
## INFO  [10:44:37.486]  0.0505        8     0.2305 5a71d631-1e06-472a-babb-bca56a001549 
## INFO  [10:44:37.489] Evaluating 1 configuration(s) 
## INFO  [10:44:37.586] Result of batch 4: 
## INFO  [10:44:37.588]       cp minsplit classif.ce                                uhash 
## INFO  [10:44:37.588]  0.02575        3     0.2461 b374c371-d4e9-4c20-970a-ce25381533b2 
## INFO  [10:44:37.591] Evaluating 1 configuration(s) 
## INFO  [10:44:37.677] Result of batch 5: 
## INFO  [10:44:37.680]   cp minsplit classif.ce                                uhash 
## INFO  [10:44:37.680]  0.1        5     0.2734 7bba368c-23f2-4088-b267-a3d4b3b15051 
## INFO  [10:44:37.682] Evaluating 1 configuration(s) 
## INFO  [10:44:37.777] Result of batch 6: 
## INFO  [10:44:37.780]      cp minsplit classif.ce                                uhash 
## INFO  [10:44:37.780]  0.0505       10     0.2305 c7beae29-c7f6-48e9-ad7a-970d057a6c4d 
## INFO  [10:44:37.783] Evaluating 1 configuration(s) 
## INFO  [10:44:37.873] Result of batch 7: 
## INFO  [10:44:37.875]       cp minsplit classif.ce                                uhash 
## INFO  [10:44:37.875]  0.07525        1     0.2305 70c74f13-b760-429e-8398-45a5d382cad3 
## INFO  [10:44:37.877] Evaluating 1 configuration(s) 
## INFO  [10:44:37.966] Result of batch 8: 
## INFO  [10:44:37.968]      cp minsplit classif.ce                                uhash 
## INFO  [10:44:37.968]  0.0505        1     0.2305 bd39bfa2-9bdb-4476-8c0f-dfa0a22db996 
## INFO  [10:44:37.971] Evaluating 1 configuration(s) 
## INFO  [10:44:38.056] Result of batch 9: 
## INFO  [10:44:38.059]       cp minsplit classif.ce                                uhash 
## INFO  [10:44:38.059]  0.07525        5     0.2305 b600bcbe-7365-4c10-90b1-676de4e392a9 
## INFO  [10:44:38.061] Evaluating 1 configuration(s) 
## INFO  [10:44:38.152] Result of batch 10: 
## INFO  [10:44:38.154]     cp minsplit classif.ce                                uhash 
## INFO  [10:44:38.154]  0.001       10     0.2891 9fd64ed1-7805-4f97-851b-97390a6f1ede 
## INFO  [10:44:38.156] Evaluating 1 configuration(s) 
## INFO  [10:44:38.245] Result of batch 11: 
## INFO  [10:44:38.248]   cp minsplit classif.ce                                uhash 
## INFO  [10:44:38.248]  0.1        3     0.2734 5cfa5b58-37b3-470f-a632-25e1f1e7c06c 
## INFO  [10:44:38.250] Evaluating 1 configuration(s) 
## INFO  [10:44:38.346] Result of batch 12: 
## INFO  [10:44:38.348]       cp minsplit classif.ce                                uhash 
## INFO  [10:44:38.348]  0.07525       10     0.2305 bc687413-d076-4ea7-b8e3-53ac1eb7ee2d 
## INFO  [10:44:38.351] Evaluating 1 configuration(s) 
## INFO  [10:44:38.444] Result of batch 13: 
## INFO  [10:44:38.446]      cp minsplit classif.ce                                uhash 
## INFO  [10:44:38.446]  0.0505        5     0.2305 a098185a-2ffd-4cd6-9b01-7f8d47fd3bed 
## INFO  [10:44:38.449] Evaluating 1 configuration(s) 
## INFO  [10:44:38.542] Result of batch 14: 
## INFO  [10:44:38.544]     cp minsplit classif.ce                                uhash 
## INFO  [10:44:38.544]  0.001        3     0.2695 7077bfc4-8331-4cac-b664-dcbd73b67bf6 
## INFO  [10:44:38.546] Evaluating 1 configuration(s) 
## INFO  [10:44:38.635] Result of batch 15: 
## INFO  [10:44:38.638]   cp minsplit classif.ce                                uhash 
## INFO  [10:44:38.638]  0.1       10     0.2734 cfd9cd8b-f2eb-40af-bb7a-4bfcfddb7632 
## INFO  [10:44:38.640] Evaluating 1 configuration(s) 
## INFO  [10:44:38.730] Result of batch 16: 
## INFO  [10:44:38.732]     cp minsplit classif.ce                                uhash 
## INFO  [10:44:38.732]  0.001        5     0.2578 a52f9369-4f15-4bef-a59e-81c1cfbf01da 
## INFO  [10:44:38.735] Evaluating 1 configuration(s) 
## INFO  [10:44:38.815] Result of batch 17: 
## INFO  [10:44:38.817]      cp minsplit classif.ce                                uhash 
## INFO  [10:44:38.817]  0.0505        3     0.2305 52932a46-905e-4b59-bf48-5cb77eab2b39 
## INFO  [10:44:38.820] Evaluating 1 configuration(s) 
## INFO  [10:44:38.911] Result of batch 18: 
## INFO  [10:44:38.914]   cp minsplit classif.ce                                uhash 
## INFO  [10:44:38.914]  0.1        8     0.2734 f4ddce7f-97e0-4346-8e65-05dbbaf4cffa 
## INFO  [10:44:38.916] Evaluating 1 configuration(s) 
## INFO  [10:44:39.004] Result of batch 19: 
## INFO  [10:44:39.006]       cp minsplit classif.ce                                uhash 
## INFO  [10:44:39.006]  0.02575        1     0.2461 b0394a4a-0054-4bba-9073-49b852425eb9 
## INFO  [10:44:39.009] Evaluating 1 configuration(s) 
## INFO  [10:44:39.099] Result of batch 20: 
## INFO  [10:44:39.101]     cp minsplit classif.ce                                uhash 
## INFO  [10:44:39.101]  0.001        8     0.2773 6f76d32f-2b17-48c3-9b9c-5ecfbcb5440a 
## INFO  [10:44:39.109] Finished optimizing after 20 evaluation(s) 
## INFO  [10:44:39.110] Result: 
## INFO  [10:44:39.112]       cp minsplit learner_param_vals  x_domain classif.ce 
## INFO  [10:44:39.112]  0.07525        3          <list[3]> <list[2]>     0.2305
```

```
##         cp minsplit learner_param_vals  x_domain classif.ce
## 1: 0.07525        3          <list[3]> <list[2]>     0.2305
```

```r
instance$result_learner_param_vals
```

```
## $xval
## [1] 0
## 
## $cp
## [1] 0.07525
## 
## $minsplit
## [1] 3
```

```r
instance$result_y
```

```
## classif.ce 
##     0.2305
```

One can investigate all resamplings which were undertaken, as they are stored in the archive of the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) and can be accessed through `$data()` method:


```r
instance$archive$data()
```

```
##          cp minsplit classif.ce                                uhash  x_domain
##  1: 0.00100        1     0.2734 24e7d138-3089-4446-b530-cd64a1b96a52 <list[2]>
##  2: 0.07525        3     0.2305 444e102b-7e59-423c-a756-ba5e16e147ce <list[2]>
##  3: 0.05050        8     0.2305 5a71d631-1e06-472a-babb-bca56a001549 <list[2]>
##  4: 0.02575        3     0.2461 b374c371-d4e9-4c20-970a-ce25381533b2 <list[2]>
##  5: 0.10000        5     0.2734 7bba368c-23f2-4088-b267-a3d4b3b15051 <list[2]>
##  6: 0.05050       10     0.2305 c7beae29-c7f6-48e9-ad7a-970d057a6c4d <list[2]>
##  7: 0.07525        1     0.2305 70c74f13-b760-429e-8398-45a5d382cad3 <list[2]>
##  8: 0.05050        1     0.2305 bd39bfa2-9bdb-4476-8c0f-dfa0a22db996 <list[2]>
##  9: 0.07525        5     0.2305 b600bcbe-7365-4c10-90b1-676de4e392a9 <list[2]>
## 10: 0.00100       10     0.2891 9fd64ed1-7805-4f97-851b-97390a6f1ede <list[2]>
## 11: 0.10000        3     0.2734 5cfa5b58-37b3-470f-a632-25e1f1e7c06c <list[2]>
## 12: 0.07525       10     0.2305 bc687413-d076-4ea7-b8e3-53ac1eb7ee2d <list[2]>
## 13: 0.05050        5     0.2305 a098185a-2ffd-4cd6-9b01-7f8d47fd3bed <list[2]>
## 14: 0.00100        3     0.2695 7077bfc4-8331-4cac-b664-dcbd73b67bf6 <list[2]>
## 15: 0.10000       10     0.2734 cfd9cd8b-f2eb-40af-bb7a-4bfcfddb7632 <list[2]>
## 16: 0.00100        5     0.2578 a52f9369-4f15-4bef-a59e-81c1cfbf01da <list[2]>
## 17: 0.05050        3     0.2305 52932a46-905e-4b59-bf48-5cb77eab2b39 <list[2]>
## 18: 0.10000        8     0.2734 f4ddce7f-97e0-4346-8e65-05dbbaf4cffa <list[2]>
## 19: 0.02575        1     0.2461 b0394a4a-0054-4bba-9073-49b852425eb9 <list[2]>
## 20: 0.00100        8     0.2773 6f76d32f-2b17-48c3-9b9c-5ecfbcb5440a <list[2]>
##               timestamp batch_nr
##  1: 2020-12-01 10:44:37        1
##  2: 2020-12-01 10:44:37        2
##  3: 2020-12-01 10:44:37        3
##  4: 2020-12-01 10:44:37        4
##  5: 2020-12-01 10:44:37        5
##  6: 2020-12-01 10:44:37        6
##  7: 2020-12-01 10:44:37        7
##  8: 2020-12-01 10:44:37        8
##  9: 2020-12-01 10:44:38        9
## 10: 2020-12-01 10:44:38       10
## 11: 2020-12-01 10:44:38       11
## 12: 2020-12-01 10:44:38       12
## 13: 2020-12-01 10:44:38       13
## 14: 2020-12-01 10:44:38       14
## 15: 2020-12-01 10:44:38       15
## 16: 2020-12-01 10:44:38       16
## 17: 2020-12-01 10:44:38       17
## 18: 2020-12-01 10:44:38       18
## 19: 2020-12-01 10:44:39       19
## 20: 2020-12-01 10:44:39       20
```

In sum, the grid search evaluated 20/25 different configurations of the grid in a random order before the [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) stopped the tuning.

The associated resampling iterations can be accessed in the [`BenchmarkResult`](https://mlr3.mlr-org.com/reference/BenchmarkResult.html):


```r
instance$archive$benchmark_result
```

```
## <BenchmarkResult> of 20 rows with 20 resampling runs
##  nr task_id    learner_id resampling_id iters warnings errors
##   1    pima classif.rpart       holdout     1        0      0
##   2    pima classif.rpart       holdout     1        0      0
##   3    pima classif.rpart       holdout     1        0      0
##   4    pima classif.rpart       holdout     1        0      0
##   5    pima classif.rpart       holdout     1        0      0
##   6    pima classif.rpart       holdout     1        0      0
##   7    pima classif.rpart       holdout     1        0      0
##   8    pima classif.rpart       holdout     1        0      0
##   9    pima classif.rpart       holdout     1        0      0
##  10    pima classif.rpart       holdout     1        0      0
##  11    pima classif.rpart       holdout     1        0      0
##  12    pima classif.rpart       holdout     1        0      0
##  13    pima classif.rpart       holdout     1        0      0
##  14    pima classif.rpart       holdout     1        0      0
##  15    pima classif.rpart       holdout     1        0      0
##  16    pima classif.rpart       holdout     1        0      0
##  17    pima classif.rpart       holdout     1        0      0
##  18    pima classif.rpart       holdout     1        0      0
##  19    pima classif.rpart       holdout     1        0      0
##  20    pima classif.rpart       holdout     1        0      0
```

The `uhash` column links the resampling iterations to the evaluated configurations stored in `instance$archive$data()`. This allows e.g. to score the included [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html)s on a different measure.


```r
instance$archive$benchmark_result$score(msr("classif.acc"))
```

```
##                                    uhash nr              task task_id
##  1: 24e7d138-3089-4446-b530-cd64a1b96a52  1 <TaskClassif[45]>    pima
##  2: 444e102b-7e59-423c-a756-ba5e16e147ce  2 <TaskClassif[45]>    pima
##  3: 5a71d631-1e06-472a-babb-bca56a001549  3 <TaskClassif[45]>    pima
##  4: b374c371-d4e9-4c20-970a-ce25381533b2  4 <TaskClassif[45]>    pima
##  5: 7bba368c-23f2-4088-b267-a3d4b3b15051  5 <TaskClassif[45]>    pima
##  6: c7beae29-c7f6-48e9-ad7a-970d057a6c4d  6 <TaskClassif[45]>    pima
##  7: 70c74f13-b760-429e-8398-45a5d382cad3  7 <TaskClassif[45]>    pima
##  8: bd39bfa2-9bdb-4476-8c0f-dfa0a22db996  8 <TaskClassif[45]>    pima
##  9: b600bcbe-7365-4c10-90b1-676de4e392a9  9 <TaskClassif[45]>    pima
## 10: 9fd64ed1-7805-4f97-851b-97390a6f1ede 10 <TaskClassif[45]>    pima
## 11: 5cfa5b58-37b3-470f-a632-25e1f1e7c06c 11 <TaskClassif[45]>    pima
## 12: bc687413-d076-4ea7-b8e3-53ac1eb7ee2d 12 <TaskClassif[45]>    pima
## 13: a098185a-2ffd-4cd6-9b01-7f8d47fd3bed 13 <TaskClassif[45]>    pima
## 14: 7077bfc4-8331-4cac-b664-dcbd73b67bf6 14 <TaskClassif[45]>    pima
## 15: cfd9cd8b-f2eb-40af-bb7a-4bfcfddb7632 15 <TaskClassif[45]>    pima
## 16: a52f9369-4f15-4bef-a59e-81c1cfbf01da 16 <TaskClassif[45]>    pima
## 17: 52932a46-905e-4b59-bf48-5cb77eab2b39 17 <TaskClassif[45]>    pima
## 18: f4ddce7f-97e0-4346-8e65-05dbbaf4cffa 18 <TaskClassif[45]>    pima
## 19: b0394a4a-0054-4bba-9073-49b852425eb9 19 <TaskClassif[45]>    pima
## 20: 6f76d32f-2b17-48c3-9b9c-5ecfbcb5440a 20 <TaskClassif[45]>    pima
##                       learner    learner_id              resampling
##  1: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
##  2: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
##  3: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
##  4: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
##  5: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
##  6: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
##  7: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
##  8: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
##  9: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 10: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 11: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 12: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 13: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 14: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 15: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 16: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 17: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 18: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 19: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
## 20: <LearnerClassifRpart[33]> classif.rpart <ResamplingHoldout[19]>
##     resampling_id iteration              prediction classif.acc
##  1:       holdout         1 <PredictionClassif[19]>      0.7266
##  2:       holdout         1 <PredictionClassif[19]>      0.7695
##  3:       holdout         1 <PredictionClassif[19]>      0.7695
##  4:       holdout         1 <PredictionClassif[19]>      0.7539
##  5:       holdout         1 <PredictionClassif[19]>      0.7266
##  6:       holdout         1 <PredictionClassif[19]>      0.7695
##  7:       holdout         1 <PredictionClassif[19]>      0.7695
##  8:       holdout         1 <PredictionClassif[19]>      0.7695
##  9:       holdout         1 <PredictionClassif[19]>      0.7695
## 10:       holdout         1 <PredictionClassif[19]>      0.7109
## 11:       holdout         1 <PredictionClassif[19]>      0.7266
## 12:       holdout         1 <PredictionClassif[19]>      0.7695
## 13:       holdout         1 <PredictionClassif[19]>      0.7695
## 14:       holdout         1 <PredictionClassif[19]>      0.7305
## 15:       holdout         1 <PredictionClassif[19]>      0.7266
## 16:       holdout         1 <PredictionClassif[19]>      0.7422
## 17:       holdout         1 <PredictionClassif[19]>      0.7695
## 18:       holdout         1 <PredictionClassif[19]>      0.7266
## 19:       holdout         1 <PredictionClassif[19]>      0.7539
## 20:       holdout         1 <PredictionClassif[19]>      0.7227
```

Now the optimized hyperparameters can take the previously created [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html), set the returned hyperparameters and train it on the full dataset.


```r
learner$param_set$values = instance$result_learner_param_vals
learner$train(task)
```

The trained model can now be used to make a prediction on external data.
Note that predicting on observations present in the `task`,  should be avoided.
The model has seen these observations already during tuning and therefore results would be statistically biased.
Hence, the resulting performance measure would be over-optimistic.
Instead, to get statistically unbiased performance estimates for the current task, [nested resampling](#nested-resamling) is required.

### Automating the Tuning {#autotuner}

The [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) wraps a learner and augments it with an automatic tuning for a given set of hyperparameters.
Because the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) itself inherits from the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) base class, it can be used like any other learner.
Analogously to the previous subsection, a new classification tree learner is created.
This classification tree learner automatically tunes the parameters `cp` and `minsplit` using an inner resampling (holdout).
We create a terminator which allows 10 evaluations, and use a simple random search as tuning algorithm:


```r
library("paradox")
library("mlr3tuning")

learner = lrn("classif.rpart")
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))
terminator = trm("evals", n_evals = 10)
tuner = tnr("random_search")

at = AutoTuner$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = tune_ps,
  terminator = terminator,
  tuner = tuner
)
at
```

```
## <AutoTuner:classif.rpart.tuned>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

We can now use the learner like any other learner, calling the `$train()` and `$predict()` method.
This time however, we pass it to [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) to compare the tuner to a classification tree without tuning.
This way, the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) will do its resampling for tuning on the training set of the respective split of the outer resampling.
The learner then undertakes predictions using the test set of the outer resampling.
This yields unbiased performance measures, as the observations in the test set have not been used during tuning or fitting of the respective learner.
This is called [nested resampling](#nested-resampling).

To compare the tuned learner with the learner that uses default values, we can use [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html):


```r
grid = benchmark_grid(
  task = tsk("pima"),
  learner = list(at, lrn("classif.rpart")),
  resampling = rsmp("cv", folds = 3)
)

# avoid console output from mlr3tuning
logger = lgr::get_logger("bbotk")
logger$set_threshold("warn")

bmr = benchmark(grid)
bmr$aggregate(msrs(c("classif.ce", "time_train")))
```

```
##    nr      resample_result task_id          learner_id resampling_id iters
## 1:  1 <ResampleResult[21]>    pima classif.rpart.tuned            cv     3
## 2:  2 <ResampleResult[21]>    pima       classif.rpart            cv     3
##    classif.ce time_train
## 1:     0.2734          0
## 2:     0.2669          0
```

Note that we do not expect any differences compared to the non-tuned approach for multiple reasons:

* the task is too easy
* the task is rather small, and thus prone to overfitting
* the tuning budget (10 evaluations) is small
* [rpart](https://cran.r-project.org/package=rpart) does not benefit that much from tuning
