# PML Project
Matt Altieri  



## Overview

Our goal with this analysis is to determine if we can detect the quality of a human activity -- whether an exercise motion was performed correctly or incorrectly.

To do this, we have split our data into training and test sets of the following size.


```r
# training data
har.train <- read.csv("data/pml-training.csv", header=T)

# test data
har.test <- read.csv("data/pml-testing.csv", header=T)
```

<!-- html table generated in R 3.1.2 by xtable 1.7-4 package -->
<!-- Sat Sep 26 13:11:29 2015 -->
<table border=1>
<caption align="bottom"> Training & Test Sets </caption>
<tr> <th>  </th> <th> Observations </th> <th> Variables </th>  </tr>
  <tr> <td align="right"> Training </td> <td align="right"> 19622 </td> <td align="right"> 160 </td> </tr>
  <tr> <td align="right"> Testing </td> <td align="right">  20 </td> <td align="right"> 160 </td> </tr>
   </table>


The data we will use was originally sourced from [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)$^1$

## Data Exploration

In the data we will use, both individual measurements (`new_window="no"`) and aggregated measurements over time (`new_window="yes"`) are available.

<!-- html table generated in R 3.1.2 by xtable 1.7-4 package -->
<!-- Sat Sep 26 13:11:29 2015 -->
<table border=1>
<caption align="bottom"> Frequency & types of records </caption>
<tr> <th> new_window </th> <th> count </th>  </tr>
  <tr> <td> no </td> <td align="right"> 19216 </td> </tr>
  <tr> <td> yes </td> <td align="right"> 406 </td> </tr>
   </table>

Each training record is assigned a `classe` of A through E. We wish to use the quantitative training data to predict these values in the test data, for individual measurements (`new_window="no"`) only, so we will filter the training data to only those types of records.


```r
# select only the measurement records, not the aggregates
har.train <- har.train[which(har.train$new_window=="no"),]
```

Many of the variables are only used to contain aggregate measurements (`min_roll_belt`, `max_yaw_belt`), so we will remove them from our list of feature candidates. Additionally, since our goal is to detect the quality of the exercise in the general case, information about the user, as well as timestamps and measurement windows will be removed.




```r
# training data
# select only the quantitative features for non-aggregate obs
har.train <- har.train[,c(features,"classe")]
rownames(har.train) <- NULL

# test data
har.test <- har.test[,c(features,"problem_id")]
rownames(har.test) <- NULL
```

The contents of the `features` vector can be found in the figure _Variables evaluated for model features_ in the **Appendix**.

## Random Forest Model

Our goal is to categorize a lot of numerical data about human movement

1. Into 5 broad categories
2. For the general case (not user or time specific)

To do this, we will model a random forest in order to fit to a high-degree of accuracy. Because of the risk of overfitting, we will use k-fold cross-validation to minimize the chance of overfitting. This will also allow us to estimate the out-of-bag error rate.


```r
# Parallel processing code adapted from https://stackoverflow.com/questions/13403427/fully-reproducible-parallel-models-using-caret
require(doParallel)
set.seed(4747)
num_repeats <- 1
k <- 10 # Number of folds for k-fold cross-validation
mtry <- ncol(har.train) - 2 # num of candidate features - 1

# generate the seeds for the random folds for reproducibility
seeds <- vector(mode="list", length=(num_repeats * k) + 1)
for(i in 1:10) seeds[[i]] <- sample.int(1000, mtry)
seeds[[11]] <- sample.int(1000, 1)

# Define k-fold cross-validation controls for 10 folds to estimate out-of-bag error
myControl <- trainControl(method="oob", seeds=seeds, index=createFolds(har.train$classe, k=k))

# Train our model using random forest. Perform training in 4 parallel threads for increased speed
cl <- makeCluster(detectCores())
registerDoParallel(cl)
fit <- train(classe ~ ., har.train, method="rf", trControl=myControl)
stopCluster(cl)
```

## Model Results


```

Call:
 randomForest(x = x, y = y, mtry = param$mtry) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 2

        OOB estimate of  error rate: 0.41%
Confusion matrix:
     A    B    C    D    E  class.error
A 5468    3    0    0    0 0.0005483458
B    8 3706    4    0    0 0.0032275417
C    0   19 3330    3    0 0.0065632458
D    0    0   37 3109    1 0.0120749921
E    0    0    0    4 3524 0.0011337868
```




```r
har.test$pred.classe <- predict(fit, har.test)

pml_write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        filename <- paste0("data/problem_id_",i,".txt")
        write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
    }
}

pml_write_files(as.character(har.test$pred.classe))
```

## Appendix

<!-- html table generated in R 3.1.2 by xtable 1.7-4 package -->
<!-- Sat Sep 26 13:14:43 2015 -->
<table border=1>
<caption align="top"> Variables evaluated for model features </caption>
<tr> <th> features </th>  </tr>
  <tr> <td> roll_belt </td> </tr>
  <tr> <td> pitch_belt </td> </tr>
  <tr> <td> yaw_belt </td> </tr>
  <tr> <td> total_accel_belt </td> </tr>
  <tr> <td> gyros_belt_x </td> </tr>
  <tr> <td> gyros_belt_y </td> </tr>
  <tr> <td> gyros_belt_z </td> </tr>
  <tr> <td> accel_belt_x </td> </tr>
  <tr> <td> accel_belt_y </td> </tr>
  <tr> <td> accel_belt_z </td> </tr>
  <tr> <td> magnet_belt_x </td> </tr>
  <tr> <td> magnet_belt_y </td> </tr>
  <tr> <td> magnet_belt_z </td> </tr>
  <tr> <td> roll_arm </td> </tr>
  <tr> <td> pitch_arm </td> </tr>
  <tr> <td> yaw_arm </td> </tr>
  <tr> <td> total_accel_arm </td> </tr>
  <tr> <td> gyros_arm_x </td> </tr>
  <tr> <td> gyros_arm_y </td> </tr>
  <tr> <td> gyros_arm_z </td> </tr>
  <tr> <td> accel_arm_x </td> </tr>
  <tr> <td> accel_arm_y </td> </tr>
  <tr> <td> accel_arm_z </td> </tr>
  <tr> <td> magnet_arm_x </td> </tr>
  <tr> <td> magnet_arm_y </td> </tr>
  <tr> <td> magnet_arm_z </td> </tr>
  <tr> <td> roll_dumbbell </td> </tr>
  <tr> <td> pitch_dumbbell </td> </tr>
  <tr> <td> yaw_dumbbell </td> </tr>
  <tr> <td> total_accel_dumbbell </td> </tr>
  <tr> <td> gyros_dumbbell_x </td> </tr>
  <tr> <td> gyros_dumbbell_y </td> </tr>
  <tr> <td> gyros_dumbbell_z </td> </tr>
  <tr> <td> accel_dumbbell_x </td> </tr>
  <tr> <td> accel_dumbbell_y </td> </tr>
  <tr> <td> accel_dumbbell_z </td> </tr>
  <tr> <td> magnet_dumbbell_x </td> </tr>
  <tr> <td> magnet_dumbbell_y </td> </tr>
  <tr> <td> magnet_dumbbell_z </td> </tr>
  <tr> <td> roll_forearm </td> </tr>
  <tr> <td> pitch_forearm </td> </tr>
  <tr> <td> yaw_forearm </td> </tr>
  <tr> <td> total_accel_forearm </td> </tr>
  <tr> <td> gyros_forearm_x </td> </tr>
  <tr> <td> gyros_forearm_y </td> </tr>
  <tr> <td> gyros_forearm_z </td> </tr>
  <tr> <td> accel_forearm_x </td> </tr>
  <tr> <td> accel_forearm_y </td> </tr>
  <tr> <td> accel_forearm_z </td> </tr>
  <tr> <td> magnet_forearm_x </td> </tr>
  <tr> <td> magnet_forearm_y </td> </tr>
  <tr> <td> magnet_forearm_z </td> </tr>
   </table>

## Citations

1: _Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013._
