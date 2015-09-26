# PML Project
Matt Altieri  






```r
# training data
har.train <- read.csv("data/pml-training.csv", header=T)
# select only the measurement records, not the aggregates
har.train <- har.train[which(har.train$new_window=="no"),]
# select only the quantitative features for non-aggregate obs
har.train <- har.train[,c(features,"classe")]
rownames(har.train) <- NULL

# test data
har.test <- read.csv("data/pml-testing.csv", header=T)
har.test <- har.test[,c(features,"problem_id")]
rownames(har.test) <- NULL
```


```r
# Code adapted from https://stackoverflow.com/questions/13403427/fully-reproducible-parallel-models-using-caret
require(doParallel)
set.seed(4747)
num_repeats <- 1
k <- 10 # Number of folds for k-fold cross-validation
mtry <- ncol(har.train) - 2 # df when response is removed

# generate the seeds for the cross-validation for reproducibility
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


```r
fit$finalModel
```

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

\pagebreak

## Appendix


```r
kable(data.frame(features=features))
```



features             
---------------------
roll_belt            
pitch_belt           
yaw_belt             
total_accel_belt     
gyros_belt_x         
gyros_belt_y         
gyros_belt_z         
accel_belt_x         
accel_belt_y         
accel_belt_z         
magnet_belt_x        
magnet_belt_y        
magnet_belt_z        
roll_arm             
pitch_arm            
yaw_arm              
total_accel_arm      
gyros_arm_x          
gyros_arm_y          
gyros_arm_z          
accel_arm_x          
accel_arm_y          
accel_arm_z          
magnet_arm_x         
magnet_arm_y         
magnet_arm_z         
roll_dumbbell        
pitch_dumbbell       
yaw_dumbbell         
total_accel_dumbbell 
gyros_dumbbell_x     
gyros_dumbbell_y     
gyros_dumbbell_z     
accel_dumbbell_x     
accel_dumbbell_y     
accel_dumbbell_z     
magnet_dumbbell_x    
magnet_dumbbell_y    
magnet_dumbbell_z    
roll_forearm         
pitch_forearm        
yaw_forearm          
total_accel_forearm  
gyros_forearm_x      
gyros_forearm_y      
gyros_forearm_z      
accel_forearm_x      
accel_forearm_y      
accel_forearm_z      
magnet_forearm_x     
magnet_forearm_y     
magnet_forearm_z     
