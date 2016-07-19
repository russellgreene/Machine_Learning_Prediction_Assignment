---
title: "ML Prediction Assignment"
author: "Russell Greene"
date: "July 19, 2016"
output: html_document
---



## Executive Summary

In this assignment, the Weight Lifting Exercise Dataset was analyzed.  I fit three different machine learning models using three different techniques.  The first technique I used a simple classification tree made using recursive partitioning for classification.  It had a validation accuracy of **0.749**.  The second method was gradient boosting which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.  The classification tree built using gradient boosting had a training accuracy of **0.958** and an accuracy of **0.964** on the validation set.  The third technique was random forests which constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes of the individual trees. The random forest model had a training accuracy of **0.99** and an accuracy of **0.99** on the validation set.  

The true test of the random forest model was on the Course Project Prediction Quiz where the model earned a 100% (20 / 20 correct).  The 99% accuracy on the validation data and the 100% match on the prediction quiz suggests that the random forest model was not overfit on the training data.

## Preparing Data Frame for Analysis

The training and test cvs files were downloaded and loaded into dataframes. 

Download data and load to dataframe:

```r
training_file <- '/home/rgreene/Coursera/Data_Science/Machine_Learning/pml-training.csv'
testing_file <- '/home/rgreene/Coursera/Data_Science/Machine_Learning/pml-testing.csv'
if (!file.exists(training_file))
  download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', training_file)
if (!file.exists(testing_file))
  download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', testing_file)

training <- read.csv(training_file)
testing <- read.csv(testing_file)
```

This data needed a fair bit of massaging before I was able to train machine learning models on it.  The first step was to remove the first 7 columns from the dataframe as these columns don't reflect data from the sensors.

After that I converted all factor columns except for the response (which is the last column of the training data) to numeric types.  This is because R has an annoying habit of turning numeric columns with only a handful of unique numbers into factors.  This turns a continuous variable into a discrete variable drastically reducing the accuracy of the algorithm.

The next step was to remove any missing values (denoted by NA) by the column mean.  The last step for the dataframe preparation was to remove any columns that still had NAs (this means that the mean was NA which means that the entire column was missing valid values).


```r
options(warn = -1)
drop_cols = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
training <- training[,!(colnames(training)) %in% drop_cols]

ncoli <- ncol(training)
ncoli <- ncoli - 1
for (i in 1:ncoli) {
    if (class(training[,i]) == "factor") {
     training[,i] = as.numeric(as.character(training[,i])) 
    }
    training[is.na(training[,i]), i] <- mean(training[,i], na.rm=TRUE)
}

# Remove any columns will NA which means that there were no valid values here since all previous missing values were replaced  by mean (this means any remaining NAs had a mean of NA)
training <-training[,colSums(is.na(training))==0]
```

Before doing any exploritory data analysis I sub-divide the training set into a training and validation set so that I can try multiple fits before moving to my test set.


```r
set.seed(123456)
inTrain <- createDataPartition(y = training$classe, p = 0.8, list = FALSE)
trainSet <- training[inTrain, ]
validationSet <- training[-inTrain, ]
```

## Exploratory Data Analysis

This has reduced the set from 160 columns down to 147 predictor variables with 15699 rows.  It's hard to graphically represent this dataset as there are over a hundred variables.


```r
print(dim(trainSet))
```

```
## [1] 15699   147
```

## Building Machine Learning Models

In this anaylsis I build three machine learning models.  The first one is the simplest which is a classification tree made using recursive partitioning for classification.  The advantage to the rpart model is that it is easy to both understand and visualize.  The second technique I used was gradient boosting which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.  The third technique was random forests which constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes of the individual trees.  I used the doMC rpackage which let me parallelize the random forest and gradient boosting algorithms taking advantage of all 4 cores on my machine.


```r
options(warn = -1)
set.seed(1234)
rpart_model <- rpart(classe ~ ., data = trainSet)

set.seed(1234)
gb_model <- train(classe ~ ., data=trainSet, method="gbm", verbose=FALSE)
```

```
## Error in names(resamples) <- gsub("^\\.", "", names(resamples)): attempt to set an attribute on NULL
```

```r
set.seed(1234)
rf_model <- train(classe ~ ., data = trainSet, method = "rf", ntree=100, importance=TRUE)
```

## Model Evaluation

The nice thing with the classification tree is that it is very straightforward.  As seen below we can graphically display the classification tree which allows us to start to build an intuition as to the most important sensors in determining the class of exercise.  However, the validation set accuracy of the single tree model is only **0.749** on the validation data which isn't really great.


```r
options(warn = -1)
prp(rpart_model)		
```

![plot of chunk ml 2](figure/ml 2-1.png)

```r
rpart_predicted_data <- predict(rpart_model, validationSet, type="class")
postResample(rpart_predicted_data, validationSet$classe)
```

```
##  Accuracy     Kappa 
## 0.7494265 0.6810424
```

The second model fit was a gradient boosting model.  This is an ensamble model which is constructed based on many different decision trees.  As you can see below, the validation set accuracy is **0.965** which is a lot better than using the single decision tree.


```r
gb_predicted_data <- predict(gb_model, validationSet)
postResample(gb_predicted_data, validationSet$classe)
```

```
##  Accuracy     Kappa 
## 0.9640581 0.9545357
```

To see if I could improve results even further, I tried a random forest classifier.  In this technique I built 100 decisions trees and took the most common tree to determine the model.  With the random forest, I was able to achieve a validation set accuracy of **0.992**!


```r
rf_predicted_data <- predict(rf_model, validationSet)
postResample(rf_predicted_data, validationSet$classe)
```

```
##  Accuracy     Kappa 
## 0.9923528 0.9903254
```

One of the nice features of random forests in R is that it is possible to determine the most important variables.  Below shows the most important variables in building this classifier.


```r
varImp(rf_model)
```

```
## rf variable importance
## 
##   variables are sorted by maximum importance across the classes
##   only 20 most important variables shown (out of 146)
## 
##                       A     B     C     D      E
## roll_belt         76.12 93.51 86.69 72.82 100.00
## pitch_belt        39.68 96.24 72.63 51.80  43.30
## pitch_forearm     53.70 64.13 85.90 61.07  63.93
## magnet_dumbbell_y 65.39 71.46 83.45 70.07  60.56
## magnet_dumbbell_z 69.72 56.67 73.39 51.03  48.03
## yaw_belt          70.07 63.93 61.22 68.92  54.27
## accel_forearm_x   26.25 48.36 38.03 56.28  43.75
## roll_forearm      52.34 44.65 49.74 40.11  40.79
## yaw_arm           48.50 39.44 35.78 36.97  23.62
## gyros_dumbbell_y  38.22 35.74 47.06 37.14  36.93
## accel_dumbbell_y  42.09 38.61 45.14 36.67  37.90
## gyros_belt_z      31.66 36.49 40.60 25.82  42.08
## magnet_arm_y      26.02 39.08 27.56 34.34  24.64
## magnet_belt_x     19.72 38.84 35.60 25.74  33.94
## yaw_dumbbell      25.24 38.68 27.33 27.48  34.10
## pitch_arm         22.38 38.50 25.70 23.32  28.06
## magnet_belt_y     22.69 38.49 31.42 28.35  30.77
## gyros_arm_x       26.26 35.02 38.45 28.04  29.90
## gyros_forearm_y   23.98 38.34 35.65 28.27  23.93
## roll_dumbbell     28.56 38.24 28.54 30.92  35.10
```

## Conclusion

The random forest predictor yields the best performance on the the validation set.  It is now finally time to open up the testing set.  However, before we can use the testing set we have to pre-process the data (in similar fashion to how we pre-processed the training data).  In the code below I drop the columns without sensor data.  I also convert any factor columns to numeric fields and fill in the remaining columns with NAs (which at this point mean there are NAs in the entire column) with 0.  Unlike the training data, I don't drop these columns as my fitted model might be looking for these fields.   


```r
options(warn = -1)
drop_cols = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
testing <- testing[,!(colnames(testing)) %in% drop_cols]

for (i in 1:ncol(testing)) {
    if (class(testing[,i]) == "factor") {
     testing[,i] = as.numeric(as.character(testing[,i])) 
    }
    testing[is.na(testing[,i]), i] <- mean(testing[,i], na.rm=TRUE)
}

# Any columns with NA which means that there were no valid values here since all previous missing values were replaced by mean (this means any remaining NAs had a mean of NA).  Replace with 0 since this is testing set.
for (i in 1:ncol(testing)) {
  testing[is.na(testing[,i]), i] <- 0
}
```

Seen below, my random forest predictor makes the follownig predictions for the first 20 classes.


```r
options(warn = -1)
testPrediction <- predict(rf_model, testing)
print(rbind(testing[1:20, 160], as.character(testPrediction)))
```

```
##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13]
## [1,] "B"  "A"  "B"  "A"  "A"  "E"  "D"  "B"  "A"  "A"   "B"   "C"   "B"  
##      [,14] [,15] [,16] [,17] [,18] [,19] [,20]
## [1,] "A"   "E"   "E"   "A"   "B"   "B"   "B"
```

The true test of the random forest model was on the Course Project Prediction Quiz where the model earned a 100% (20 / 20 correct).  The 99% accuracy on the validation data and the 100% match on the prediction quiz suggests that the random forest model was not overfit on the training data.
