#Practical Machine Learning Assignment

##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

##Data 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

##Goal
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. I may use any of the other variables to predict with.

##Analysis

First of all, I load the libraries which I used for analysis, then, I remove all the previous variables stored in memory for carrying down my analysis.


```r
rm(list = ls())
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(Hmisc))

suppressPackageStartupMessages(library(foreach))
suppressPackageStartupMessages(library(doSNOW))
suppressPackageStartupMessages(library(randomForest))
```

 Then, I set seed for making my research reproducible
 

```r
set.seed(32343)
```

Then, I take data into R workspace 

```r
fileUrl <- "pml-training.csv"
TrainingData <- read.csv(fileUrl, header = T)
```

But, before proceeding further into analysis. I check the data for any abnormalitites or getting general overview of the data.


```r
describe(TrainingData)
summary(TrainingData)
str(TrainingData)
sapply(TrainingData, class)
```

After exploring results of that, I catch up some problems in this dataset. Like After looking at the csv, it have many #DIV/0! terms. Which I think might be indication of na's. And also there are some columns that have very few entries. And also some of the columns are converted into factors, which should be numeric. But due to #DIV/0!. 

And looking at the columns, I came up with removing first 7 columns, which I think that have no impact on classe. So, I solve them by

```r
TrainingData <- read.csv(fileUrl, header = T, na.strings = "#DIV/0!")

# FOr removing not for use columns
TrainingData <- TrainingData[, -(1:7)]
```

And converting them into numeric.


```r
classe <- TrainingData[, ncol(TrainingData)]
TrainingData <- TrainingData[, -1]
TrainingData <- as.data.frame(sapply(TrainingData, function(x) as.numeric(as.character(x))))
TrainingData[, "classe"] <- classe
```

Then, only considering those columns that have ateats 75% of the data, and choosing those columns that qualify for that criteria.


```r
colIndexes <- colSums(is.na(TrainingData)) <= nrow(TrainingData) * 0.25
colIndexes <- as.data.frame(colIndexes)[, 1]
colNames <- colnames(TrainingData)[colIndexes]

TrainingData <- TrainingData[, colNames]
```

So, now data is cleaned enough to be used for prediction. So, Firstly I split the data into trainig and testing set.

```r
inTrain <- createDataPartition(y = TrainingData$classe, p = 0.75, list = FALSE)
training <- TrainingData[inTrain, ]
testing <- TrainingData[-inTrain, ]
```

Firstly, I used generalized linear model, and spline model. But, they have very low accuracy. Then, I tried random forest and it end up in quite good accuracy. And also it takes enormous time to compute serially. So, I used foreach and DoSnow package for running this code on multiple threads. So, it end up in small elapse of time. So I ask to process 4 random forest with 200 trees each and combine then to have a random forest model with a total of 800 trees.


```r
registerDoSNOW(makeCluster(4, type = "SOCK"))
rf <- foreach(ntree = rep(200, 4), .combine = combine, .packages = "randomForest") %dopar% 
    randomForest(training[, -ncol(training)], training[, "classe"], ntree = ntree)
```

```r
rf
```

Then, I predicted results of my analysis on my testing data and see confusion matrix for seeing accuracy and various other error measures.

```r
results <- predict(rf, newdata = training)
```



```r
confusionMatrix(results, training$classe)
```


And it comes up with awesome accuracy. The, I tested on testing data set. 

```r
results <- predict(rf, newdata = testing)
```

`

```r
confusionMatrix(results, testing$classe)
```


And it again end up in 0.99% accuracy, which indicates that my models performs quite well. And also look at the sensitivity and specificity, which are quite good. Then I tested my results on tesing dataset provided for that assignment. And I get quite good results. I get 20/20 in assigment online submission. Which indicates that my model god quite good accuracy. 
           
