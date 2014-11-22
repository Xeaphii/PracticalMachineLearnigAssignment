library(caret)
library(Hmisc)

library(foreach)
library(doSNOW)
library(randomForest)

set.seed(32343)

rm(list=ls())

fileUrl<-"./Assignments/Practical Machine Learning/pml-training.csv"
TrainingData<-read.csv(fileUrl,header = T);

#describe(TrainingData)

#After looking at the csv, I save many #DIV/0! terms. Which make my analyis filthy. So firsly I need
#remove them. 
#So, 
TrainingData<-read.csv(fileUrl,header = T,na.strings="#DIV/0!");
#FOr removing not for use columns
TrainingData<-TrainingData[,-(1:7)]

#And then looking at data, some of the variables that are numeric are converted into factors
#So convert them into numeric first. Exploring at the data, it seems that we need to start 
#from column 8

classe<-TrainingData[,ncol(TrainingData)]
TrainingData <-TrainingData[,-1]
TrainingData <- as.data.frame(sapply( TrainingData, function(x) as.numeric(as.character(x))))
TrainingData[,"classe"] <- classe

colIndexes <- colSums(is.na(TrainingData)) <= nrow(TrainingData)*0.25
colIndexes <- as.data.frame(colIndexes)[,1]
colNames<-colnames(TrainingData)[colIndexes]

TrainingData <- TrainingData[,colNames]

inTrain <- createDataPartition(y=TrainingData$classe,
                               p=0.75, list=FALSE)
training <- TrainingData[inTrain,]
testing <- TrainingData[-inTrain,]

registerDoSNOW(makeCluster(4, type="SOCK"))
rf <- foreach(ntree = rep(200, 4), .combine = combine, .packages = "randomForest") %dopar%
            randomForest(training[,-ncol(training)], training[,"classe"], ntree = ntree)
rf
predictionsTr <- predict(rf, newdata=training)
confusionMatrix(predictionsTr,training$classe)

predictionsTr <- predict(rf, newdata=testing)
confusionMatrix(predictionsTr,testing$classe)

fileUrl<-"./Assignments/Practical Machine Learning/pml-testing.csv"
testingData<-read.csv(fileUrl,header = T);

predictionsTr <- predict(rf, newdata=testingData)