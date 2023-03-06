#Voice/Person Recognition: Support Vector Machines
#Author: Max Weinhold

#In order to create a SVR model with R you will need the package e1071
if(!require("e1071")) install.packages("e1071")
library(e1071)

library(tidyverse)
library(sandwich)
library(caret)

#Clean up memory
rm(list=ls())

#Load Data
setwd("D:/ComicandSonsProductions/GameJam1/SpeakerRecognition/TestData")
VoiceData = read.csv(file = "Test.csv",sep=",")

VoiceData[1]=NULL
names(VoiceData)[1]="Voice"
class(VoiceData$Voice)
VoiceData$Voice = as.factor(VoiceData$Voice)
nrow(VoiceData)

# Split data to reduce duration of computation
training.samples <- VoiceData$Voice %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- VoiceData[training.samples, ]
test.data <- VoiceData[-training.samples, ]

model <- svm(Voice ~., data =  train.data, type = "C-classification", kernel = "polynomial", epsilon = 0.1)

test_predict <- as.data.frame(predict(model, newdata = test.data, type='response'))
train_predict <- as.data.frame(predict(model, data = train.data, type='response'))

EvaluationData<-data.frame(test.data$Voice,test_predict)
names(EvaluationData)<-c("Observation","Prediction")
EvaluationData$Accuracy = EvaluationData$Prediction == EvaluationData$Observation
sum(EvaluationData$Prediction == EvaluationData$Observation)/nrow(EvaluationData)*100

EvaluationData<-data.frame(train.data$Voice,train_predict)
names(EvaluationData)<-c("Observation","Prediction")
EvaluationData$Accuracy = EvaluationData$Prediction == EvaluationData$Observation
sum(EvaluationData$Prediction == EvaluationData$Observation)/nrow(EvaluationData)*100

