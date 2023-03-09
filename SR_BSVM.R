#Voice/Person Recognition: Bootstrapped Support Vector Machines
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

n_machines = 100

model <- list()

training.samples <- VoiceData$Voice %>%
  createDataPartition(p = 0.8, list = FALSE)
TrainData <- VoiceData[training.samples,]
TestData <- VoiceData[-training.samples,]


for(i in 1:n_machines){
  
  print(paste0("Computing model",i))
  
  training.samples <- TrainData$Voice %>%
    createDataPartition(p = 0.8, list = FALSE)
  train.data  <- TrainData[training.samples, ]
  
  model[[i]] <- svm(Voice ~., data =  train.data, type = "nu-classification", kernel = "radial", epsilon = 0.1, nu = 0.0075, tolerance = 0.001, shrinking = TRUE)
  
}

response <- list()
predictions = as.factor(1:nrow(TestData))

i=1
j=1

for(i in 1:n_machines){
  
  print(paste("Making predictions for the model",i))
  
  response[[i]] <- as.data.frame(predict(model[[i]], newdata = TestData, type='response'))
  names(response[[i]])[1] = paste("model",i,sep="_")
  
  if(i==1){
    EvaluationData<-data.frame(TestData$Voice,response[[i]])
    names(EvaluationData)[1] = "Observations"
  }
  else{
    EvaluationData<-data.frame(EvaluationData,response[[i]])
  }
}

EvaluationData[100,2:n_machines+1]
levels(as.factor(TestData$Voice))

matrix = as.data.frame(matrix(1:nrow(TestData)*nlevels(as.factor(TestData$Voice)), nrow = nrow(TestData), ncol = nlevels(as.factor(TestData$Voice))))

names(matrix) = levels(as.factor(TestData$Voice))

i = 1
j = 1
k = 1

EvaluationData[3,2]

for(i in 1:nrow(TestData)){
  
  print(i/nrow(TestData)*100)
  
  #matrix[i,]=0
  
  for(j in 1:nlevels(as.factor(TestData$Voice))){
    
    n = 0
    
    for(k in 1:n_machines){
      
      if(as.character(levels(as.factor(TestData$Voice))[j]) == as.character(EvaluationData[i,k+1])){n = n+1;}
      
    }
    
    matrix[i,j] = n
    
  }
  
}

cummulated_predictions = as.character(1:nrow(TestData))
for(i in 1:nrow(TestData)){
  
  cummulated_predictions[i] = as.character(names(which.max(matrix[i,])))
  
  
}

EvaluationData2<-data.frame(TestData$Voice,cummulated_predictions)
names(EvaluationData2)[1] = "Observations"
EvaluationData2$Accuracy = EvaluationData2$cummulated_predictions == EvaluationData2$Observation
sum(EvaluationData2$cummulated_predictions == EvaluationData2$Observation)/nrow(EvaluationData2)*100
