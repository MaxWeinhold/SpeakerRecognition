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

Training = c(1:12)
Testing = c(1:12)

for(a in 1:12){
  
  print(a)
  
  #Load Data
  setwd("D:/STUDIUM/Münster/VPRonaRaspberryPi/SpeakerRecognitionDB")
  
  if(a==1){VoiceData = read.csv(file = "Frequencies12.csv",sep=",")}
  if(a==2){VoiceData = read.csv(file = "Frequencies24.csv",sep=",")}
  if(a==3){VoiceData = read.csv(file = "Frequencies36.csv",sep=",")}
  if(a==4){VoiceData = read.csv(file = "Frequencies48.csv",sep=",")}
  if(a==5){VoiceData = read.csv(file = "Frequencies60.csv",sep=",")}
  if(a==6){VoiceData = read.csv(file = "Frequencies90.csv",sep=",")}
  if(a==7){VoiceData = read.csv(file = "Frequencies120.csv",sep=",")}
  if(a==8){VoiceData = read.csv(file = "Frequencies150.csv",sep=",")}
  if(a==9){VoiceData = read.csv(file = "Frequencies200.csv",sep=",")}
  if(a==10){VoiceData = read.csv(file = "Frequencies300.csv",sep=",")}
  if(a==11){VoiceData = read.csv(file = "Frequencies400.csv",sep=",")}
  if(a==12){VoiceData = read.csv(file = "Frequencies500.csv",sep=",")}
  
  VoiceData[1]=NULL
  names(VoiceData)[1]="Voice"
  class(VoiceData$Voice)
  VoiceData$Voice = as.factor(VoiceData$Voice)
  nrow(VoiceData)
  ncol(VoiceData)
  
  print(levels(VoiceData$Voice))
  
  set.seed(2023)
  trainHits = c(1:10)
  testHits = c(1:10)
  
  for(b in 1:10){
    
    training.samples <- VoiceData$Voice %>%
      createDataPartition(p = 0.8, list = FALSE)
    train.data  <- VoiceData[training.samples, ]
    test.data <- VoiceData[-training.samples, ]
    
    model <- svm(Voice ~., data =  train.data, type = "nu-classification", kernel = "radial", epsilon = 0.1, nu = 0.0075, tolerance = 0.001, shrinking = TRUE)
    
    test_predict <- as.data.frame(predict(model, newdata = test.data, type='response'))
    train_predict <- as.data.frame(predict(model, data = train.data, type='response'))
    
    EvaluationData<-data.frame(test.data$Voice,test_predict)
    names(EvaluationData)<-c("Observation","Prediction")
    EvaluationData$Accuracy = EvaluationData$Prediction == EvaluationData$Observation
    testHits[b] = sum(EvaluationData$Prediction == EvaluationData$Observation)/nrow(EvaluationData)*100
    
    EvaluationData<-data.frame(train.data$Voice,train_predict)
    names(EvaluationData)<-c("Observation","Prediction")
    EvaluationData$Accuracy = EvaluationData$Prediction == EvaluationData$Observation
    trainHits[b] = sum(EvaluationData$Prediction == EvaluationData$Observation)/nrow(EvaluationData)*100
    
  }
  
  Training[a] = mean(trainHits)
  Testing[a] = mean(testHits)
  
}

visualData=as.data.frame(cbind(c(12,24,36,48,60,90,120,150,200,300,400,500),Testing))
plot(c(12,24,36,48,60,90,120,150,200,300,400,500),Testing)

plot = ggplot(data=visualData)+
  geom_line(aes(x=V1, y=Testing), size = 2) +
  labs(y = "Correct Predictions in %"
       , x = "Number of Frequencies"
       , title = "Support Vector Machine Performance") +
  theme_bw() + 
  theme(legend.text=element_text(size=12)) +
  theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold"))

print(plot)

setwd("D:/ComicandSonsProductions/GameJam1/SpeakerRecognition")
png(file="Plot_SVM_FreqcuencyValidation.png",width=800, height=800)
plot
dev.off()
