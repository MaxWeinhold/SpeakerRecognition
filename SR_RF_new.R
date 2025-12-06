#===============================
# Clean and reset safe space
#===============================
rm(list = ls())
setwd("D:/ComicandSonsProductions/GameJam1/SpeakerRecognition")

#===============================
# Load packages
#===============================
needed_packages <- c("randomForest","Rcpp", "caret", "dplyr")

new.packages <- needed_packages[!(needed_packages %in% installed.packages()[,"Package"])]
if (length(new.packages)) install.packages(new.packages)

lapply(needed_packages, require, character.only = TRUE)

#===============================
# Load Feature Dataset
#===============================
data_file <- "features/features_r.csv"

if (!file.exists(data_file)) {
  stop("Feature-Datei nicht gefunden!")
}

VoiceData <- read.csv(data_file, stringsAsFactors = FALSE)

# Sprecher-Variable (speaker_label) in Faktor umwandeln
VoiceData$voice <- as.factor(VoiceData$voice)

#===============================
# Train/Test Split – by full audio files
#===============================

# Alle eindeutigen Audiodateien
files <- unique(VoiceData$file)

set.seed(123)  # Reproduzierbarkeit
train_file_ids <- sample(files, size = floor(0.8 * length(files)), replace = FALSE)

# Höhere Ebene: Auf Zeilenebene danach filtern
train.data <- VoiceData %>% filter(file %in% train_file_ids)
test.data  <- VoiceData %>% filter(!(file %in% train_file_ids))

cat("Training rows:", nrow(train.data), "\n")
cat("Test rows:", nrow(test.data), "\n")

train.data$file <- NULL
test.data$file  <- NULL
train.data$emotion <- NULL
test.data$emotion  <- NULL

train.data.clean <- train.data[complete.cases(train.data), ]
test.data.clean <- test.data[complete.cases(test.data), ]

#===============================
# Train Random Forrest Model
#===============================

# Beispiel: Sprecherklassifikation
model <- randomForest(
  voice ~., 
  data =  train.data.clean, 
  ntree=500, 
  importance=TRUE)

#===============================
# Predictions
#===============================

train_predict <- predict(model, train.data.clean)

test_predict <- predict(model, test.data.clean)

#===============================
# Evaluation – Test Accuracy
#===============================

EvaluationTest <- data.frame(
  Observation = test.data.clean$voice,
  Prediction  = test_predict
)

test_accuracy <- mean(EvaluationTest$Observation == EvaluationTest$Prediction) * 100
cat("Test Accuracy (%):", test_accuracy, "\n") # Im ersten Test: Test Accuracy (%): 52.98831 

#===============================
# Evaluation – Training Accuracy
#===============================

EvaluationTrain <- data.frame(
  Observation = train.data.clean$voice,
  Prediction  = train_predict
)

train_accuracy <- mean(EvaluationTrain$Observation == EvaluationTrain$Prediction) * 100
cat("Training Accuracy (%):", train_accuracy, "\n") # Im ersten Test: Training Accuracy (%): 100 
