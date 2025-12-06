#===============================
# Clean and reset safe space
#===============================
rm(list = ls())
setwd("C:/Users/MaxWe/Documents/!Erfindungen/SpeakerRecognition")

#===============================
# Load packages
#===============================

needed_packages <- c("e1071","caret","dplyr","ggplot2","openxlsx","Boruta","randomForest")

new.packages <- needed_packages[!(needed_packages %in% installed.packages()[,"Package"])]
if (length(new.packages)) install.packages(new.packages)

lapply(needed_packages, require, character.only = TRUE)

#===============================
# Load Feature Dataset
#===============================

nmb_of_features = c(20)
window_sizes <- c(1,5,10,20,30,50)
training_time_min = c(1:length(window_sizes))*0

all_train_accuracy = c(1:length(window_sizes))*0
all_train_kappa = c(1:length(window_sizes))*0
all_train_recall = c(1:length(window_sizes))*0
all_train_specificity = c(1:length(window_sizes))*0
all_train_positive_predictive_value = c(1:length(window_sizes))*0
all_train_positive_predictive_value = c(1:length(window_sizes))*0
all_train_negative_predictive_value = c(1:length(window_sizes))*0
all_train_negative_balanced_accuracy = c(1:length(window_sizes))*0

all_test_accuracy = c(1:length(window_sizes))*0
all_test_kappa = c(1:length(window_sizes))*0
all_test_recall = c(1:length(window_sizes))*0
all_test_specificity = c(1:length(window_sizes))*0
all_test_positive_predictive_value = c(1:length(window_sizes))*0
all_test_positive_predictive_value = c(1:length(window_sizes))*0
all_test_negative_predictive_value = c(1:length(window_sizes))*0
all_test_negative_balanced_accuracy = c(1:length(window_sizes))*0

data_files <- c("features/features_ws_01_r.csv",
                "features/features_ws_05_r.csv",
                "features/features_ws_10_r.csv",
                "features/features_ws_20_r.csv",
                "features/features_ws_30_r.csv",
                "features/features_ws_50_r.csv")

x=1
for(x in 1:length(data_files)){
  
  data_file <- data_files[x]
  
  if (!file.exists(data_file)) {
    stop("Feature-Datei nicht gefunden!")
  }
  
  print(paste("Loaded file:",data_files[x]))
  
  VoiceData <- read.csv(data_file, stringsAsFactors = FALSE)
  
  # Sprecher-Variable (speaker_label) in Faktor umwandeln
  VoiceData$voice <- as.factor(VoiceData$voice)
  nmb_of_voices = nlevels(VoiceData$voice)
  voices = levels(VoiceData$voice)
  oneVSall_training_time_min = c(1:nmb_of_voices)*0
  
  voices_train_accuracy = c(1:nmb_of_voices)*0
  voices_train_kappa = c(1:length(nmb_of_voices))*0
  voices_train_recall = c(1:length(nmb_of_voices))*0
  voices_train_specificity = c(1:length(nmb_of_voices))*0
  voices_train_positive_predictive_value = c(1:length(nmb_of_voices))*0
  voices_train_positive_predictive_value = c(1:length(nmb_of_voices))*0
  voices_train_negative_predictive_value = c(1:length(nmb_of_voices))*0
  voices_train_negative_balanced_accuracy = c(1:length(nmb_of_voices))*0
  
  voices_test_accuracy = c(1:nmb_of_voices)*0
  voices_test_kappa = c(1:length(nmb_of_voices))*0
  voices_test_recall = c(1:length(nmb_of_voices))*0
  voices_test_specificity = c(1:length(nmb_of_voices))*0
  voices_test_positive_predictive_value = c(1:length(nmb_of_voices))*0
  voices_test_positive_predictive_value = c(1:length(nmb_of_voices))*0
  voices_test_negative_predictive_value = c(1:length(nmb_of_voices))*0
  voices_test_negative_balanced_accuracy = c(1:length(nmb_of_voices))*0
  
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
  
  train.data <- train.data[complete.cases(train.data), ]
  test.data <- test.data[complete.cases(test.data), ]
  
  #===============================
  # One-vs-All-loop
  #===============================
  
  i=2
  for(i in 1:nmb_of_voices){
    
    print(paste("Classify voice ",i,"/",nmb_of_voices,sep=""))
    
    oneVSall_train.data = train.data
    oneVSall_test.data = test.data
    
    oneVSall_train.data$voice = train.data$voice == voices[i]
    oneVSall_test.data$voice = test.data$voice == voices[i]
    any(oneVSall_train.data$voice)
    
    #===============================
    # Feature Selection Boruta
    #===============================
    
    print("Boruta Feature Selection")
    
    # feat_cols: alle mfcc-Spalten im oneVSall_train.data
    feat_cols <- grep("^mfcc_", names(oneVSall_train.data), value = TRUE)
    
    # Boruta auf Training (binäres target voice)
    set.seed(42)
    bor <- Boruta(as.formula(paste("voice ~", paste(feat_cols, collapse = "+"))),
                  data = oneVSall_train.data, doTrace = 0, maxRuns = 100)
    
    # Wähle bestätigte Features
    sel <- getSelectedAttributes(bor, withTentative = FALSE)
    length(sel)   # Anzahl ausgewählter Features
    
    # Falls nur wenige Features ausgewählt, optional Tentative klären:
    if(length(sel) < 5){
      bor <- TentativeRoughFix(bor)
      sel <- getSelectedAttributes(bor, withTentative = FALSE)
    }
    
    # Subset Daten
    oneVSall_train_reduced <- oneVSall_train.data[, c("voice", sel)]
    oneVSall_test_reduced  <- oneVSall_test.data[, c("voice", sel)]
    
    #===============================
    # Train SVM Model
    #===============================
    
    start_time <- Sys.time()
    
    print(paste("Model training at:",start_time))
    
    # Beispiel: Sprecherklassifikation
    model <- svm(
      voice ~ .,   # file & emotion nicht als Features benutzen
      data = oneVSall_train_reduced,
      type = "C-classification", 
      kernel = "radial",
      cost = 1, 
      gamma = 1 / ncol(oneVSall_train_reduced),
      probability = TRUE
    )
    
    end_time <- Sys.time()
    
    runtime_minutes <- as.numeric(difftime(end_time, start_time, units = "mins"))
    oneVSall_training_time_min[i] <- runtime_minutes
    
    #===============================
    # Predictions
    #===============================
    
    train_predict <- predict(model, oneVSall_train.data)
    
    test_predict <- predict(model, oneVSall_test.data)
    
    #===============================
    # Evaluation – Test Accuracy
    #===============================
    
    EvaluationTest <- data.frame(
      Observation = oneVSall_test.data$voice,
      Prediction  = test_predict
    )
    
    test_accuracy <- mean(EvaluationTest$Observation == EvaluationTest$Prediction) * 100
    cat("Test Accuracy (%):", test_accuracy, "\n")
    voices_test_accuracy[i] = test_accuracy
    
    cm = confusionMatrix(as.factor(EvaluationTest$Prediction),
                    as.factor(EvaluationTest$Observation),
                    positive = "TRUE")
    
    cat("Test balance Accuracy (%):", round((as.numeric(cm$byClass[11]))*100,2), "\n")
    
    voices_test_kappa[i] = as.numeric(cm$overall[2])
    voices_test_recall[i] = as.numeric(cm$byClass[6])
    voices_test_specificity[i] = as.numeric(cm$byClass[2])
    voices_test_positive_predictive_value[i] = as.numeric(cm$byClass[3])
    voices_test_negative_predictive_value[i] = as.numeric(cm$byClass[4])
    voices_test_negative_balanced_accuracy[i] = as.numeric(cm$byClass[11])
    
    #===============================
    # Evaluation – Training Accuracy
    #===============================
    
    EvaluationTrain <- data.frame(
      Observation = oneVSall_train.data$voice,
      Prediction  = train_predict
    )
    
    train_accuracy <- mean(EvaluationTrain$Observation == EvaluationTrain$Prediction) * 100
    cat("Training Accuracy (%):", train_accuracy, "\n")
    voices_train_accuracy[i] = train_accuracy
    
    cm = confusionMatrix(as.factor(EvaluationTrain$Prediction),
                         as.factor(EvaluationTrain$Observation),
                         positive = "TRUE")
    
    cat("Train balance Accuracy (%):", round((as.numeric(cm$byClass[11]))*100,2), "\n")
    
    voices_train_kappa[i] = as.numeric(cm$overall[2])
    voices_train_recall[i] = as.numeric(cm$byClass[6])
    voices_train_specificity[i] = as.numeric(cm$byClass[2])
    voices_train_positive_predictive_value[i] = as.numeric(cm$byClass[3])
    voices_train_negative_predictive_value[i] = as.numeric(cm$byClass[4])
    voices_train_negative_balanced_accuracy[i] = as.numeric(cm$byClass[11])
  
  }
  
  training_time_min[x] <- mean(oneVSall_training_time_min)
  all_test_accuracy[x] = mean(voices_test_accuracy)
  all_test_kappa[x] = mean(voices_test_kappa)
  all_test_recall[x] = mean(voices_test_recall)
  all_test_specificity[x] = mean(voices_test_specificity)
  all_test_positive_predictive_value[x] = mean(voices_test_positive_predictive_value)
  all_test_negative_predictive_value[x] = mean(voices_test_negative_predictive_value)
  all_test_negative_balanced_accuracy[x] = mean(voices_test_negative_balanced_accuracy)
  
  all_train_accuracy[x] = mean(voices_train_accuracy)
  all_train_kappa[x] = mean(voices_train_kappa)
  all_train_recall[x] = mean(voices_train_recall)
  all_train_specificity[x] = mean(voices_train_specificity)
  all_train_positive_predictive_value[x] = mean(voices_train_positive_predictive_value)
  all_train_negative_predictive_value[x] = mean(voices_train_negative_predictive_value)
  all_train_negative_balanced_accuracy[x] = mean(voices_train_negative_balanced_accuracy)
  
}

results = as.data.frame(cbind(
  data_files,
  nmb_of_features,
  window_sizes,
  round(all_train_accuracy,2),
  all_train_kappa,
  all_train_recall,
  all_train_specificity,
  all_train_positive_predictive_value,
  all_train_negative_predictive_value,
  round(all_train_negative_balanced_accuracy,2),
  round(all_test_accuracy,2),
  all_test_kappa,
  all_test_recall,
  all_test_specificity,
  all_test_positive_predictive_value,
  all_test_negative_predictive_value,
  round(all_test_negative_balanced_accuracy,2),
  training_time_min))

p <- ggplot(results, aes(x = as.numeric(window_sizes))) +
  geom_line(aes(y = all_train_negative_balanced_accuracy, color = "Train balanced Accuracy", group = 1), linewidth = 1) +
  geom_line(aes(y = all_test_negative_balanced_accuracy,  color = "Test balanced Accuracy",  group = 1), linewidth = 1) +
  geom_point(aes(y = all_train_negative_balanced_accuracy, color = "Train balanced Accuracy"), size = 3) +
  geom_point(aes(y = all_test_negative_balanced_accuracy,  color = "Test balanced Accuracy"),  size = 3) +
  scale_color_manual(values = c("Train balanced Accuracy" = "red", "Test balanced Accuracy" = "blue")) +
  labs(
    x = "Number of Frames from Audio (Window)",
    y = "Accuracy (%)",
    color = "Metric"
  ) +
  theme_minimal()

print(p)
ggsave("accuracy_ws_plot.png", p, width = 8, height = 5, dpi = 300)
write.xlsx(results, file = "results_svm_ws_r.xlsx", overwrite = TRUE)
