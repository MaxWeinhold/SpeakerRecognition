#===============================
# Clean and reset safe space
#===============================
rm(list = ls())
setwd("D:/ComicandSonsProductions/GameJam1/SpeakerRecognition")

#===============================
# Load packages
#===============================
needed_packages <- c("tuneR")

new.packages <- needed_packages[!(needed_packages %in% installed.packages()[,"Package"])]
if (length(new.packages)) install.packages(new.packages)

lapply(needed_packages, require, character.only = TRUE)

#===============================
# Config and load data
#===============================

N_MFCCs <- c(20)
window_sizes <- c(1,5,10,20,30,50)

for(x in 1:length(window_sizes)){
  
  N_MFCC <- N_MFCCs[1]
  N_window_size_name <- as.character(window_sizes[x])
  if(N_window_size_name=="1"){N_window_size_name="01"}
  if(N_window_size_name=="5"){N_window_size_name="05"}
  
  input_dir <- "D:/STUDIUM/Münster/VPRonaRaspberryPi/EmoDB/wav"
  output_file <- paste("features/features_ws_",N_window_size_name,"_r.csv",sep="")
  
  files <- list.files(input_dir, pattern = "\\.wav$", full.names = TRUE)
  
  if (length(files) == 0) {
    stop("Keine WAV-Dateien im angegebenen Ordner gefunden!")
  }
  
  features_list <- list()
  
  #===============================
  # Main loop — MFCC extraction
  #===============================
  
  f=files[1]
  for (f in files) {
    cat("Verarbeite:", basename(f), "\n")
    
    #load single wave file
    wav <- readWave(f)
    
    # Mono erzwingen
    if (wav@stereo) {
      wav <- mono(wav, "left")
    }
    
    # Direkt melfcc anwenden (Wave-Objekt wird angenommen)
    mfcc_result <- melfcc(
      samples = wav,      # wichtig: Argument heißt 'samples'
      numcep  = N_MFCC,   # Anzahl der MFCCs
      sr      = wav@samp.rate
    )
    
    # melfcc gibt ein Matrix-Objekt zurück (Frames x Coeffs)
    global_feature_vec <- colMeans(mfcc_result)
    feature_vec <- mfcc_result
    
    window_size <- window_sizes[x]
    i=10
    # i springt in Schritten von window_size → kein Overlap!
    for (i in seq(window_size, nrow(feature_vec), by = window_size)) {
      
      # Fenster definieren
      frame_range <- (i - window_size + 1):i
      
      # MFCC-Frames in Vektor umwandeln
      feature_sub_vec <- as.vector(t(feature_vec[frame_range, ]))
      
      # Sprecher extrahieren
      speaker_label <- substr(basename(f), 1, 2)
      
      # Emotion extrahieren
      emotion_label <- substr(basename(f), 6, 6)
      if(emotion_label=="A"){
        emotion_label="fear"    
      }else if(emotion_label=="E"){
        emotion_label="disgust"      
      }else if(emotion_label=="F"){
        emotion_label="happiness"      
      }else if(emotion_label=="L"){
        emotion_label="boredom"      
      }else if(emotion_label=="N"){
        emotion_label="neutral"      
      }else if(emotion_label=="T"){
        emotion_label="sadness"      
      }else if(emotion_label=="W"){
        emotion_label="anger"      
      }
      # Zeile speichern
      row <- c(file = basename(f), speaker_label, emotion_label, feature_sub_vec,global_feature_vec)
      
      features_list[[length(features_list) + 1]] <- row
    }
    
  }
  
  #===============================
  # Save results
  #===============================
  df <- as.data.frame(do.call(rbind, features_list))
  
  colnames(df) <- c("file","voice","emotion", paste0("mfcc_", 1:(N_MFCC*window_size+length(global_feature_vec))))
  
  nrow(df)
  
  setwd("C:/Users/MaxWe/Documents/!Erfindungen/SpeakerRecognition")
  write.csv(df, output_file, row.names = FALSE)
  
  cat("\nFertig! Features gespeichert in:", output_file, "\n")
  
}
