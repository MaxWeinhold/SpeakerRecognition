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
# Config
#===============================
N_MFCC <- 20

input_dir <- "D:/STUDIUM/Münster/VPRonaRaspberryPi/EmoDB/wav"
output_file <- "features_r.csv"

files <- list.files(input_dir, pattern = "\\.wav$", full.names = TRUE)

if (length(files) == 0) {
  stop("Keine WAV-Dateien im angegebenen Ordner gefunden!")
}

features_list <- list()

#===============================
# Main loop — MFCC extraction
#===============================
for (f in files) {
  cat("Verarbeite:", basename(f), "\n")
  
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
  feature_vec <- colMeans(mfcc_result)
  
  row <- c(file = basename(f), feature_vec)
  features_list[[length(features_list) + 1]] <- row
}

#===============================
# Save results
#===============================
df <- as.data.frame(do.call(rbind, features_list))

colnames(df) <- c("file", paste0("mfcc_", 1:N_MFCC))

write.csv(df, output_file, row.names = FALSE)

cat("\nFertig! Features gespeichert in:", output_file, "\n")
