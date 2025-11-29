library(tuneR)
library(seewave)
library(dplyr)
library(readr)
library(stringr)

AUDIO_DIR <- "D:/STUDIUM/Münster/VPRonaRaspberryPi/EmoDB/wav"
FILE_LIST_CSV <- "D:/STUDIUM/Münster/VPRonaRaspberryPi/EmoDB/wav/file_list.csv"
OUTPUT_CSV <- "D:/ComicandSonsProductions/GameJam1/SpeakerRecognition/features_fixed_R.csv"

SR <- 16000
N_MFCC <- 20
MIN_FRAMES <- 20
MAX_FRAMES <- 200

# Datei-Liste einlesen
load_file_list <- function(path) {
  if (!file.exists(path)) {
    return(character(0))
  }
  vec <- read_lines(path)
  vec <- str_trim(vec)
  vec <- vec[vec != ""]
  # Falls CSV mit Kommata:
  vec <- sapply(vec, function(x) str_split(x, ",")[[1]][1])
  vec
}

# MFCC + Delta + Delta2 + Aggregation
extract_features <- function(y, sr) {
  # MFCC berechnen (seewave)
  m <- melfcc(wave = y, sr = sr, numcep = N_MFCC, usecmp = FALSE)
  
  if (is.null(m) || nrow(m) == 0) {
    return(NULL)
  }
  
  # Delta und Delta2
  d1 <- deltas(m)
  d2 <- deltas(d1)
  
  feat <- rbind(m, d1, d2)
  
  n_frames <- ncol(feat)
  
  # Padding/Trimming
  if (n_frames < MIN_FRAMES) {
    padded <- matrix(0, nrow = nrow(feat), ncol = MAX_FRAMES)
    padded[, 1:n_frames] <- feat
    feat <- padded
  } else if (n_frames > MAX_FRAMES) {
    feat <- feat[, 1:MAX_FRAMES]
  }
  
  # Mittelwert + Standardabweichung pro Feature
  mean_vec <- apply(feat, 1, mean)
  std_vec <- apply(feat, 1, sd)
  
  c(mean_vec, std_vec)
}

# ==== Hauptteil =======
file_names <- load_file_list(FILE_LIST_CSV)
cat("[INFO] Einträge in Datei-Liste:", length(file_names), "\n")

# Fallback: WAV suchen, wenn CSV leer
if (length(file_names) == 0) {
  cat("[INFO] CSV leer oder fehlend – suche WAV-Dateien im AUDIO_DIR\n")
  wavs <- list.files(AUDIO_DIR, pattern = "\\.(wav|mp3|flac)$", ignore.case = TRUE)
  file_names <- wavs
}

rows <- list()
labels <- c()
skipped <- 0

for (i in seq_along(file_names)) {
  fname <- file_names[i]
  full_path <- file.path(AUDIO_DIR, fname)
  
  if (!file.exists(full_path)) {
    cat("[WARN] Datei nicht gefunden:", full_path, "\n")
    skipped <- skipped + 1
    next
  }
  
  cat("[", i, "/", length(file_names), "] Verarbeite:", fname, "\n")
  
  # Audio laden
  wav <- tryCatch({
    readWave(full_path)
  }, error = function(e) {
    cat("[ERROR] Fehler beim Laden:", fname, "->", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(wav)) {
    skipped <- skipped + 1
    next
  }
  
  # Resampling auf 16 kHz
  if (wav@samp.rate != SR) {
    wav <- tryCatch({
      tuneR::downsample(wav, SR)
    }, error = function(e) {
      cat("[ERROR] Resampling fehlgeschlagen:", fname, "\n")
      return(NULL)
    })
    
    if (is.null(wav)) next
  }
  
  # Stereo -> Mono
  if (wav@stereo) {
    wav <- mono(wav, "left")
  }
  
  # Rohsignal extrahieren
  y <- wav@left / (2^(wav@bit - 1))
  
  feat <- extract_features(y, SR)
  if (is.null(feat)) {
    cat("[WARN] Ungültige Features für:", fname, "\n")
    skipped <- skipped + 1
    next
  }
  
  rows[[length(rows) + 1]] <- feat
  
  # Speaker Label (Beispiel: erste 2 Zeichen)
  labels <- c(labels, substr(fname, 1, 2))
}

cat("[INFO] Erfolgreich:", length(rows), "übersprungen:", skipped, "\n")

if (length(rows) == 0) {
  stop("Keine Features extrahiert – bitte Pfade/Dateien prüfen.")
}

# DataFrame bauen
mat <- do.call(rbind, rows)
df <- as.data.frame(mat)
colnames(df) <- paste0("feat_", seq_len(ncol(mat)))
df$label <- labels

# CSV speichern
write_csv(df, OUTPUT_CSV)
cat("[OK] Features gespeichert in:", OUTPUT_CSV, "\n")
