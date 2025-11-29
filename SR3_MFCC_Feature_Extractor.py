# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 20:49:18 2025

@author: MaxWe
"""

# SR3_MFCC_Feature_Extractor_fixed.py
# -*- coding: utf-8 -*-
import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ==== Einstellungen ====
AUDIO_DIR = r"D:\STUDIUM\Münster\VPRonaRaspberryPi\EmoDB\wav"
OUTPUT_CSV = r"D:\ComicandSonsProductions\GameJam1\SpeakerRecognition\features_fixed.csv"
FILE_LIST_CSV = r"D:\STUDIUM\Münster\VPRonaRaspberryPi\EmoDB\wav\file_list.csv"

SR = 16000
N_MFCC = 20
HOP_LENGTH = 512
N_FFT = 2048
MIN_FRAMES = 20
MAX_FRAMES = 200

def load_file_list(file_list_csv):
    files = []
    if os.path.isfile(file_list_csv):
        with open(file_list_csv, newline='', encoding='utf-8', errors='replace') as f:
            for row in f:
                r = row.strip()
                if not r:
                    continue
                # falls CSV mit Komma-Separiertem Format: nimm erstes Feld
                if ',' in r:
                    r = r.split(',')[0].strip().strip('"').strip("'")
                files.append(r)
    return files

def extract_aggregated_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack([mfcc, d1, d2])
    n_frames = stacked.shape[1]
    if n_frames < MIN_FRAMES:
        padded = np.zeros((stacked.shape[0], MAX_FRAMES))
        padded[:, :n_frames] = stacked
        stacked = padded
    elif n_frames > MAX_FRAMES:
        stacked = stacked[:, :MAX_FRAMES]
    mean = np.mean(stacked, axis=1)
    std = np.std(stacked, axis=1)
    return np.concatenate([mean, std])

# === Datei-Liste laden ===
file_names = load_file_list(FILE_LIST_CSV)
print(f"[INFO] file_list.csv enthielt {len(file_names)} Einträge.")

# Fallback: wenn CSV leer, suche WAV/MP3 im Verzeichnis
if len(file_names) == 0:
    print("[INFO] Keine Liste gefunden oder CSV leer. Durchsuche AUDIO_DIR mit glob als Fallback.")
    p = Path(AUDIO_DIR)
    exts = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a")
    for ext in exts:
        for f in p.glob(ext):
            file_names.append(str(f.name))
    print(f"[INFO] Gefundene Dateien per glob: {len(file_names)}")

rows = []
labels = []
skipped = 0
for idx, fname in enumerate(file_names, start=1):
    full_path = os.path.join(AUDIO_DIR, fname)
    if not os.path.isfile(full_path):
        print(f"[WARN] Datei nicht gefunden (übersprungen): {full_path}")
        skipped += 1
        continue
    print(f"[{idx}/{len(file_names)}] Verarbeite: {fname}")
    try:
        y, sr = librosa.load(full_path, sr=SR, mono=True)
    except Exception as e:
        print(f"[ERROR] Fehler beim Laden von {fname}: {e}")
        skipped += 1
        continue
    feat = extract_aggregated_features(y, sr)
    if feat is None or np.isnan(feat).any():
        print(f"[WARN] ungültige Features für {fname} -> übersprungen")
        skipped += 1
        continue
    rows.append(feat)
    # Label-Extraktion: passe je nach Dateinamen-Schema an
    speaker_label = os.path.basename(fname)[:2]
    labels.append(speaker_label)

print(f"[INFO] Erfolgreich verarbeitete Dateien: {len(rows)}, übersprungen: {skipped}")

if len(rows) == 0:
    print("[ERROR] Keine Features extrahiert. Bitte prüfe die Pfade, Dateiformate oder die Datei-Liste.")
else:
    # DataFrame erstellen
    n_features = len(rows[0])
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(rows, columns=cols)
    df['label'] = labels

    # Skalierung
    feat_cols = cols
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Features gespeichert in: {OUTPUT_CSV}")

