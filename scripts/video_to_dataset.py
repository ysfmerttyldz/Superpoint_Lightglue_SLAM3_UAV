#!/usr/bin/env python3
"""
Video -> SP_SLAM3 Dataset Dönüştürücü (Termal Destekli)
======================================================
"""

import cv2
import os
import sys
import argparse
import time
import numpy as np

def debug(msg):
    print(f"[DEBUG] {msg}")

def info(msg):
    print(f"[INFO] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")

def apply_thermal_preprocessing(image):
    """
    Termal görüntü iyileştirme: CLAHE + Median Blur + Detail Enhancement (DDE)
    """
    # Görüntü renkliyse griye çevir (Termal SLAM genelde tek kanal çalışır)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Kontrastı yerel olarak artırır, parlamayı engeller
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # 2. Median Blur
    # Termal sensörlerdeki 'salt and pepper' gürültüsünü temizler
    blur_img = cv2.medianBlur(clahe_img, 3)

    return blur_img

def main():
    parser = argparse.ArgumentParser(description='Video -> SP_SLAM3 Dataset')
    parser.add_argument('video', help='Video dosyası yolu')
    parser.add_argument('--fps', type=float, required=True, help='Çıkış FPS (ör: 7.5, 15, 30)')
    parser.add_argument('--output', default='Datasets/images9', help='Çıkış klasörü')
    parser.add_argument('--width', type=int, default=640, help='Çıkış genişlik')
    parser.add_argument('--height', type=int, default=360, help='Çıkış yükseklik')
    parser.add_argument('--max-frames', type=int, default=0, help='Maksimum kare sayısı')
    parser.add_argument('--start-sec', type=float, default=0, help='Başlangıç saniyesi')
    # Yeni eklenen termal mod parametresi
    parser.add_argument('--thermal', action='store_true', help='Termal preprocessing uygula (CLAHE+Median+DDE)')
    
    args = parser.parse_args()

    video_path = args.video
    target_fps = args.fps
    output_base = args.output
    out_w = args.width
    out_h = args.height

    if not os.path.exists(video_path):
        error(f"Video bulunamadı: {video_path}")
        sys.exit(1)

    data_dir = os.path.join(output_base, "mav0", "cam0", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        error(f"Video açılamadı: {video_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    info(f"İşlem başlıyor... Termal Mod: {'AÇIK' if args.thermal else 'KAPALI'}")

    frame_interval = src_fps / target_fps
    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_sec * src_fps))

    timestamps = []
    out_idx = 0
    src_idx = 0
    next_grab = 0.0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if src_idx >= next_grab:
            # Önce boyutlandır
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # --- Termal Preprocessing Uygula ---
            if args.thermal:
                frame = apply_thermal_preprocessing(frame)

            # Kaydet
            out_path = os.path.join(data_dir, f"{out_idx}.png")
            cv2.imwrite(out_path, frame)

            timestamps.append(out_idx)
            if out_idx % 100 == 0:
                debug(f"Kare {out_idx} işlendi...")

            out_idx += 1
            next_grab += frame_interval

            if args.max_frames > 0 and out_idx >= args.max_frames:
                break

        src_idx += 1

    cap.release()
    
    # Timestamp kaydı
    ts_path = os.path.join(output_base, "timestamp.txt")
    with open(ts_path, 'w') as f:
        for ts in timestamps:
            f.write(f"{ts}\n")

    info(f"Bitti! {out_idx} kare {output_base} klasörüne kaydedildi.")

if __name__ == '__main__':
    main()