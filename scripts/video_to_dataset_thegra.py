#!/usr/bin/env python3
"""
Video -> SP_SLAM3 Dataset Dönüştürücü (Thegra Preprocessing)
============================================================
Thegra: Graph-based SLAM for Thermal Imagery (arXiv:2602.08531)
En iyi preprocessing pipeline:
    Chambolle TV Denoising (weight=4) -> Histogram Equalization -> Median Filter (3x3)
"""

import cv2
import os
import sys
import argparse
import time
import numpy as np

try:
    from skimage.restoration import denoise_tv_chambolle
    from skimage.util import img_as_float, img_as_ubyte
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[WARNING] scikit-image kurulu değil. Chambolle TV denoising çalışmayacak.")
    print("          Kurulum: pip install scikit-image")


def debug(msg):
    print(f"[DEBUG] {msg}")

def info(msg):
    print(f"[INFO] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")


def apply_chambolle_tv_denoising(image, weight=4.0):
    """
    Chambolle Total Variation denoising.
    Thegra makalesine göre: weight=4.0 en iyi sonucu veriyor.
    Kenarları korurken gürültüyü bastırır.
    """
    if not SKIMAGE_AVAILABLE:
        # Fallback: Bilateral filter (Thegra'da 2. sırada)
        info("Chambolle yerine Bilateral Filter fallback kullanılıyor...")
        return cv2.bilateralFilter(image, 9, 35, 35)

    # uint8 -> float [0, 1]
    img_float = img_as_float(image)

    # Chambolle TV denoising
    denoised = denoise_tv_chambolle(img_float, weight=weight, channel_axis=None)

    # float -> uint8
    return img_as_ubyte(denoised)


def apply_histogram_equalization(image, clip_limit=None):
    """
    Thegra makalesine göre STANDART Histogram Equalization CLAHE'den daha iyi.
    Cumulative pixel count threshold = 10000 (makalede belirtilen).
    """
    # Standart histogram equalization
    return cv2.equalizeHist(image)


def apply_median_filter(image, kernel_size=3):
    """
    Median filtering - salt & pepper gürültüsünü temizler.
    Thegra: kernel_size = 3
    """
    return cv2.medianBlur(image, kernel_size)


def apply_thegra_preprocessing(image, chambolle_weight=4.0, median_kernel=3):
    """
    Thegra (arXiv:2602.08531) en iyi preprocessing pipeline:

    1. Chambolle TV Denoising (weight=4.0)
       - Kenarları korur, gürültüyü bastırır

    2. Histogram Equalization (standart, CLAHE değil!)
       - Thegra'da CLAHE'den daha iyi performans gösteriyor
       - Global kontrast artırımı

    3. Median Filtering (kernel=3)
       - Salt & pepper gürültüsünü temizler
       - Kontrast artırma SONRASI uygulanınca daha etkili

    NOT: Eski pipeline (CLAHE -> Median -> DDE) YERİNE bu kullanılmalı.
         DDE/Unsharp masking gürültüyü artırır, SLAM için zararlı.
    """
    # Görüntü renkliyse griye çevir
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. Chambolle TV Denoising (Thegra Fig. 4: en iyi sonuç)
    gray = apply_chambolle_tv_denoising(gray, weight=chambolle_weight)

    # 2. Histogram Equalization (Thegra Fig. 3: CLAHE'den iyi)
    gray = apply_histogram_equalization(gray)

    # 3. Median Filtering (Thegra: kontrast sonrası uygulanınca daha iyi)
    gray = apply_median_filter(gray, kernel_size=median_kernel)

    return gray


def main():
    parser = argparse.ArgumentParser(
        description='Video -> SP_SLAM3 Dataset (Thegra Preprocessing)'
    )
    parser.add_argument('video', help='Video dosyası yolu')
    parser.add_argument('--fps', type=float, required=True, 
                        help='Çıkış FPS (ör: 7.5, 15, 30)')
    parser.add_argument('--output', default='Datasets/images9', 
                        help='Çıkış klasörü')
    parser.add_argument('--width', type=int, default=640, 
                        help='Çıkış genişlik')
    parser.add_argument('--height', type=int, default=360, 
                        help='Çıkış yükseklik')
    parser.add_argument('--max-frames', type=int, default=0, 
                        help='Maksimum kare sayısı')
    parser.add_argument('--start-sec', type=float, default=0, 
                        help='Başlangıç saniyesi')

    # Thegra preprocessing parametreleri
    parser.add_argument('--thermal', action='store_true', 
                        help='Thegra preprocessing uygula (Chambolle+HistEq+Median)')
    parser.add_argument('--chambolle-weight', type=float, default=4.0, 
                        help='Chambolle TV denoising weight (Thegra default: 4.0)')
    parser.add_argument('--median-kernel', type=int, default=3, 
                        help='Median filter kernel boyutu (Thegra default: 3)')
    parser.add_argument('--legacy-pipeline', action='store_true', 
                        help='Eski pipeline kullan (CLAHE+Median+DDE) - ÖNERİLMEZ')

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    info(f"=== Thegra Preprocessing Pipeline ===")
    info(f"Video: {video_path}")
    info(f"Kaynak FPS: {src_fps:.2f}, Hedef FPS: {target_fps}")
    info(f"Toplam kare: {total_frames}")
    info(f"Termal Mod: {'AÇIK (Thegra)' if args.thermal else 'KAPALI'}")

    if args.thermal:
        info(f"  -> Chambolle TV weight: {args.chambolle_weight}")
        info(f"  -> Median kernel: {args.median_kernel}")
        info(f"  -> Histogram Equalization: STANDART (CLAHE değil)")
        if args.legacy_pipeline:
            info(f"  -> UYARI: Eski pipeline (CLAHE+DDE) kullanılıyor - Thegra önermiyor!")
        if not SKIMAGE_AVAILABLE:
            info(f"  -> UYARI: scikit-image yok, Bilateral Filter fallback kullanılıyor")

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

            # --- Thegra Preprocessing Uygula ---
            if args.thermal:
                if args.legacy_pipeline:
                    # Eski pipeline (karşılaştırma için)
                    frame = apply_legacy_preprocessing(frame)
                else:
                    # Thegra pipeline (önerilen)
                    frame = apply_thegra_preprocessing(
                        frame, 
                        chambolle_weight=args.chambolle_weight,
                        median_kernel=args.median_kernel
                    )

            # Kaydet
            out_path = os.path.join(data_dir, f"{out_idx}.png")
            cv2.imwrite(out_path, frame)

            timestamps.append(out_idx)
            if out_idx % 100 == 0:
                elapsed = time.time() - t_start
                fps_proc = out_idx / elapsed if elapsed > 0 else 0
                debug(f"Kare {out_idx}/{total_frames} işlendi... ({fps_proc:.1f} FPS)")

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

    elapsed_total = time.time() - t_start
    info(f"Bitti! {out_idx} kare {output_base} klasörüne kaydedildi.")
    info(f"Toplam süre: {elapsed_total:.1f}s, Ortalama: {out_idx/elapsed_total:.1f} FPS")


def apply_legacy_preprocessing(image):
    """Eski pipeline - karşılaştırma amaçlı, Thegra önermiyor."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    gaussian_blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, gaussian_blur, -0.5, 0)
    return gray


if __name__ == '__main__':
    main()
