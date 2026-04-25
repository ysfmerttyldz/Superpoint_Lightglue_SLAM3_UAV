#!/usr/bin/env python3
"""
Video -> SP_SLAM3 Dataset Dönüştürücü (Termal Preprocessing)
============================================================
Thegra makalesinden esinlenilmiş ama sensöre göre ayarlanmış preprocessing.

Önemli: Her termal sensör farklı! Chambolle agresif olursa detayları yok eder.
         weight=0.0 ile Chambolle devre dışı bırakılabilir.
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


def apply_chambolle_tv_denoising(image, weight=1.0):
    """
    Chambolle Total Variation denoising.

    UYARI: weight değeri kritik!
        weight=4.0 (Thegra default): Çok agresif, detayları yok eder
        weight=1.0-2.0: Daha güvenli, detayları korur
        weight=0.0: Chambolle devre dışı (sadece CLAHE+Median)
    """
    if weight <= 0.0:
        return image

    if not SKIMAGE_AVAILABLE:
        info("Chambolle yerine Bilateral Filter fallback kullanılıyor...")
        return cv2.bilateralFilter(image, 9, 35, 35)

    # uint8 -> float [0, 1]
    img_float = img_as_float(image)

    # Chambolle TV denoising
    denoised = denoise_tv_chambolle(img_float, weight=weight, channel_axis=None)

    # float -> uint8
    return img_as_ubyte(denoised)


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    CLAHE: Contrast Limited Adaptive Histogram Equalization.
    Thegra'da standart Hist Eq önerilmiş ama bazı sensörlerde CLAHE daha iyi.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def apply_histogram_equalization(image):
    """Standart histogram equalization."""
    return cv2.equalizeHist(image)


def apply_median_filter(image, kernel_size=3):
    """Median filtering - salt & pepper gürültüsünü temizler."""
    return cv2.medianBlur(image, kernel_size)


def apply_thermal_preprocessing(image, 
                                 use_chambolle=True,
                                 chambolle_weight=1.0,
                                 use_clahe=True,
                                 clahe_clip=2.0,
                                 median_kernel=3):
    """
    Termal preprocessing pipeline - sensöre göre ayarlanabilir.

    Önerilen ayarlar:
    -----------------
    1. Gürültülü/düşük kaliteli sensör (Thegra tarzı):
       --chambolle-weight 2.0 --clahe-clip 2.0

    2. Temiz sensör, düşük kontrast (senin görüntün gibi):
       --chambolle-weight 0.0 --clahe-clip 2.0
       (Chambolle devre dışı, sadece CLAHE+Median)

    3. Çok gürültülü, NUC artefaktlı:
       --chambolle-weight 4.0 --clahe-clip 2.0
       (Thegra default ama dikkatli kullan!)
    """
    # Görüntü renkliyse griye çevir
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. Chambolle TV Denoising (opsiyonel)
    if use_chambolle and chambolle_weight > 0:
        gray = apply_chambolle_tv_denoising(gray, weight=chambolle_weight)

    # 2. Kontrast artırma (CLAHE veya standart Hist Eq)
    if use_clahe:
        gray = apply_clahe(gray, clip_limit=clahe_clip)
    else:
        gray = apply_histogram_equalization(gray)

    # 3. Median Filtering (salt & pepper)
    gray = apply_median_filter(gray, kernel_size=median_kernel)

    return gray


def main():
    parser = argparse.ArgumentParser(
        description='Video -> SP_SLAM3 Dataset (Termal Preprocessing)'
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

    # Preprocessing anahtarları
    parser.add_argument('--thermal', action='store_true', 
                        help='Termal preprocessing uygula')
    parser.add_argument('--no-chambolle', action='store_true',
                        help='Chambolle TV denoising DEVRE DIŞI (önerilen: temiz sensörler için)')
    parser.add_argument('--no-clahe', action='store_true',
                        help='CLAHE yerine standart Histogram Equalization kullan')

    # Preprocessing parametreleri
    parser.add_argument('--chambolle-weight', type=float, default=1.0, 
                        help='Chambolle TV weight (0.0=kapalı, 1.0=varsayılan, 4.0=Thegra)')
    parser.add_argument('--clahe-clip', type=float, default=2.0, 
                        help='CLAHE clip limit (2.0-5.0 arası)')
    parser.add_argument('--median-kernel', type=int, default=3, 
                        help='Median filter kernel boyutu (3 veya 5)')

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

    # Chambolle kontrolü
    use_chambolle = not args.no_chambolle and args.chambolle_weight > 0
    use_clahe = not args.no_clahe

    info(f"=== Termal Preprocessing Pipeline ===")
    info(f"Video: {video_path}")
    info(f"Kaynak FPS: {src_fps:.2f}, Hedef FPS: {target_fps}")
    info(f"Toplam kare: {total_frames}")
    info(f"Termal Mod: {'AÇIK' if args.thermal else 'KAPALI'}")

    if args.thermal:
        info(f"  -> Chambolle TV: {'AÇIK (weight=' + str(args.chambolle_weight) + ')' if use_chambolle else 'KAPALI'}")
        info(f"  -> Kontrast: {'CLAHE (clip=' + str(args.clahe_clip) + ')' if use_clahe else 'Histogram Equalization'}")
        info(f"  -> Median kernel: {args.median_kernel}")

        if args.chambolle_weight >= 4.0:
            info(f"  -> UYARI: Chambolle weight={args.chambolle_weight} çok agresif!")
            info(f"     Detay kaybı yaşayabilirsin. Önerilen: 0.0-2.0")

        if not SKIMAGE_AVAILABLE and use_chambolle:
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

            # --- Termal Preprocessing Uygula ---
            if args.thermal:
                frame = apply_thermal_preprocessing(
                    frame,
                    use_chambolle=use_chambolle,
                    chambolle_weight=args.chambolle_weight,
                    use_clahe=use_clahe,
                    clahe_clip=args.clahe_clip,
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

    if args.thermal and use_chambolle and args.chambolle_weight >= 4.0:
        info(f"UYARI: Chambolle weight={args.chambolle_weight} kullandın.")
        info(f"       Eğer SLAM'da zayıf keypoint varsa --chambolle-weight 0.0 veya 1.0 dene.")


if __name__ == '__main__':
    main()