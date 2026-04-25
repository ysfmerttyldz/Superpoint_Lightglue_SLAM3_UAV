import torch
import sys
import os

# specarmi reposundan kopyaladığın model dosyasının yolunu belirt
# Eğer superpoint_net.py olarak kopyaladıysan:
from superpoint_net import SuperPointNet_gauss2

def convert():
    # 1. Model mimarisini oluştur
    model = SuperPointNet_gauss2()

    # 2. .pth ağırlıklarını yükle
    pth_path = "thermal_superpoint.pth" # Rar'dan çıkardığın dosya adı
    if not os.path.exists(pth_path):
        print(f"Hata: {pth_path} bulunamadı!")
        return

    state_dict = torch.load(pth_path, map_location="cpu")

    # Ağırlıkların 'model' veya 'state_dict' anahtarı altında olup olmadığını kontrol et
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    elif 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)

    model.eval()

    # 3. Tracing işlemi
    # config.yml'de belirtilen 320x256 boyutunu kullanıyoruz
    # (Batch: 1, Kanal: 1, Yükseklik: 256, Genişlik: 320)
    example_input = torch.randn(1, 1, 256, 320)

    print("Model TorchScript formatına dönüştürülüyor (Tracing)...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # 4. Kaydet
    output_name = "superpoint_thermal.pt"
    traced_model.save(output_name)
    print(f"Başarılı! C++ uyumlu model dosyası oluştu: {output_name}")

if __name__ == "__main__":
    convert()