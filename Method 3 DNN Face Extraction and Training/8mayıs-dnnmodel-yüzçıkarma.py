import os
import cv2
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

# DNN yüz algılama modeli
modelFile = r"C:\Users\Admin\Desktop\TEZ\Deepfake\kodlar\res10_300x300_ssd_iter_140000.caffemodel"
configFile = r"C:\Users\Admin\Desktop\TEZ\Deepfake\kodlar\deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
if net.empty():
    print("Hata: DNN modeli yüklenemedi. Model dosyalarını kontrol et.")
    exit()

# Transform: Görüntüyü 150x150 boyutuna getir
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

# Celeb-DF veri seti yolları
input_paths = {
    'train_fake': r"C:\Users\Admin\Desktop\TEZ\Deepfake\DeepFake-train-test\_faces\train-fake-frame",
    'train_real': r"C:\Users\Admin\Desktop\TEZ\Deepfake\DeepFake-train-test\_faces\train-real-frame",
    'test_fake': r"C:\Users\Admin\Desktop\TEZ\Deepfake\DeepFake-train-test\_faces\test-fake-frame",
    'test_real': r"C:\Users\Admin\Desktop\TEZ\Deepfake\DeepFake-train-test\_faces\test-real-frame"
}

output_paths = {
    'train_fake': r"C:\Users\Admin\Desktop\TEZ\Deepfake\DeepFake-train-test\_faces_preprocessed_dnn\train-fake-frame",
    'train_real': r"C:\Users\Admin\Desktop\TEZ\Deepfake\DeepFake-train-test\_faces_preprocessed_dnn\train-real-frame",
    'test_fake': r"C:\Users\Admin\Desktop\TEZ\Deepfake\DeepFake-train-test\_faces_preprocessed_dnn\test-fake-frame",
    'test_real': r"C:\Users\Admin\Desktop\TEZ\Deepfake\DeepFake-train-test\_faces_preprocessed_dnn\test-real-frame"
}

# Çıktı dizinlerini oluştur
for key in output_paths:
    os.makedirs(output_paths[key], exist_ok=True)

# Görüntüleri işleme fonksiyonu
def process_images(input_path, output_path):
    image_files = [f for f in os.listdir(input_path) if f.endswith('.jpg')]
    for img_name in tqdm(image_files, desc=f"İşleniyor: {os.path.basename(input_path)}"):
        img_path = os.path.join(input_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Görüntü yüklenemedi: {img_path}")
            continue

        # Görüntüyü blob’a çevir
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # İlk yüzü al (en yüksek güven skoru olan)
        if len(detections) > 0:
            confidence = detections[0, 0, 0, 2]
            if confidence > 0.5:  # Güven skoru eşiği
                box = detections[0, 0, 0, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (x, y, x2, y2) = box.astype(int)
                w = x2 - x
                h = y2 - y

                # Yüzü kırp (BGR formatında)
                x = max(0, x)
                y = max(0, y)
                face = image[y:y+h, x:x+w] if x + w <= image.shape[1] and y + h <= image.shape[0] else image

                # Yüzü RGB’ye çevir (OpenCV BGR -> RGB)
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # Tensor dönüşümünü uygula (RGB ile)
                face_tensor = transform(face_rgb)
                face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Çıktıyı BGR’ye çevirip kaydet (OpenCV uyumlu)
                face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                output_img_path = os.path.join(output_path, img_name)
                cv2.imwrite(output_img_path, face_bgr)
            else:
                print(f"Yüz algılanamadı (düşük güven): {img_path}")
        else:
            print(f"Yüz algılanamadı: {img_path}")
            # Yüz algılanamazsa orijinal görüntüyü kaydet (150x150'e yeniden boyutlandır)
            resized_img = cv2.resize(image, (150, 150))
            output_img_path = os.path.join(output_path, img_name)
            cv2.imwrite(output_img_path, resized_img)  # BGR olarak zaten uyumlu

# Tüm veri setini işle
for key in input_paths:
    print(f"İşleniyor: {key}")
    process_images(input_paths[key], output_paths[key])

print("Celeb-DF görüntüleri OpenCV DNN ile işlendi ve kaydedildi.")