import os
import time
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tqdm import tqdm  # İlerleme çubuğu için tqdm kütüphanesini ekliyoruz

# 📁 Yüzlerin çıkarıldığı klasörler
face_output_root = r"C:\Users\Admin\Desktop\TEZ\Deepfake\DeepFake-train-test\_faces"
train_real_faces = os.path.join(face_output_root, "train-real-frame")
train_fake_faces = os.path.join(face_output_root, "train-fake-frame")
test_real_faces = os.path.join(face_output_root, "test-real-frame")
test_fake_faces = os.path.join(face_output_root, "test-fake-frame")

# 📦 Dataset sınıfı
class FrameDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        self.samples = [(img, 0) for img in self.real_images] + [(img, 1) for img in self.fake_images]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 🎨 Görüntü dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def train_and_evaluate(model_name):
    print(f"\n🧠 Model başlatılıyor: {model_name}")
    
    # GPU kontrolü
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA kullanılamıyor, lütfen GPU destekli bir cihaz kullanın.")
    device = torch.device("cuda")
    print(f"📦 Kullanılan cihaz: {device}")

    # ✅ Model seçimi
    if model_name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        raise ValueError("Geçersiz model ismi")

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    EPOCHS = 20
    BATCH_SIZE = 32
    PATIENCE = 5  # Early stopping için sabır sayısı
    best_loss = float('inf')
    patience_counter = 0

    # Dataset ve Dataloader
    train_dataset = FrameDataset(train_real_faces, train_fake_faces, transform=transform)
    test_dataset = FrameDataset(test_real_faces, test_fake_faces, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"🚀 {model_name} eğitimi başlıyor...")
    start_train = time.time()
    model.train()
    for epoch in range(EPOCHS):
        print(f"\n📘 Epoch {epoch+1} başlıyor...")
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # tqdm ile ilerleme çubuğu ekliyoruz
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True):
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy() > 0.5
            all_preds.extend(preds.astype(int).flatten())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        print(f"✅ Epoch {epoch+1} tamamlandı - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

        # Early stopping kontrolü
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # En iyi modeli kaydet
            torch.save(model.state_dict(), f"{model_name}_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping: {epoch+1}. epoch'ta durduruldu.")
                break

    end_train = time.time()

    # 🔐 Final modeli kaydet (.pth formatında)
    model_save_path = f"{model_name}_deepfake_model_face.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Final model kaydedildi: {model_save_path}")

    # Test
    print(f"\n🔬 {model_name} test işlemi başlıyor...")
    start_test = time.time()
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test", leave=True):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.extend(preds.astype(int).flatten())
            all_labels.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"]))
    end_test = time.time()

    print(f"⏱ Eğitim süresi: {end_train - start_train:.2f} saniye")
    print(f"⏱ Test süresi: {end_test - start_test:.2f} saniye")

# 🔧 Ana çalıştırma
if __name__ == "__main__":
    for model_name in ["mobilenetv2", "efficientnetb0", "resnet18"]:
        train_and_evaluate(model_name)