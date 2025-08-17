# veri arttırıp zenginleştirme fonksiyonu

import cv2
import os
from albumentations import (
    HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, GaussianBlur, Compose
)

# Veri zenginleştirme fonksiyonu
def augment_image(image):
    augmentations = Compose([
        HorizontalFlip(p=0.5),  # Görüntüyü yatay çevirme
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),  # Döndürme ve ölçekleme
        RandomBrightnessContrast(p=0.5),  # Parlaklık ve kontrast değişikliği
        GaussianBlur(p=0.3)  # Bulanıklık ekleme
    ])
    augmented = augmentations(image=image)
    return augmented["image"]

# Verilerin bulunduğu klasör
data_folder = "dataset"
augmented_folder = "augmented_dataset"
os.makedirs(augmented_folder, exist_ok=True)

# Her harf için zenginleştirme işlemi
for label in os.listdir(data_folder):
    label_path = os.path.join(data_folder, label)
    augmented_label_path = os.path.join(augmented_folder, label)
    os.makedirs(augmented_label_path, exist_ok=True)

    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        image = cv2.imread(img_path)

        # Zenginleştirilmiş görüntüleri kaydet
        for i in range(5):  # Her görüntüden 5 zenginleştirilmiş görüntü üret
            augmented_image = augment_image(image=image)
            aug_img_path = os.path.join(augmented_label_path, f"aug_{i}_{img_file}")
            cv2.imwrite(aug_img_path, augmented_image)

print(f"Zenginleştirme tamamlandı. Veriler '{augmented_folder}' klasörüne kaydedildi.")
