import os
import shutil
from sklearn.model_selection import train_test_split

# Veri yolları
original_dataset_dir = "augmented_dataset"
base_dir = "prepared_dataset"

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Klasörleri oluştur
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# Veriyi bölme
for label in os.listdir(original_dataset_dir):
    label_dir = os.path.join(original_dataset_dir, label)
    images = os.listdir(label_dir)
    
    # Eğitim, doğrulama, test bölme
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.33, random_state=42)
    
    # Klasör oluştur ve veriyi kopyala
    for dataset, dataset_dir in zip([train_images, val_images, test_images], [train_dir, val_dir, test_dir]):
        dataset_label_dir = os.path.join(dataset_dir, label)
        os.makedirs(dataset_label_dir, exist_ok=True)
        for image in dataset:
            src = os.path.join(label_dir, image)
            dst = os.path.join(dataset_label_dir, image)
            shutil.copy(src, dst)

print("Veri seti hazır!")
