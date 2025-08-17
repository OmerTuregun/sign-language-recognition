from transformers import TFViTForImageClassification, ViTImageProcessor
import tensorflow as tf
from pathlib import Path

# Veri seti dizinlerini belirtin
train_dir = Path(r"C:\Users\trkur\isaret_dili\prepared_dataset\train")
val_dir = Path(r"C:\Users\trkur\isaret_dili\prepared_dataset\val")

# Dizinin varlığını kontrol et
if not train_dir.exists():
    raise FileNotFoundError(f"Eğitim veri seti yolu bulunamadı: {train_dir}")
if not val_dir.exists():
    raise FileNotFoundError(f"Doğrulama veri seti yolu bulunamadı: {val_dir}")

# Model ve feature extractor yükleme
print("Model ve feature extractor yükleniyor...")
model = TFViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=26,  # 26 harf (A'dan Z'ye)
    id2label={i: chr(65 + i) for i in range(26)},
    label2id={chr(65 + i): i for i in range(26)},
)

# Modelin yapılandırmasını kontrol et ve güncelle
print("Model yapılandırması kontrol ediliyor...")
config = model.config
config.image_size = 224  # Giriş boyutları
config.num_channels = 3  # RGB kanalları
model.config = config

# Feature extractor yükleme
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
print("Model ve feature extractor başarıyla yüklendi.")

# Veri setlerini TensorFlow dataset'e dönüştürme
print("Veri setleri oluşturuluyor...")
batch_size = 32
image_size = (224, 224)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    str(train_dir),
    labels="inferred",
    label_mode="int",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    str(val_dir),
    labels="inferred",
    label_mode="int",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False,
)

# Veri kümelerini ön işleme
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_image(image, label):
    """
    Görüntüyü normalize eder ve etiketiyle birlikte döner.
    """
    image = tf.image.resize(image, (224, 224))  # Boyutlandır
    image = tf.image.per_image_standardization(image)  # Normalize
    return image, label

train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("Veri setleri başarıyla oluşturuldu.")

# Giriş boyutlarını kontrol et ve model ile doğrula
for image_batch, label_batch in train_dataset.take(1):
    print("Model giriş boyutu:", image_batch.shape)
    try:
        model_output = model(image_batch)  # Girişlerin modelle uyumlu olup olmadığını test et
        print("Model giriş testi başarılı.")
    except Exception as e:
        print("Model giriş testi başarısız:", e)

# GPU belleği yönetimi (opsiyonel)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU bellek yönetimi etkinleştirildi.")
    except RuntimeError as e:
        print(f"GPU bellek yönetimi hatası: {e}")
else:
    print("GPU kullanılmıyor, işlemler CPU'da gerçekleşecek.")

# Modeli derleme
print("Model derleniyor...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
print("Model başarıyla derlendi.")

# Modeli eğitme
print("Eğitim başlıyor...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    verbose=1,  # Eğitim sürecini detaylı göster
)
print("Eğitim tamamlandı.")

# Modeli kaydetme
print("Model kaydediliyor...")
model.save_pretrained("isaret_dili_transformer_modeli")
print("Model başarıyla kaydedildi.")
