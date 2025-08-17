import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Eğitilmiş modeli yükleyin
model = load_model('sign_language_model.h5')

# Eğitim sırasında kullanılan sınıflar
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Eğitimde kullanılan harfler

# Kamerayı başlat
cap = cv2.VideoCapture(0)

print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")

while True:
    # Kamera görüntüsünü al
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    # Görüntüyü model giriş boyutuna göre yeniden boyutlandır
    img = cv2.resize(frame, (128, 128))  # Modelin beklediği boyut
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Model RGB görüntü bekliyorsa
    img = img.astype('float32') / 255.0  # Normalizasyon
    img = np.expand_dims(img, axis=0)  # Batch boyutunu ekle

    # Model ile tahmin yap
    predictions = model.predict(img, verbose=0)  # Tahmin değerlerini alın
    predicted_label = labels[np.argmax(predictions)]  # En yüksek olasılıklı sınıf

    # Kamera görüntüsüne tahmini yazdır
    cv2.putText(frame, f"Tahmin: {predicted_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Tahmin", frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
