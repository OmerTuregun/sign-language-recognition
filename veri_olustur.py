# veri oluşturma dosyası

import cv2
import os

# Veri seti klasörünün yolu
output_folder = "dataset"
os.makedirs(output_folder, exist_ok=True)

# Harf için klasör oluşturma
label = input("Kaydedilecek harfi girin (örneğin, 'A', 'B', ...): ").upper()
label_folder = os.path.join(output_folder, label)
os.makedirs(label_folder, exist_ok=True)

# Kamera açma
cap = cv2.VideoCapture(0)
frame_count = 0
max_frames = 1000  # Her harf için kaydedilecek maksimum kare sayısı

print(f"Kamera açıldı. '{label}' işaretini yaparak verileri kaydetmeye başlayabilirsiniz.")
print(f"Kayıt otomatik olarak {max_frames} kare kaydedildikten sonra duracaktır.")

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break

    # Görüntüyü göster
    cv2.imshow("Video", frame)

    # Her kareyi kaydet
    frame_file = os.path.join(label_folder, f"{label}_{frame_count:04d}.jpg")
    cv2.imwrite(frame_file, frame)
    frame_count += 1

    # 'q' tuşuna basarak manuel durdurma seçeneği
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Manuel olarak kayıt durduruldu.")
        break

cap.release()
cv2.destroyAllWindows()

print(f"Kayıt tamamlandı. Toplam {frame_count} kare '{label}' klasörüne kaydedildi.")
