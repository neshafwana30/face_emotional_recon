import cv2
import os
import re

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # ganti 0/1 sesuai webcam
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

person_id = 3 # id for person that we will detect
count = 0 # count for image name id

existing_files = [f for f in os.listdir(dataset_path) if f.startswith(f"person-{person_id}-")]


if existing_files:
    # Ambil nomor terakhir dari nama file
    numbers = [int(re.findall(r"person-\d+-(\d+)\.jpg", f)[0]) for f in existing_files]
    count = max(numbers)
    print(f"📁 Ditemukan {len(existing_files)} file sebelumnya. Melanjutkan dari {count+1}.")
else:
    print("📁 Belum ada file sebelumnya. Mulai dari 1.")


# --- Capture wajah ---
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        filename = f"{dataset_path}person-{person_id}-{count}.jpg"
        cv2.imwrite(filename, gray[y:y+h, x:x+w])
        print(f"📸 {filename} tersimpan")

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q') or count >= (len(existing_files) + 10):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Pengambilan selesai.")