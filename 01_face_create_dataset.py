import cv2
import os
import re

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

person_id = 2
count = 0

existing_files = [f for f in os.listdir(dataset_path) if f.startswith(f"person-{person_id}-")]
if existing_files:
    numbers = [int(re.findall(r"person-\d+-(\d+)\.jpg", f)[0]) for f in existing_files]
    count = max(numbers)
    print(f"📁 Ditemukan {len(existing_files)} file sebelumnya. Melanjutkan dari {count+1}.")
else:
    print("📁 Belum ada file sebelumnya. Mulai dari 1.")

frame_skip = 0  # untuk cooldown

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=7, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        # Pastikan wajah cukup besar
        if w < 100 or h < 100:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Simpan setiap 5 frame
        if frame_skip % 5 == 0:
            count += 1
            filename = f"{dataset_path}person-{person_id}-{count}.jpg"
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(filename, face_img)
            print(f"📸 {filename} tersimpan")

    frame_skip += 1
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q') or count >= (len(existing_files) + 100):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Pengambilan selesai.")