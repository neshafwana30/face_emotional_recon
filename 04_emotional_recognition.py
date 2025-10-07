import cv2
from fer import FER

detector = FER(mtcnn=True)  # bisa pakai MTCNN untuk deteksi wajah lebih akurat

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FER detect emosi
    # detector.detect_emotions return list, bisa langsung analisis frame
    result = detector.detect_emotions(frame)
    # result bentuk: [{'box': [x, y, w, h], 'emotions': {'happy': 0.9, 'sad':0.05, ...}}]

    for face in result:
        (x, y, w, h) = face['box']
        emotions = face['emotions']
        # cari emosi dominan
        dominant = max(emotions, key=emotions.get)
        score = emotions[dominant]

        # gambarkan kotak & teks
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{dominant} ({score:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Emotion (FER)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
