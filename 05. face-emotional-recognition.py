import cv2
from fer import FER

# ===== Inisialisasi FER =====
detector = FER(mtcnn=False)  # ga pakai MTCNN, biar cepat dan stabil

# ===== Inisialisasi Face Recognizer =====
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("face-model.yml")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_COMPLEX
names = ['None', 'Shafwa', 'Auliya', 'Jobi']

# ===== Load emoji =====
emoji_map = {
    'happy': cv2.imread('emoji/happy.png', cv2.IMREAD_UNCHANGED),
    'sad': cv2.imread('emoji/sad.png', cv2.IMREAD_UNCHANGED),
    'angry': cv2.imread('emoji/angry.png', cv2.IMREAD_UNCHANGED),
    'fear': cv2.imread('emoji/fear.png', cv2.IMREAD_UNCHANGED),
    'surprise': cv2.imread('emoji/surprised.png', cv2.IMREAD_UNCHANGED)
}

def overlay_emoji(frame, emoji, x, y, size=70):
    """Tempel emoji PNG transparan di frame"""
    if emoji is None:
        return frame
    emoji = cv2.resize(emoji, (size, size))
    y1, y2 = y, y + emoji.shape[0]
    x1, x2 = x, x + emoji.shape[1]

    if y1 < 0 or x1 < 0 or y2 > frame.shape[0] or x2 > frame.shape[1]:
        return frame

    alpha_s = emoji[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        frame[y1:y2, x1:x2, c] = (alpha_s * emoji[:, :, c] +
                                  alpha_l * frame[y1:y2, x1:x2, c])
    return frame


# ===== Jalankan Webcam =====
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # lebar
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # tinggi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ===== Deteksi wajah hanya pakai HaarCascade =====
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        if roi_gray.size == 0:
            continue

        # ===== Face Recognition =====
        id_pred, confidence = recognizer.predict(roi_gray)
        if confidence < 100:
            id_text = names[id_pred] if id_pred < len(names) else "unknown"
        else:
            id_text = "unknown"

        # ===== Deteksi Emosi (dari ROI wajah yg sama) =====
        emotions_result = detector.detect_emotions(roi_color)
        if emotions_result:
            emotions = emotions_result[0]['emotions']
            dominant = max(emotions, key=emotions.get)
            score = emotions[dominant]
        else:
            dominant, score = "neutral", 0

        # ===== Gambar border & teks =====
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        label = f"{id_text} | {dominant} ({score:.2f})"
        cv2.putText(frame, label, (x, y - 10), font, 0.7, (255, 255, 255), 2)

        # ===== Tambah emoji =====
        emoji = emoji_map.get(dominant)
        frame = overlay_emoji(frame, emoji, x + w + 10, y, size=90)

    cv2.imshow("Face Recognition + Emotion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
