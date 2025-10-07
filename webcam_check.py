import cv2

camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # ubah 1 ke angka 0 - 3 buat cek kamera mana yang aktif

while True:
    _, frame = camera.read()
    
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()