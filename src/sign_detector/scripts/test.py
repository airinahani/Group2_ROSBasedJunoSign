import cv2

cap = cv2.VideoCapture('/dev/video2')  # or /dev/video0 if that’s your camera

if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received")
        break

    cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
