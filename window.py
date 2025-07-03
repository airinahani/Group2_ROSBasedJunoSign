import cv2
from ultralytics import YOLO
import torch
import pyttsx3
import time
import subprocess
import platform
from collections import deque, Counter

# Text-to-speech that works on Windows with pyttsx3, on robot use espeak
def speak(text):
    if platform.system() == "Windows":
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    else:
        subprocess.run(["espeak", text])

def main():
    model_path = r"D:\\signing\\best.pt"
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam.")

    window_name = "HandSign Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    buffer = ""
    sentence = ""
    last_letter = ""
    letter_queue = deque(maxlen=5)

    sos_triggered = False
    sos_timestamp = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not received.")
                break

            result = model(frame, conf=0.8)[0]
            boxes = result.boxes
            annotated = result.plot()

            detected_letter = ""

            if boxes and boxes.cls.numel() > 0:
                class_id = int(boxes.cls[0].item())
                confidence = float(boxes.conf[0].item())

                if confidence >= 0.8:
                    detected_letter = model.names[class_id]
                    letter_queue.append(detected_letter)
                else:
                    letter_queue.append("")
            else:
                letter_queue.append("")

            if len(letter_queue) == letter_queue.maxlen:
                most_common = Counter(letter_queue).most_common(1)[0]
                if most_common[1] >= 3 and most_common[0] != last_letter and most_common[0] != "":
                    buffer += most_common[0]
                    last_letter = most_common[0]
                    print(f"✅ Confirmed: {most_common[0]} → Buffer: {buffer}")

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                if buffer:
                    sentence += buffer
                    speak(buffer)
                break

            elif key == 8:
                if buffer:
                    removed = buffer[-1]
                    buffer = buffer[:-1]
                    print(f"❌ Removed: {removed} → Buffer: {buffer}")

            elif key == ord(" "):
                if buffer:
                    sentence += buffer + " "
                    word = buffer.strip().upper()
                    print(f"📝 Word added: {buffer}")
                    speak(buffer)

                    # Emergency trigger for "SOS"
                    if word in ["SOS", "HELP", "EMERGENCY"]:
                        emergency_msg = "🚨 EMERGENCY DETECTED: Please call 999 or your nearest help center!"
                        print(emergency_msg)
                        speak("Emergency detected. Please call 999 or your nearest help center immediately.")
                        sos_triggered = True
                        sos_timestamp = time.time()

                    buffer = ""
                    last_letter = ""
                    letter_queue.clear()

            # Display buffer and sentence
            cv2.putText(annotated, f"Buffer: {buffer}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated, f"Sentence: {sentence.strip()}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show emergency message on screen for 5 seconds
            if sos_triggered and (time.time() - sos_timestamp < 5):
                cv2.putText(annotated, "EMERGENCY: CALL 999!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                sos_triggered = False

            cv2.imshow(window_name, annotated)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"📢 Final Sentence: {sentence.strip()}")
        speak(sentence.strip())
        print("✅ Camera released and windows closed.")

if __name__ == "__main__":
    torch_device = 0 if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(torch_device) if torch.cuda.is_available() else None
    main()

