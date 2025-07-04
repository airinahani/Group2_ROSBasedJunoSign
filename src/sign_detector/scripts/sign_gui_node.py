#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import subprocess
import platform
import time
from threading import Lock

class SignGUI:
    def __init__(self):
        rospy.init_node("sign_gui_node")
        self.bridge = CvBridge()
        self.lock = Lock()
        self.buffer = ""  # This accumulates letters to form words
        self.sentence = ""  # This stores complete words/sentences
        self.sos_triggered = False
        self.sos_timestamp = 0
        self.latest_frame = None
        
        # Subscribers
        rospy.Subscriber("/sign_detector/image_processed", Image, self.image_callback)
        rospy.Subscriber("/sign_detector/buffer", String, self.buffer_callback)
        rospy.Subscriber("/sign_detector/sentence", String, self.sentence_callback)
        
        # Publishers
        self.tts_pub = rospy.Publisher("/sign_detector/sentence", String, queue_size=10)
        # Publisher to update buffer in detection node
        self.buffer_update_pub = rospy.Publisher("/sign_detector/buffer_update", String, queue_size=10)
        
        self.window_name = "🖐️ Sign Detection Viewer"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Print instructions
        rospy.loginfo("=== SIGN LANGUAGE DETECTION CONTROLS ===")
        rospy.loginfo("📝 Letters accumulate in buffer to form words")
        rospy.loginfo("SPACE: Complete word → Add to sentence + Speak word")
        rospy.loginfo("BACKSPACE: Remove last letter from current word")
        rospy.loginfo("R: Repeat/Read current sentence")
        rospy.loginfo("Q: Speak current word buffer + Quit")
        rospy.loginfo("=========================================")
    
    def speak(self, text):
        if not text.strip():
            return
        try:
            if platform.system() == "Windows":
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            else:
                subprocess.run(["espeak", text], check=False)
        except Exception as e:
            rospy.logwarn(f"TTS failed: {e}")
    
    def buffer_callback(self, msg):
        with self.lock:
            # Only update buffer if it's adding new letters (not clearing)
            new_buffer = msg.data
            if len(new_buffer) > len(self.buffer):
                self.buffer = new_buffer
                rospy.loginfo(f"📝 Building word: '{self.buffer}'")
    
    def sentence_callback(self, msg):
        with self.lock:
            self.sentence = msg.data
    
    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.latest_frame = frame
        except Exception as e:
            rospy.logwarn(f"Failed to convert image: {e}")
    
    def update_buffer_in_detection_node(self, new_buffer):
        """Send buffer update to detection node"""
        self.buffer_update_pub.publish(String(data=new_buffer))
    
    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                buffer = self.buffer
                sentence = self.sentence
                sos_triggered = self.sos_triggered
                sos_timestamp = self.sos_timestamp
            
            if frame is not None:
                # Overlay text with better visibility
                cv2.putText(frame, f"Current Word: {buffer}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Sentence: {sentence}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Instructions
                cv2.putText(frame, "SPACE=Complete Word | BACKSPACE=Delete Letter | R=Read Sentence | Q=Quit", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                if sos_triggered and (time.time() - sos_timestamp < 5):
                    cv2.putText(frame, "EMERGENCY: CALL 999!", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    with self.lock:
                        self.sos_triggered = False
                
                cv2.imshow(self.window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Speak current word before quitting
                if buffer:
                    self.speak(buffer)
                rospy.loginfo("👋 Quit pressed, exiting...")
                break
            elif key == 8:  # Backspace - remove last letter from current word
                with self.lock:
                    if self.buffer:
                        removed = self.buffer[-1]
                        self.buffer = self.buffer[:-1]
                        # Update the detection node's buffer
                        self.update_buffer_in_detection_node(self.buffer)
                        rospy.loginfo(f"❌ Removed letter: {removed} → Current word: '{self.buffer}'")
            elif key == ord('r'):
                # Read the current sentence
                if sentence:
                    self.speak(sentence)
                    rospy.loginfo(f"🔊 Reading sentence: '{sentence}'")
                else:
                    rospy.loginfo("📭 No sentence to read")
            elif key == ord(' '):  # Space - complete the current word
                with self.lock:
                    if self.buffer:
                        word = self.buffer.strip().upper()
                        
                        # Add word to sentence
                        self.sentence += self.buffer + " "
                        
                        # Speak the completed word
                        self.speak(self.buffer)
                        rospy.loginfo(f"✅ Word completed: '{self.buffer}' → Added to sentence")
                        
                        # Check for emergency words
                        if word in ["SOS", "HELP", "EMERGENCY"]:
                            rospy.logwarn("🚨 EMERGENCY DETECTED!")
                            self.speak("Emergency detected. Please call 999 or your nearest help center immediately.")
                            self.sos_triggered = True
                            self.sos_timestamp = time.time()
                        
                        # Clear buffer for next word
                        self.buffer = ""
                        # Clear the detection node's buffer too
                        self.update_buffer_in_detection_node(self.buffer)
                        
                        # Publish updated sentence
                        self.tts_pub.publish(self.sentence.strip())
                    else:
                        rospy.loginfo("📝 No word to complete (buffer empty)")
            
            rate.sleep()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gui = SignGUI()
    try:
        gui.run()
    except rospy.ROSInterruptException:
        pass