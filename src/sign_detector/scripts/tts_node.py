#!/usr/bin/env python3

import rospy
import subprocess
from std_msgs.msg import String

def speak(text):
    try:
        subprocess.run(["espeak", text], check=True)
    except subprocess.CalledProcessError as e:
        rospy.logerr(f"espeak failed: {e}")
    except Exception as e:
        rospy.logerr(f"Unexpected error in TTS: {e}")

def callback(msg):
    sentence = msg.data.strip()
    if sentence:
        rospy.loginfo(f"🔊 Speaking: {sentence}")
        speak(sentence)

def listener():
    rospy.init_node('tts_node')
    rospy.Subscriber("/sign_detector/sentence", String, callback)
    rospy.loginfo("🗣️ TTS node is running...")
    rospy.spin()

if __name__ == '__main__':
    listener()
