#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def camera_publisher():
    rospy.init_node('camera_node')
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)  # Smaller queue
    cap = cv2.VideoCapture('/dev/video2')
    bridge = CvBridge()
    
    # Optimize camera settings for low latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    rate = rospy.Rate(30)  # 30 FPS
    
    if not cap.isOpened():
        rospy.logerr("❌ Could not open camera.")
        return
    
    rospy.loginfo("✅ Camera optimized for low latency")
    
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("⚠️ Frame not received.")
            continue
        
        # Create message with timestamp
        msg = bridge.cv2_to_imgmsg(frame, "bgr8")
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera"
        
        pub.publish(msg)
        rate.sleep()
    
    cap.release()

if __name__ == '__main__':
    try:
        camera_publisher()
    except rospy.ROSInterruptException:
        pass