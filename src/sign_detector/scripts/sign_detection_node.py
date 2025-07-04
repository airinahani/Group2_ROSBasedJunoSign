#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO
import numpy as np
import time
from threading import Thread, Lock
import queue

class SignDetectionNode:
    def __init__(self):
        rospy.init_node('sign_detection_node')
        
        # Get parameters
        model_path = rospy.get_param('~model_path', 'best.pt')
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.8)
        
        # Load YOLO model with optimizations
        try:
            self.model = YOLO(model_path)
            
            # Model optimizations
            if torch.cuda.is_available():
                self.model.to('cuda')
                rospy.loginfo("🚀 Using GPU acceleration")
            else:
                rospy.loginfo("🖥️ Using CPU (consider GPU for better performance)")
            
            # Set model to evaluation mode for faster inference
            self.model.model.eval()
            
            rospy.loginfo(f"✅ Model loaded: {model_path}")
        except Exception as e:
            rospy.logerr(f"❌ Failed to load model: {e}")
            return
        
        # Initialize
        self.bridge = CvBridge()
        self.buffer = ""
        self.last_detection_time = 0
        self.detection_cooldown = 0.5  # 500ms cooldown between detections
        
        # NEW: Detection stability parameters
        self.detection_stability_time = rospy.get_param('~stability_time', 1.0)  # 1 second default
        self.current_detections = {}  # {class_name: first_detection_time}
        
        # Frame processing optimizations
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to avoid lag
        self.result_queue = queue.Queue(maxsize=2)
        self.processing_lock = Lock()
        self.latest_frame = None
        self.processing = False
        
        # Publishers
        self.image_pub = rospy.Publisher('/sign_detector/image_processed', Image, queue_size=1)
        self.buffer_pub = rospy.Publisher('/sign_detector/buffer', String, queue_size=1)
        self.sentence_pub = rospy.Publisher('/sign_detector/sentence', String, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/camera/image_raw', Image, self.image_callback, queue_size=1)
        # NEW: Subscribe to buffer updates from GUI
        rospy.Subscriber('/sign_detector/buffer_update', String, self.buffer_update_callback, queue_size=1)
        
        # Start processing thread
        self.processing_thread = Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        rospy.loginfo("🤖 Sign Detection Node is running (optimized)...")
        rospy.loginfo("📝 Letters will stick together to form words!")
    
    def buffer_update_callback(self, msg):
        """Handle buffer updates from GUI (like backspace operations)"""
        with self.processing_lock:
            self.buffer = msg.data
            rospy.loginfo(f"📝 Buffer updated from GUI: '{self.buffer}'")
    
    def image_callback(self, msg):
        # Drop old frames to reduce latency
        try:
            if not self.frame_queue.full():
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.frame_queue.put((frame, msg.header.stamp), block=False)
        except queue.Full:
            pass  # Drop frame if queue is full
        except Exception as e:
            rospy.logwarn(f"Frame conversion error: {e}")
    
    def process_frames(self):
        """Background thread for processing frames"""
        while not rospy.is_shutdown():
            try:
                # Get frame with timeout
                frame, timestamp = self.frame_queue.get(timeout=0.1)
                
                # Skip processing if we're behind
                if not self.frame_queue.empty():
                    continue  # Skip this frame, process the newer one
                
                # Resize frame for faster processing
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = 640
                    new_height = int(height * scale)
                    frame_resized = cv2.resize(frame, (new_width, new_height))
                else:
                    frame_resized = frame
                    scale = 1.0
                
                # Run detection with optimizations
                with torch.no_grad():
                    results = self.model(frame_resized, 
                                       conf=self.conf_threshold, 
                                       verbose=False,
                                       imgsz=640,  # Fixed input size
                                       half=True if torch.cuda.is_available() else False)  # FP16 if GPU
                
                # Process results on original frame
                processed_frame = self.process_detections(frame, results, scale)
                
                # Publish processed image
                try:
                    processed_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
                    processed_msg.header.stamp = timestamp
                    self.image_pub.publish(processed_msg)
                except Exception as e:
                    rospy.logwarn(f"Publishing error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logwarn(f"Processing error: {e}")
    
    def process_detections(self, frame, results, scale=1.0):
        current_time = rospy.Time.now().to_sec()
        
        # Track current frame detections
        frame_detections = set()
        
        # Draw bounding boxes and collect detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates and scale back to original size
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    if scale != 1.0:
                        x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                    
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.model.names[cls] if cls < len(self.model.names) else f"Class_{cls}"
                    frame_detections.add(class_name)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Check if this detection is stable enough
                    detection_duration = 0
                    if class_name in self.current_detections:
                        detection_duration = current_time - self.current_detections[class_name]
                    
                    # Draw label with stability indicator
                    if detection_duration >= self.detection_stability_time:
                        label = f"{class_name}: {conf:.2f} ✓"
                        label_color = (0, 255, 0)  # Green for stable
                    else:
                        progress = detection_duration / self.detection_stability_time
                        label = f"{class_name}: {conf:.2f} ({progress:.1%})"
                        label_color = (0, 165, 255)  # Orange for waiting
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), label_color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Update detection tracking
        with self.processing_lock:
            # Add new detections
            for class_name in frame_detections:
                if class_name not in self.current_detections:
                    self.current_detections[class_name] = current_time
                    rospy.loginfo(f"🔍 Starting to track: {class_name}")
            
            # Remove detections that are no longer visible
            to_remove = []
            for class_name in self.current_detections:
                if class_name not in frame_detections:
                    detection_duration = current_time - self.current_detections[class_name]
                    if detection_duration < self.detection_stability_time:
                        rospy.loginfo(f"⏰ Lost track of {class_name} after {detection_duration:.1f}s (needed {self.detection_stability_time}s)")
                    to_remove.append(class_name)
            
            for class_name in to_remove:
                del self.current_detections[class_name]
            
            # Add stable detections to buffer
            for class_name in frame_detections:
                if class_name in self.current_detections:
                    detection_duration = current_time - self.current_detections[class_name]
                    
                    # Check if detection is stable and we haven't added it recently
                    if (detection_duration >= self.detection_stability_time and 
                        current_time - self.last_detection_time > self.detection_cooldown):
                        
                        # FIXED: Add letter WITHOUT space to make them stick together
                        self.buffer += class_name  # No space added here!
                        self.buffer_pub.publish(String(data=self.buffer))
                        self.last_detection_time = current_time
                        rospy.loginfo(f"✅ Added letter to word: '{class_name}' → Current word: '{self.buffer}'")
                        
                        # Remove from tracking since it's been added
                        del self.current_detections[class_name]
                        break  # Only add one detection per frame
        
        return frame

if __name__ == '__main__':
    try:
        node = SignDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass