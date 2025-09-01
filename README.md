# ROSBasedJunoSign
Real-time Sign Language Recognition and Speech Synthesis

Communication is an essential human right, yet numerous individuals who are deaf or hard of hearing encounter daily challenges when engaging with non-signers. Although sign language is a key means of communication in deaf communities, the limited comprehension of sign language by the general public poses major communication obstacles.

Despite the increasing recognition of inclusivity in technology, there remains a shortage of real-time, autonomous systems capable of interpreting sign language and vocalising it to facilitate effective communication between deaf people and non-signers. There is a need for an intelligent and real-time solution that can recognise sign language, translate it to text, and convert it into speech, particularly on a robotic platform capable of interacting autonomously.

This project proposes JUNOSIGN, a ROS-integrated system within the Juno robot designs to function as an independent real-time sign language interpreter. It utilises a camera along with a computer vision gesture recognition model (YOLO) to identify American Sign Language (ASL) signs, transform them into test, and verbally express the result using a text-to-speech engine such as pyttsx3. The robot can comprehend fundamental gestures, such as signs related to emergencies, and is able to react to it using predefined responses. This approach encourages inclusivity and provides tangible support in everyday interactions.
