#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
out = cv2.VideoWriter(
    '/home/fjh/output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
index = 0
class PersonDetector:
    def __init__(self):

        # init cv_bridge
        self.bridge = CvBridge()
        # init publisher
        self.image_pub = rospy.Publisher("detected_image", Image, queue_size=1)

        # initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # init subscriber, to read ROS image msg
        self.image_sub = rospy.Subscriber("/kinect2/hd/image_color", Image, self.image_callback, queue_size=1)

    def image_callback(self, data):
        # read image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")     
            frame = np.array(cv_image, dtype=np.uint8)
        except CvBridgeError as e:
            print(e)
        # resizing for faster detection
        frame = cv2.resize(frame, (640, 480))
        # using a greyscale picture, also for faster detection
        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = self.hog.detectMultiScale(frame, winStride=(8,8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB),
                            (0, 255, 0), 2)
        
        # Write the output video 
        out.write(frame.astype('uint8'))
        # print for set threshold
        print("{} index weight is {}".format(index, weights))
        # Display the resulting frame
        cv2.imshow('frame',frame)

        
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

def main(args):
    detector = PersonDetector()
    rospy.init_node('person_detector', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
