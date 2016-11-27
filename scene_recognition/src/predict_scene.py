#!/usr/bin/env python
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from std_msgs.msg import String, Float64MultiArray
from SceneRecognizer import SceneRecognizer
import sensor_msgs.msg
from PIL import Image
import numpy as np


class ScenePredictor:
    def __init__(self):
        self.bridge = CvBridge()
        self.current_image = None
        self.image_sub = rospy.Subscriber('/camera/rgb/image_color',sensor_msgs.msg.Image,self.image_callback,queue_size=1)
        self.scene_pub = rospy.Publisher('/scene_prediction/raw_scene',String,queue_size=1)
        self.prob_pub = rospy.Publisher('/scene_prediction/raw_prob',Float64MultiArray,queue_size=1)
        self.average_pub = rospy.Publisher('/scene_prediction/comp_prob',Float64MultiArray,queue_size=1)
        self.svm = SceneRecognizer(None)
        self.svm.load_model('/home/servicerobot2/catkin_ws/src/scene_recognition/src/SVM_model.pkl')
        self.averages = np.array([0, 0, 0])
        self.alpha = 0.50

    def image_callback(self,imMsg):
         try:
             cv2_image = self.bridge.imgmsg_to_cv2(imMsg, "bgr8")
             cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGB)
             self.current_image = Image.fromarray(cv2_image)
             self.raw_scene()
             self.running_averages()
             # print(probs)

         except CvBridgeError as e:
             print(e)

    def raw_scene(self):
        if self.current_image:
            category = self.svm.predict(self.current_image)
            self.scene_pub.publish(category)

    def running_averages(self):
        if self.current_image:
            probs = np.array(self.svm.probs())
            self.averages = self.alpha * probs + (1 - self.alpha) * self.averages
            msg_probs = Float64MultiArray()
            msg_probs.data.append(probs[0][0])
            msg_probs.data.append(probs[0][1])
            msg_probs.data.append(probs[0][2])
            msg_avg = Float64MultiArray()
            msg_avg.data.append(self.averages[0][0])
            msg_avg.data.append(self.averages[0][1])
            msg_avg.data.append(self.averages[0][2])
            self.prob_pub.publish(msg_probs)
            self.average_pub.publish(msg_avg)







def main():
    rospy.init_node('scene_predictor' , anonymous=True)
    sp = ScenePredictor()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

    return


if __name__ == '__main__':
    main()

