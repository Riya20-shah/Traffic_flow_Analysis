import time
from threading import Thread

import numpy
import cv2 as cv
import numpy as np
#from object_tracking import *
from object_detection import *
from count_vehicals_analysis import *

# data_path = "/home/scaledge-riya/Downloads/video.mp4"
# # data_path = "/home/scaledge-riya/Documents/Vehicals_dataset/train/images/Datacluster Labs Auto (38).jpg"
# cap = cv.VideoCapture(data_path)
# # -----------------  load the onnx model of yolov5 which is train on custom data ----------------
# net = cv.dnn.readNetFromONNX('/home/scaledge-riya/Desktop/Traffic_flow_Analysis/Models/yolov5x.onnx')
#
# file = open('classes.txt', 'r')
# classes = file.read().split('\n')
# # print(classes)
# detector = object_detection()
# tracker = obj_track_class()
# counterAndanlysier = Analysis()
# pTime = 0  #previous time
# while True:
#     rec,frm = cap.read()
#     if frm is None:
#         break
#     cTime = time.time()
#     fps = 1/(cTime - pTime)
#     pTime = cTime
#     # fps = cap.get(cv.CAP_PROP_FPS)
#     frm_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
#     duration = frm_count / fps
#     min = duration / 60
#     sec = duration % 60
#     print(f"fps:{fps} , frm_count {frm_count} , duration:{duration} , min {min} , sec {sec} ")
#
#     new_frm = cv.resize(frm , (1280,720))
#     height , width = new_frm.shape[:2]    # shape(720,1280,3)
#
#
#
#
#     # find the region
#     # roi = frm[int(height/1.5):,:]     # shape(240,1280,3)
#     # draw the line first so if object cross that line so we can consider its count
#     # cv.line(frm , (0 , 720-240) , (width,720-240) , (0,100,0) , 2)
#
#     detections = detector.detect_object(new_frm, net)
#
#     bbox , class_conf , classes_id , indices= detector.get_cordinates(detections,width,height)
#     detector.plot_boxes(class_conf , bbox , classes , classes_id , indices , new_frm , tracker)
#     counterAndanlysier.count_vehicals_traffic_classify(bbox , classes_id , classes , indices , new_frm)
#
#
#
#
#     # print(bbox)
#     cv.imshow("trayal" , new_frm)
#     # cv.imshow("roi", roi)
#     key = cv.waitKey(1)
#     if key == ord('q'):
#         break
# cv.destroyAllWindows()


class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv.VideoCapture(src)
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1 / 2
        self.FPS_MS = int(self.FPS * 50)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def show_frame(self):
        cv.imshow('frame', self.frame)
        cv.waitKey(self.FPS_MS)


if __name__ == '__main__':
    # data_path = "/home/scaledge-riya/Downloads/traffic.mp4"
    data_path = "/home/scaledge-riya/Desktop/Traffic_flow_Analysis/Dataset_test videos/los_angeles.mp4"
    threaded_camera = ThreadedCamera(data_path)
    # -----------------  load the onnx model of yolov5 which is train on custom data ----------------
    net = cv.dnn.readNetFromONNX('/home/scaledge-riya/Desktop/Traffic_flow_Analysis/Models/yolov5x.onnx')

    file = open('classes.txt', 'r')
    classes = file.read().split('\n')
    # print(classes)
    detector = object_detection()
    counterAndanlysier = Analysis()
    pTime = 0  # previous time
    Framenum = 1
    t1 = datetime.datetime.now()  # Current datetime


    while True:
        rec, frm = threaded_camera.capture.read()
        if frm is None:
            break

        fps = threaded_camera.capture.get(cv.CAP_PROP_FPS)
        new_frm = cv.resize(frm, (1280, 720))
        height, width = new_frm.shape[:2]  # shape(720,1280,3)

        # detect the object
        detections = detector.detect_object(new_frm, net)
        # get the cordinates of object
        bbox, class_conf, classes_id, indices = detector.get_cordinates(detections, width, height)
        #                                       draw rectangle box over object
        detector.plot_boxes(class_conf, bbox, classes, classes_id, indices, new_frm)
        # count the vehicals and classify the traffic status
        total_vehicals, n_frames = counterAndanlysier.count_vehicals_traffic_classify(bbox, classes_id, classes, indices, new_frm)




        cv.imshow("trayal", new_frm)
        # cv.imshow("roi", roi)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cv.destroyAllWindows()

    # saving all results in Excel file
    # counterAndanlysier.save_results()

