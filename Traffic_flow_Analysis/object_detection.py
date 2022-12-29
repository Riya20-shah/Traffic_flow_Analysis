import cv2 as cv
import numpy as np


class object_detection():
    def detect_object(self ,img,net):
        try:
            # convert into a blob
            blobimg = cv.dnn.blobFromImage(cv.resize(img, (640, 640)), scalefactor=1 / 255, swapRB=True, size=(640, 640))
            # print(frm.shape , blobimg.shape)
            net.setInput(blobimg)
            detection = net.forward()[0]

            # print(detection[0][0])     # (1 , 25200 , 14) meaning of that is that are
            # detected 25200 bounding boxes and each of having 14 entries as columns
            # 14 entries classify  (cx , cy , w, h,confidence , our 9 classes score
            return detection
        except:
            pass

    def get_cordinates(self , detection , width , height):

        # now we are finding  class id  , confidence, boxes

        classes_id = []
        class_conf = []
        boxes = []
        rows = detection.shape[0]  # numbers of rows detected bounding box
        X_scale = width / 640  # we devide with 640 because blob image size is 640
        y_scale = height / 640  # we devide with 640 because blob image size is 640
        for i in range(rows):
            row = detection[i]
            confidence = row[4]
            if confidence > 0.5:
                classes_score = row[5:]
                index = np.argmax(classes_score)
                if classes_score[index] > 0.5:
                    classes_id.append(index)
                    class_conf.append(confidence)
                    cx, cy, w, h = row[:4]

                    #  so in yolo we get the normalize cordinates of box which is bx , by ,w , h
                    #  so to get x, y cordinate we have to multiply it with the image size x with image width and y with image height
                    x1 = int((cx - (w / 2)) * X_scale)
                    y1 = int((cy - (h / 2)) * y_scale)
                    width = int(w * X_scale)
                    height = int(h * y_scale)
                    box = np.array([x1, y1, height, width])
                    boxes.append(box)
        indices = cv.dnn.NMSBoxes(boxes, class_conf, 0.5, 0.5)

        return boxes , class_conf , classes_id , indices

    def plot_boxes(self , class_conf,boxes,classes,classes_id,indices ,img):
        # indices = cv.dnn.NMSBoxes(boxes, class_conf, 0.5, 0.5)

        for i in indices:
            x1, y1, h, w = boxes[i]
            label = classes[classes_id[i]]
            conf = class_conf[i]
            text = label + "{:.2f}".format(conf)
            cv.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
            cv.putText(img, text, (int(x1), int(y1 - 2)), cv.FONT_HERSHEY_COMPLEX, 0.7, (120, 120, 255), 2)
