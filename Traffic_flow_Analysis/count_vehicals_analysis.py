import datetime
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import time

import psutil    #this library we use for ploting live graph

class Analysis():

    # Min confidence in % for the object detection
    min_conf_perc = 50
    # Frontier X coordinate (to define left lane )
    X_lane_frontier = 640
    # low & high limit for the traffic class (Light , heavy , moderate)
    low_limit = 5
    high_limit = 10
    frame = 0
    # printing information
    currentdate = datetime.datetime.now()

    n_vehicals_left = []
    n_vehicals_right = []
    n_total_vehicals = []
    currentDateTime = []
    frames_num= []
    traffic_state_left = []
    traffic_state_right = []


    videoResults_df = pd.DataFrame()

    def count_vehicals_traffic_classify(self ,bbox, classes_id , classes , indices ,img):
        self.frame += 1
        # width & height of the video
        w = img.shape[0]
        h = img.shape[1]

        # Frontier defination
        lanes_left = (0, self.X_lane_frontier)
        lanes_down = (self.X_lane_frontier, w)

        # count variable initialise
        counter1 = 1
        n_cars = n_trucks = n_moterbikes = n_buses = n_cycles = 0
        l_n_cars = l_n_truck = l_n_moterbike = l_n_bus = l_n_cycle = 0
        r_n_cars = r_n_truck = r_n_moterbike = r_n_bus = r_n_cycle = 0
        # get the cordinates from the detection class
        for i in indices:

            label = classes[classes_id[i]]
            if label == 'car':
                n_cars += 1
            if label == 'truck':
                n_trucks += 1
            if label == 'motorcycle':
                n_moterbikes += 1
            if label == 'bus':
                n_buses += 1
            n_vehicles = n_cars + n_trucks + n_moterbikes + n_buses


            roi = bbox[i]
            # x_center_roi = (roi[0] + roi[2]) / 2
            # y_center_roi = (roi[1] + roi[3]) / 2
            # print("roi",roi[0])
            if roi[0] < self.X_lane_frontier:  # counting vehicals for the left lane
                if label == 'car':
                    l_n_cars += 1
                if label == 'truck':
                    l_n_truck += 1
                if label == 'motorcycle':
                    l_n_moterbike += 1
                if label == 'bus':
                    l_n_bus += 1
            n_vehicles_l = l_n_cars + l_n_truck + l_n_moterbike + l_n_bus

            if roi[0] > self.X_lane_frontier:  # counting vehicals for the left lane
                if label == 'car':
                    r_n_cars += 1
                if label == 'truck':
                    r_n_truck += 1
                if label == 'motorcycle':
                    r_n_moterbike += 1
                if label == 'bus':
                    r_n_bus += 1
            n_vehicles_r = r_n_cars + r_n_truck + r_n_moterbike + r_n_bus
        # print(f"current time min {self.currentdate.min } , current time {self.currentdate.second} and number of vehicals {n_vehicles}" , )

        # Traffic class

        if n_vehicles_l < self.low_limit:
            left_traffic_state = "Low Traffic"
        if n_vehicles_l >= self.low_limit and n_vehicles_l < self.high_limit:
            left_traffic_state = "Moderate Traffic"
        if n_vehicles_l >= self.high_limit:
            left_traffic_state = "High Traffic"

        if n_vehicles_r < self.low_limit:
            right_traffic_state = "Low Traffic"
        if n_vehicles_r >= self.low_limit and n_vehicles_r <self.high_limit:
            right_traffic_state = "Moderate Traffic"
        if n_vehicles_r >= self.high_limit:
            right_traffic_state = "High Traffic"


        # we appending the all records in list so we can use in analysis purpose
        self.n_total_vehicals.append(n_vehicles)
        self.n_vehicals_right.append(n_vehicles_r)
        self.n_vehicals_left.append(n_vehicles_l)
        self.traffic_state_left.append(left_traffic_state)
        self.traffic_state_right.append(right_traffic_state)
        self.frames_num.append(self.frame)
        self.currentDateTime.append(self.currentdate)

        totalvehicles = "Total number of vehicules = " + str(n_vehicles)
        totalvehicles_l = "Total number of vehicules in left = " + str(n_vehicles_l)
        totalvehicles_r = "Total number of vehicules in Right = " + str(n_vehicles_r)
        traffic_state_l = "Traffic state = " + str(left_traffic_state)
        traffic_state_r = "Traffic state = " + str(right_traffic_state)
        cv.line(img, (self.X_lane_frontier, 0), (self.X_lane_frontier, h), (100, 200, 200), 2)  # Frontier line
        cv.putText(img , totalvehicles , (10,80) ,cv.FONT_HERSHEY_COMPLEX, 0.7, (100, 0, 100), 2)
        cv.putText(img, totalvehicles_l, (10, 110), cv.FONT_HERSHEY_COMPLEX, 0.7, (125, 0, 255), 2)
        cv.putText(img, totalvehicles_r, (670, 105), cv.FONT_HERSHEY_COMPLEX, 0.7, (125, 0, 255), 2)
        cv.putText(img, traffic_state_l, (10, 140), cv.FONT_HERSHEY_COMPLEX, 0.7, (200, 0, 0), 2)
        cv.putText(img, traffic_state_r, (670, 140), cv.FONT_HERSHEY_COMPLEX, 0.7, (200, 0, 0), 2)


        return n_vehicles , self.frame


    def save_results(self):
        #  Plotting graph

       #  create Data frame
        self.videoResults_df["Date_Time"] = self.currentDateTime
        self.videoResults_df["Number of Frames"] = self.frames_num
        self.videoResults_df["Total Vehicals"] = self.n_total_vehicals
        self.videoResults_df["Number of Vehicals in Left"] = self.n_vehicals_left
        self.videoResults_df["Number of Vehicals in Right"] = self.n_vehicals_right
        self.videoResults_df["Traffic in Left"] = self.traffic_state_left
        self.videoResults_df["Traffic in Right"] = self.traffic_state_right

        self.videoResults_df.to_csv("videoresults.csv")

    def plot_live_graph(self , num_vehicals , frame):
        i = 0
        x,y = [] , []
        plt.rcParams['animation.html'] = 'jshtml'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.show()
        while True:
            x.append(frame)
            y.append(num_vehicals)
            ax.plot(x, y)
            fig.canvas.draw()
            # time.sleep(0.1)
            i = i + 1