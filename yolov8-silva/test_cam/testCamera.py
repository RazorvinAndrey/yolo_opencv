import numpy as np
import datetime
import triangulation as tri
import random
import cv2
from ultralytics import YOLO

# opening the file in read mode
my_file = open("../utils/coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolo.pt", "v8")
# import Calibration as calib

cv_file = cv2.FileStorage()
# cv_file.open('stereoMap.xml', cv2.FileStorage_READ)
#
# stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
# stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
# stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
# stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# ТУТ МЕНЯЕШЬ НАЗВАНИЕ ФАЙЛА КУДА ЗАПИСЫВАЕТСЯ ЛОГ
name = "local_test1/f1.txt"

file1 = open(name, 'w')

# Open both cameras
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 60) # Частота кадров
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Ширина кадров в видеопотоке. 640x360
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540) # Высота кадров в видеопотоке.

frame_rate = 60    #Camera frame rate (maximum at 120 fps)

B = 12               #Distance between the cameras [cm]
f = 2.12               #Camera lense's focal length [mm]
alpha = 110        #Camera field of view in the horisontal plane [degrees]

koef = 0.8
#не знаю почему нужен коэффициент, но дальность с ним показывает правильно,
# возможно какие-то параметры камеры записаны неправильно

#Initial values
circles_right = None
circles_left = None
count = -1
depth = 0
while(True):
    count += 1
    ret, frame = cap.read()
    frame_left = frame[0:539, 0:639]  #разделение видео на левый и правый поток
    frame_right = frame[0:539, 640:1279]
    if ret==False:
        break

    else:


        #------------------------------------ ЭТО ДЛЯ ДЕТЕКЦИИ БЕЗ НЕЙРОНКИ
        # APPLYING HSV-FILTER:
        # mask_right = hsv.add_HSV_filter(frame_right, 1)
        # mask_left = hsv.add_HSV_filter(frame_left, 0)
        #
        # # Result-frames after applying HSV-filter mask
        # res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
        # res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)
        #
        # # APPLYING SHAPE RECOGNITION:
        # circles_right = shape.find_circles(frame_right, mask_right)
        # circles_left = shape.find_circles(frame_left, mask_left)

        #--------------------------------------ЭТО НЕЙРОНКА

        detect_params_left = model.predict(source=[frame], conf=0.45, save=False)
        # detect_params_right = model.predict(source=[frame], conf=0.45, save=False)

        # Convert tensor array to numpy
        DP_left = detect_params_left[0].numpy()
        # DP_right = detect_params_right[0].numpy()

        if len(DP_left) != 0:
            for i in range(len(detect_params_left[0])):
                boxes = detect_params_left[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb_l = box.xyxy.numpy()[0]
                print(conf)

                cv2.rectangle(frame,(int(bb_l[0]), int(bb_l[1])),(int(bb_l[2]), int(bb_l[3])),detection_colors[int(clsID)],3,)
                x_l = (int(bb_l[2]) - int(bb_l[0]))//2 + int(bb_l[0])
                y_l = (int(bb_l[3]) - int(bb_l[1]))//2 + int(bb_l[1])
                # circles_left = (x_l, y_l)
                # print(i)
                if x_l > 639:
                    circles_right = (x_l-639, y_l)
                else:
                    circles_left = (x_l, y_l)
        else:
            circles_left = None
            circles_right = None

        print('left = ', circles_left, 'right = ', circles_right)
        # if len(DP_right) != 0:
        #     for i in range(len(detect_params_right[0])):
        #         boxes = detect_params_right[0].boxes
        #         box = boxes[i]  # returns one box
        #         clsID = box.cls.numpy()[0]
        #         conf = box.conf.numpy()[0]
        #         bb_r = box.xyxy.numpy()[0]
        #
        #         cv2.rectangle(frame_right, (int(bb_r[0]), int(bb_r[1])),(int(bb_r[2]), int(bb_r[3])),detection_colors[int(clsID)],3,)
        #         x_r = (int(bb_r[2]) - int(bb_r[0]))//2 + int(bb_r[0])
        #         y_r = (int(bb_r[3]) - int(bb_r[1]))//2 + int(bb_r[1])
        #         circles_right = (x_r, y_r)
        #         print('x = ', x_r)
        #         print('y = ', y_r)
        #


        ################## CALCULATING BALL DEPTH #########################################################

        # If no ball can be caught in one camera show text "TRACKING LOST"
        if np.all(circles_right) == None or np.all(circles_left) == None:
            cv2.putText(frame, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            # cv2.putText(frame, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        else:
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
            # All formulas used to find depth is in video presentaion
            depth = tri.find_depth(circles_right, circles_left, frame_right, frame_left, B, f, alpha)*koef

            cv2.putText(frame, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            # cv2.putText(frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            # cv2.putText(frame_right, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            cv2.putText(frame, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            print("Depth: ", depth)

        # # Show the frames
        dtC = str(datetime.datetime.now())
        cv2.imshow("frame right", frame)
        #
        file1 = open(name, 'a')
        # file1.write("Distance = " + str(depth) + " Data: " + dtC + '\n')
        file1.close()
        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()
