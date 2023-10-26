# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw, ImageFont
import math

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../bin/python/openpose/Release')
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' +  dir_path + '/../bin;'
        import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="../examples/person/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir)
    start = time.time()
    counter = 1

    # Process and display images
    person_array = []

    for imagePath in imagePaths:
        print(counter , "人目")
        # im = Image.new("RGB", (2448, 3264), (255, 255, 255))
        # draw = ImageDraw.Draw(im)

        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        print(imageToProcess.shape)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # print("Body keypoints: \n" + str(datum.poseKeypoints[0]))  
        # print("Body keypoints: \n" + str(datum.poseKeypoints[0][0]))
        # px_ave_array = [] #B,G,R 
        keypoint_array = [] #x, y
        # # print("h", len(h))
        keypoint_neck_x = int(datum.poseKeypoints[0][1][0])
        keypoint_neck_y = int(datum.poseKeypoints[0][1][1])

        keypoint_hip_x = int(datum.poseKeypoints[0][8][0])
        keypoint_hip_y = int(datum.poseKeypoints[0][8][1])
        print("首", "keypoint_x",int(keypoint_neck_x), "keypoint_y",int(keypoint_neck_y))
        print("腰", "keypoint_x",int(keypoint_hip_x), "keypoint_y",int(keypoint_hip_y))

        sekitui = math.sqrt(abs(keypoint_hip_x - keypoint_neck_x) ** 2  + abs(keypoint_hip_y - keypoint_neck_y) ** 2)
        print(sekitui)
        threshold = 1000
        resize_rate = round(threshold / sekitui, 2)
        print(resize_rate)
        print(imageToProcess.shape[0], imageToProcess.shape[1])
        print(int(imageToProcess.shape[0]  *resize_rate), int(imageToProcess.shape[1] * resize_rate))

        # imageToProcess = cv2.imread(imagePath)
        img_resize = cv2.resize(imageToProcess, (int(imageToProcess.shape[1] * resize_rate), int(imageToProcess.shape[0] * resize_rate)))

        # print(imageToProcess.shape[0], imageToProcess.shape[1])
        # print(imagePath.width, imagePath.height)
        datum.cvInputData = img_resize
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        print(img_resize.shape)

        keypoint_neck_x = int(datum.poseKeypoints[0][1][0])
        keypoint_neck_y = int(datum.poseKeypoints[0][1][1])

        keypoint_hip_x = int(datum.poseKeypoints[0][8][0])
        keypoint_hip_y = int(datum.poseKeypoints[0][8][1])
        print("首", "keypoint_x",int(keypoint_neck_x), "keypoint_y",int(keypoint_neck_y))
        print("腰", "keypoint_x",int(keypoint_hip_x), "keypoint_y",int(keypoint_hip_y))

        sekitui = math.sqrt(abs(keypoint_hip_x - keypoint_neck_x) ** 2  + abs(keypoint_hip_y - keypoint_neck_y) ** 2)
        print(sekitui)


        HSV_img = cv2.cvtColor(imageToProcess,cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(HSV_img)

        for i in range(25):
            
            prob = int(datum.poseKeypoints[0][i][2] * 100)
 
            # keypoint = keypoint.split()
            # print("keypoint",keypoint)
            # print("keypoint_x",int(keypoint_x))
            # print("keypoint_y",int(keypoint_y))
        
            h_array = []

            px_sum_h = 0
            width_r =50
            height_r =50
            if(prob == 0):
                keypoint_array.append([-1, -1])
            else:
                keypoint_x = int(datum.poseKeypoints[0][i][0])
                keypoint_y = int(datum.poseKeypoints[0][i][1])
                # print("番号", i, "keypoint_x",int(keypoint_x), "keypoint_y",int(keypoint_y))
        
                keypoint_array.append([int(keypoint_x), int(keypoint_y)])

        
                for j in range(-height_r, height_r, 1):
                    for k in range(-width_r, width_r, 1):
                        # print(keypoint_y + j)
                        if (keypoint_y + j) >= h.shape[0] or (keypoint_x + k) >= h.shape[1]:

                            px = 0
                        else:

                            px = h[keypoint_y + j][keypoint_x + k]
                        # print(px)
                        px_sum_h += px
                        h_array.append(px)
                if len(h_array) == 0:
                    px_ave_h = -1
                else:
                    px_ave_h = int(px_sum_h / len(h_array))

                # print("px_sum_h",int(px_sum_h))
                # print("h_array",len(h_array))
                
                print(px_ave_h)

        # im.show()
    
        if not args[0].no_display:
            # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)

            HSV_img = cv2.cvtColor(imageToProcess,cv2.COLOR_BGR2HSV)
            # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API",HSV_img)
            cv2.imwrite('../examples/output/output' + str(counter) + '.jpg', HSV_img)
        

            cv2.imwrite('../examples/output/output' + str(2) + '.jpg',  HSV_img)
            key = cv2.waitKey(15)
            if key == 27: break
        counter+=1
    end = time.time()
    # print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")



except Exception as e:
    print(e)
    sys.exit(-1)
