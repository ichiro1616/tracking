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
        im = Image.new("RGB", (400,600), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        # print(imageToProcess.shape)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # print("Body keypoints: \n" + str(datum.poseKeypoints[0]))
        # print("Body keypoints: \n" + str(datum.poseKeypoints[0][0]))
        px_ave_array = [] #B,G,R 
        keypoint_array = [] #x, y



        for i in range(25):
            
            prob = int(datum.poseKeypoints[0][i][2] * 100)

            # keypoint = keypoint.split()
            # print("keypoint",keypoint)
            # print("keypoint_x",int(keypoint_x))
            # print("keypoint_y",int(keypoint_y))
        
        
            px_array_B = []
            px_array_G = []
            px_array_R = []
            px_sum_B = 0
            px_sum_G = 0
            px_sum_R = 0
            width_r =30
            height_r =30
            if(prob == 0):
                keypoint_array.append([-1, -1])
                px_ave_array.append([-1, -1, -1])
            else:
                keypoint_x = int(datum.poseKeypoints[0][i][0])
                keypoint_y = int(datum.poseKeypoints[0][i][1])
                keypoint_array.append([int(keypoint_x), int(keypoint_y)])

        
                for j in range(-height_r, height_r, 1):
                    for k in range(-width_r, width_r, 1):
                        px = imageToProcess[keypoint_y + j, keypoint_x + k]
                        px_str = str(px).replace("[", "").replace("]", "").split()
                        px_str[0] = int(px_str[0])
                        px_str[1] = int(px_str[1])
                        px_str[2] = int(px_str[2])
                        px_array_B.append(px_str[0])
                        px_array_G.append(px_str[1])
                        px_array_R.append(px_str[2])
                        px_sum_B += px_str[0]
                        px_sum_G += px_str[1]
                        px_sum_R += px_str[2]


                # print("B", px_array_B)
                # print("G", px_array_G)
                # print("R", px_array_R)
                px_ave_B = int(px_sum_B / len(px_array_B))
                px_ave_G = int(px_sum_G / len(px_array_G))
                px_ave_R = int(px_sum_R / len(px_array_R))
                px_ave_array.append([px_ave_B, px_ave_G, px_ave_R])
                # draw.point((int(keypoint_x), int(keypoint_y)), fill=(px_str[0], px_str[1], px_str[2]))
                draw.rectangle([(keypoint_x - width_r, keypoint_y - height_r), (keypoint_x + width_r, keypoint_y + height_r)], fill=(px_ave_R, px_ave_G, px_ave_B)) # fillは左からR,G,B



        # print("B_ave", px_ave_B)
        # print("G_ave", px_ave_G)
        # print("R_ave", px_ave_R)
        person_array.append([keypoint_array, px_ave_array])
        # print("person_array",person_array) 
        # print("px_ave_array",px_ave_array)
        # print("keypoint_array",keypoint_array)


    
        im.show()
    
        if not args[0].no_display:
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            # HSV_img = cv2.cvtColor(imageToProcess,cv2.COLOR_BGR2HSV)
            # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API",HSV_img)
            # cv2.imwrite('../examples/output/output' + str(counter) + '.jpg', HSV_img)
        

            cv2.imwrite('../examples/output/output' + str(counter) + '.jpg', datum.cvOutputData)
            key = cv2.waitKey(15)
            if key == 27: break
        counter+=1
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")

    # print(len(person_array[0][0]))
    dif_x_array = []
    dif_y_array = []
    dif_x_per_array = []
    dif_y_per_array = []

    dif_color_array = []
    dif_array = []



    for i in range(len(person_array[0][0])):
        if(person_array[0][0][i][0] == -1 or person_array[1][0][i][0] == -1):
            dif_x = -1
            dif_x_per = -1

        else:
            dif_x_per = int((abs(person_array[0][0][i][0] - person_array[1][0][i][0]) / 2448) * 100)
            dif_x = abs(person_array[0][0][i][0] - person_array[1][0][i][0])

        if(person_array[0][0][i][1] == -1 or person_array[1][0][i][1] == -1):
            dif_y = -1
            dif_y_per = -1

        else:
            dif_y_per = int((abs(person_array[0][0][i][1] - person_array[1][0][i][1]) / 3264) * 100)
            dif_y = abs(person_array[0][0][i][1] - person_array[1][0][i][1]) 

        dif_x_array.append(dif_x)
        dif_y_array.append(dif_y)
        dif_x_per_array.append(dif_x_per)
        dif_y_per_array.append(dif_y_per)

        if(person_array[0][1][i][0] == -1 or person_array[1][1][i][0] == -1):
            dif_color = -1
        else:
            dif_color_B = (abs(person_array[0][1][i][0] - person_array[1][1][i][0]) / 255) * 100
            dif_color_G = (abs(person_array[0][1][i][1] - person_array[1][1][i][1]) / 255) * 100
            dif_color_R = (abs(person_array[0][1][i][2] - person_array[1][1][i][2]) / 255) * 100
            dif_color = int((dif_color_B + dif_color_G + dif_color_R) / 3)

        dif_color_array.append(dif_color)
    print("dif_x", dif_x_array)
    print("dif_x_per", dif_x_per_array)

    print("dif_y", dif_y_array)
    print("dif_y_per", dif_y_per_array)

    print("dif_color", dif_color_array)
    print("person_array",person_array) 

    # x = dif_x_array
    # y = dif_x_per_array
    # plt.scatter(x, y)
    # plt.xlim(0, 2448)
    # # y軸の範囲を変更する
    # plt.ylim(0, 100)
    # plt.show() # プロットを表示



except Exception as e:
    print(e)
    sys.exit(-1)
