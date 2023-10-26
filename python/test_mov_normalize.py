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
import openpyxl
import math


def write_list_2d(sheet, l_2d, start_row, start_col):
    row_count = 0
    col_count = 0 

    for y, row_data in enumerate(l_2d):
        # for x, cell_data in enumerate(row_data):
            if col_count == 25:
                row_count += 1
                col_count = 0
            sheet.cell(row=start_row + row_count, column=start_col + col_count, value=l_2d[y])
            col_count += 1

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
    # parser.add_argument("--image_dir", default="../examples/person/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../models/"

    moviefile = "hinata_front.mov"
    # params["render_pose"] = 2  #レンダーにGPUを使用する
    # params["ip_camera"] = 'rtsp://root:P7CEqNF5ui8e@10.5.5.145:554/live1s1.sdp'
    params["video"] = "../examples/movie/" + moviefile  #ファイルパス

    # params["video"] = "../examples/movie/uemura.MOV"  #ファイルパス
    # params["render_threshold"] = 0.05  #レンダーするキーポイントのスレッショルド
    params["write_video"] = "../examples/movie/output_" + moviefile
    # params["net_resolution"] = "-1x512"
    params["number_people_max"] = 1
    params["write_json"] = "../examples/json"
    # params["maximize_positives"] = True  #偽陽性と真陽性の両方を高度に増加させるため、平均精度を害する恐れがある




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
    opWrapper = op.WrapperPython(op.ThreadManagerMode.AsynchronousOut)

    opWrapper.configure(params)
    opWrapper.start()

    counter = 1
    # Main loop
    userWantsToExit = False
    list_output = []
    while not userWantsToExit:
        # list_output.append("特徴量計算場所")
        # list_output.append(str(counter) + "枚目")
        print("特徴量計算場所")

        print(counter, "枚目")
        counter += 1
        # Pop frame
        if counter !=  2700:
            datumProcessed = op.VectorDatum()
            if opWrapper.waitAndPop(datumProcessed):
                if not args[0].no_display:
                    datum = datumProcessed[0]
                    
                    imageToProcess = datum.cvInputData
                    # print(imageToProcess.shape)
                    # datum = op.Datum()
                    # imageToProcess = cv2.imread(datumProcessed[0])
                    # datum.cvInputData = imageToProcess
                    # opWrapper.emplaceAndPop(op.VectorDatum([datum]))



                    keypoint_neck_x = int(datum.poseKeypoints[0][1][0])
                    keypoint_neck_y = int(datum.poseKeypoints[0][1][1])

                    keypoint_hip_x = int(datum.poseKeypoints[0][8][0])
                    keypoint_hip_y = int(datum.poseKeypoints[0][8][1])
                    # print("首", "keypoint_x",int(keypoint_neck_x), "keypoint_y",int(keypoint_neck_y))
                    # print("腰", "keypoint_x",int(keypoint_hip_x), "keypoint_y",int(keypoint_hip_y))

                    sekitui = math.sqrt(abs(keypoint_hip_x - keypoint_neck_x) ** 2  + abs(keypoint_hip_y - keypoint_neck_y) ** 2)
                    # print(sekitui)
                    threshold = 80
                    if sekitui != 0:
                        resize_rate = round(threshold / sekitui, 2)
                        # print(resize_rate)
                        # print(imageToProcess.shape[0], imageToProcess.shape[1])
                    
                        # imageToProcess = cv2.imread(imagePath)
                        img_resize = cv2.resize(imageToProcess, (int(imageToProcess.shape[1] * resize_rate), int(imageToProcess.shape[0] * resize_rate)))

                        datum.cvInputData = img_resize
                        # opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                        HSV_img_save = cv2.cvtColor(imageToProcess,cv2.COLOR_BGR2HSV)

                        HSV_img = cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
                        h, s, v = cv2.split(HSV_img)
                        # print("H:", h, "S:", s, "V:", v)


                        keypoint_array = [] #x, y
                        # print("h_shape:",h.shape)
                
                        
                        for i in range(25):
                    
                            prob = int(datum.poseKeypoints[0][i][2] * 100)

                            h_array = []
                            px_sum_h = 0
                            width_r =10 #半径
                            height_r =10 #半径
                            if(prob == 0):
                                keypoint_array.append([-1, -1])
                                print(-10000000)

                                list_output.append(-10000000)

                            else:
                                keypoint_x = int(datum.poseKeypoints[0][i][0] * resize_rate)
                                keypoint_y = int(datum.poseKeypoints[0][i][1] * resize_rate)
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
                                list_output.append(px_ave_h)
                    # else:
                        # break;      


                    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvInputData)
                    outimg_files = [HSV_img_save]
                    # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", HSV_img)


                    key = cv2.waitKey(1)
                    key == 27
            else:
                break
        else:
            break;    
    # # 動画作成
    # fourcc = cv2.VideoWriter_fourcc('m','o','v', 'v')
    # video  = cv2.VideoWriter("../examples/movie/outputHSV_" + moviefile, fourcc, 20.0)

    # for img_file in outimg_files:
    #     img = cv2.imread(img_file)
    #     video.write(img)

    # video.release()
    # excel_path = r"C:\Users\isapo\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\examples\20x20.xlsx"
    # wb = openpyxl.load_workbook(excel_path)
    # sheet = wb.worksheets[0]

    # write_list_2d(sheet, list_output, 3,2) #行、列

    # wb.save(excel_path)
    # wb.close()

except Exception as e:
    print(e)
    sys.exit(-1)
