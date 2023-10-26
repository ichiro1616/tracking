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


def write_list_2d(sheet, l_2d, start_row, start_col):
    for y, row_data in enumerate(l_2d):
        # for x, cell_data in enumerate(row_data):
            sheet.cell(row=start_row + y, column=start_col, value=l_2d[y])


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


    # params["render_pose"] = 2  #レンダーにGPUを使用する
    # params["ip_camera"] = 'rtsp://root:P7CEqNF5ui8e@10.5.5.145:554/live1s1.sdp'
    params["video"] = "../examples/movie/ichiro_back.MOV"  #ファイルパス

    # params["video"] = "../examples/movie/uemura.MOV"  #ファイルパス
    # params["render_threshold"] = 0.05  #レンダーするキーポイントのスレッショルド
    params["write_video"] = "../examples/movie/output3.MOV"
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
        list_output.append("特徴量計算場所")
        list_output.append(str(counter) + "枚目")

        print(counter, "枚目")
        counter += 1
        # Pop frame
        if counter != 292:
            datumProcessed = op.VectorDatum()
            if opWrapper.waitAndPop(datumProcessed):
                if not args[0].no_display:
                    datum = datumProcessed[0]
                    original_image = datum.cvInputData
                    HSV_img = cv2.cvtColor(original_image,cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(HSV_img)
                    print("H:", h, "S:", s, "V:", v)
                    keypoint_array = [] #x, y
            
                    
                    for i in range(25):
                
                        prob = int(datum.poseKeypoints[0][i][2] * 100)

                        h_array = []
                        px_sum_h = 0
                        width_r =50
                        height_r =50
                        if(prob == 0):
                            keypoint_array.append([-1, -1])
                            print(-10000000)

                            list_output.append(-10000000)
                        else:
                            keypoint_x = int(datum.poseKeypoints[0][i][0])
                            keypoint_y = int(datum.poseKeypoints[0][i][1])
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


                    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvInputData)

                    key = cv2.waitKey(1)
                    key == 27
            else:
                break
        else:
            break
    excel_path = r"C:\Users\isapo\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\examples\distance.xlsx"

    wb = openpyxl.load_workbook(excel_path)
    sheet = wb.worksheets[3]

    write_list_2d(sheet, list_output, 4,73)

    wb.save(excel_path)
    wb.close()

except Exception as e:
    print(e)
    sys.exit(-1)
