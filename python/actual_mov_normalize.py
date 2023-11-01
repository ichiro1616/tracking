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


def sim_distance(person_now, person_previous): #1フレーム前の人と色相を比較し類似度を算出
    sum_of_sqrt = 0
    for x in range(len(person_now)):
        if person_now[x] != -1 or person_previous != -1:
            calculation_source =  abs(int(person_now[x]) - int(person_previous[x]))
            ring_calculate = abs(calculation_source - 180) if round((calculation_source / 180)) * 180 > 90 else calculation_source
            # print(calculation_source, round((calculation_source / 180)) * 180, ring_calculate)
            sum_of_sqrt += (ring_calculate)**2
    sum_of_sqrt = math.sqrt(sum_of_sqrt)
    print(sum_of_sqrt, sum_of_sqrt/(90 * len(person_now)))
    return sum_of_sqrt/(90 * len(person_now))


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

    moviefile = "ichiro_otsuka.mov"
    # params["render_pose"] = 2  #レンダーにGPUを使用する
    # params["ip_camera"] = 'rtsp://root:P7CEqNF5ui8e@10.5.5.145:554/live1s1.sdp'
    params["video"] = "../examples/movie/" + moviefile  #ファイルパス

    # params["render_threshold"] = 0.05  #レンダーするキーポイントのスレッショルド
    params["write_video"] = "../examples/movie/output_no_tracking_" + moviefile
    # params["net_resolution"] = "-1x512"
    params["number_people_max"] = 5
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
    person_h_array_previous = []
    
    keypoint_judge =  {"鼻" : 0, "胸" : 1, "右肩" : 1, "右ひじ" : 1, "右手首" : 0, "左肩" : 1, "左ひじ" : 1, "左手首" : 0, "腰" : 1, "右腰" : 0, "右ひざ" : 1, "右足首" : 0, "左腰" : 0, "左ひざ" : 1, "左足首" : 0, "右目" : 0, "左目" : 0, "右耳" : 0, "左耳" : 0, "左親指" : 0, "左小指" : 0, "左かかと" : 0, "右親指" : 0, "右小指" : 0, "右かかと" : 0}
    while not userWantsToExit:
        person_h_array = []
        person_keypoints_array = []
        person_box_array = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255]] #B,G,Rを人数分
        person_text_array = ["person1", "person2","person3","person4", "person5"]
        # h_list_output.append("特徴量計算場所")
        # h_list_output.append(str(counter) + "枚目")
        print("-------------------------------------------------")
        
        print("特徴量計算場所")

        print(counter, "枚目")
        # Pop frame
        datumProcessed = op.VectorDatum()
        if opWrapper.waitAndPop(datumProcessed):
            if not args[0].no_display:
                datum = datumProcessed[0]
                imageToProcess = datum.cvInputData
                # print(len(datum.poseKeypoints))
                #首と腰のキーポイントを利用して画像をリサイズする。
                # keypoint_chest_x = int(datum.poseKeypoints[0][1][0])
                # keypoint_chest_y = int(datum.poseKeypoints[0][1][1])

                # keypoint_hip_x = int(datum.poseKeypoints[0][8][0])
                # keypoint_hip_y = int(datum.poseKeypoints[0][8][1])

                # sekitui = math.sqrt((keypoint_hip_x - keypoint_chest_x) ** 2  + (keypoint_hip_y - keypoint_chest_y) ** 2)
                # print("1人目脊椎:", sekitui)
                
                # keypoint_chest_x = int(datum.poseKeypoints[1][1][0])
                # keypoint_chest_y = int(datum.poseKeypoints[1][1][1])

                # keypoint_hip_x = int(datum.poseKeypoints[1][8][0])
                # keypoint_hip_y = int(datum.poseKeypoints[1][8][1])

                # sekitui = math.sqrt((keypoint_hip_x - keypoint_chest_x) ** 2  + (keypoint_hip_y - keypoint_chest_y) ** 2)
                # print("2人目脊椎:", sekitui)

                h_array = []
                keypoints_array = []
                for d in range(len(datum.poseKeypoints)): #識別されている人数だけ回す
                    keypoints_list_output = []
                    h_list_output = []

                    for i, val in enumerate(keypoint_judge.values()): #keypointの数だけ回す
                        if val == 1: #keypointを使うかどうか

                            prob = int(datum.poseKeypoints[d][i][2] * 100)

                            width_r =25 #半径  一人ひとり試していた時はsekitui = 80px width,height_r = 10で試していたが、このコードではリサイズしていない、sekituiがデフォルトで200
                            height_r =25 #半径

                            if(prob == 0):
                                keypoints_list_output.append([-1, -1])
                                h_list_output.append(-1)


                            else:
                                keypoint_x = int(datum.poseKeypoints[d][i][0])
                                keypoint_y = int(datum.poseKeypoints[d][i][1])
                                keypoints_list_output.append([keypoint_x, keypoint_y])

                                # img_area = img_resize[keypoint_y - height_r : keypoint_y + height_r, keypoint_x - width_r : keypoint_x + width_r]
                                img_area = imageToProcess[keypoint_y - height_r : keypoint_y + height_r, keypoint_x - width_r : keypoint_x + width_r]

                                rsum,gsum,bsum = 0.0,0.0,0.0

                                #画像の範囲指定をしたimg_areaからRGBをそれぞれ2次元配列で全px分取得。それをravelで1次元化してから平均を取得している
                                ravg = np.ravel(img_area[:, :, 2]).mean() 
                                gavg = np.ravel(img_area[:, :, 1]).mean()
                                bavg = np.ravel(img_area[:, :, 0]).mean()

                                hsv = cv2.cvtColor(np.array([[[bavg, gavg, ravg]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
                                # print("H:", hsv[0])
                                h_list_output.append(hsv[0])
                    h_array.append(h_list_output)
                    keypoints_array.append(keypoints_list_output)



                person_h_array = h_array[:]
                person_keypoints_array = keypoints_array[:]
                # print(len(person_h_array), len(person_keypoints_array))

                print("現フレーム：",person_h_array)
                print("前フレーム：",person_h_array_previous)
                if len(person_h_array_previous) == 0:
                    person_h_array_previous = person_h_array[:]
        

                    for d in range(len(datum.poseKeypoints)): #識別されている人数だけ回す
                        start_x = person_keypoints_array[d][0][0] -100 #person_keypoints_array[何人目か][どこのキーポイントか][xかyか]
                        start_y = person_keypoints_array[d][0][1]-300
                        end_x = person_keypoints_array[d][5][0]+100
                        end_y = person_keypoints_array[d][5][1]+300
                        cv2.rectangle(imageToProcess,(start_x, start_y), (end_x, end_y), (person_box_array[d][0], person_box_array[d][1], person_box_array[d][2]), 10) #person_keypoints_array[何人目か][どこのキーポイントか][xかyか]
                        cv2.putText(imageToProcess, person_text_array[d], (start_x, start_y), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (person_box_array[d][0], person_box_array[d][1], person_box_array[d][2]))


                else:
                    person_h_array_list_output =[]
                    exclusion_array = [] #1フレーム前のすでに選ばれた人の重複を避ける
                    for d in range(len(datum.poseKeypoints)): #識別されている人数だけ回す
                        print(d + 1, "人目")
                        sim_array = [] #現フレームの一人の人物に対して、1フレーム前の全員分の類似度を追加する
                        for h in person_h_array_previous: #過去の人の人数分回す
                            sim_array.append(sim_distance(person_h_array[d], h)) #h:1人分の色相データ、person_h_array_previous:1フレーム前の人の色相データ
                        print(sim_array)
                        exclusion_judge = False
                        while exclusion_judge == False:
                            print("while")
                            sim_max_index = sim_array.index(max(sim_array)) #過去の何人目の人との類似度が一番高いか
                            print(exclusion_array, sim_max_index)
                            print(sim_max_index in exclusion_array)
                            if((sim_max_index in exclusion_array) == True):
                                sim_array[sim_max_index] = -1
                                print("過去の", sim_max_index + 1, "番目の人はすでに選ばれています")

                            else:
                                exclusion_judge = True
                                exclusion_array.append(sim_max_index)
                                print("過去の", sim_max_index + 1, "番目の人と類似度が一番近い")
                        
                        start_x = person_keypoints_array[d][0][0] -100
                        start_y = person_keypoints_array[d][0][1]-300
                        end_x = person_keypoints_array[d][5][0]+100
                        end_y = person_keypoints_array[d][5][1]+300
                        cv2.rectangle(imageToProcess,(start_x, start_y), (end_x, end_y), (person_box_array[sim_max_index][0], person_box_array[sim_max_index][1], person_box_array[sim_max_index][2]), 10)
                        cv2.putText(imageToProcess, person_text_array[sim_max_index], (start_x, start_y), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (person_box_array[sim_max_index][0], person_box_array[sim_max_index][1], person_box_array[sim_max_index][2]))
                        person_h_array_list_output.append(person_h_array[sim_max_index])
                    
                    person_h_array_previous = person_h_array_list_output[:]
                cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", imageToProcess)

                key = cv2.waitKey(1)
                key == 27
            
        else:
            break
        counter += 1


except Exception as e:
    print(e)
    sys.exit(-1)


# def sim_distance(prefs, person1, person2):
#     # person1とperson2が共に評価してるもののリスト
#     si = {}

#     for item in prefs[person1]:
#         if item in prefs[person2]:
#             si[item] = 1

#     # person1とperson2がどちらも評価してるものが無ければ類似性は0
#     if len(si) == 0 :
#         return 0

#     # 各項目ごとの差の平方
#     squares = [(prefs[person1][item] - prefs[person2][item]) ** 2 for item in si]
#     sum_of_sqrt = math.sqrt(sum(squares))
#     return 1/(1 + sum_of_sqrt)

