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


def sim_distance(person_now, person_previous): #0フレーム目の人と色相を比較し類似度を算出
    sum_of_sqrt = 0
    for x in range(len(person_now)):
        if person_now[x] != -1 or person_previous != -1:
            calculation_source =  abs(int(person_now[x]) - int(person_previous[x]))
            ring_calculate = abs(calculation_source - 180) if round((calculation_source / 180)) * 180 > 90 else calculation_source
            # print(calculation_source, round((calculation_source / 180)) * 180, ring_calculate)
            sum_of_sqrt += (ring_calculate)**2
    sum_of_sqrt = math.sqrt(sum_of_sqrt)
    bottom = math.sqrt((90)**2 * len(person_now)) #色相の差分の最大値の2乗xキーポイントの数の平方根をとっている
    print(sum_of_sqrt, 1 - sum_of_sqrt/bottom)
    return 1 - sum_of_sqrt/ bottom

def list_tranpose(matrix): #2次元配列を転置している
    # print(type(matrix))
    # <class 'numpy.ndarray'>
    matrix_np_t = matrix.T
    # print(matrix_np_t)

    matrix_np_t_list = matrix_np_t.tolist()
    # print(type(matrix_np_t_list))
    # <class 'list'>
    return matrix_np_t_list

def search_minmax(list_tr): #信頼度が一定以上のキーポイントの中で最小最大の値をxy座標ごとに返す
    keypoint_x_array = list_tr[0] #人物を四角で囲うために全キーポイントでの最小、最大のxy座標を取得
    keypoint_y_array = list_tr[1]
    reli_array = list_tr[2]
    for rel in range(len(reli_array)): #各キーポイントで信頼度が一定以下なら、そのxy座標を平均化して最小、最大で取得できないようにする。
        if list_tr[2][rel] < 0.5:
            keypoint_x_array[rel] = np.mean(keypoint_x_array)
            keypoint_y_array[rel] = np.mean(keypoint_y_array)


    keypoint_x_min = min(keypoint_x_array)
    keypoint_y_min = min(keypoint_y_array)
    keypoint_x_max = max(keypoint_x_array)
    keypoint_y_max = max(keypoint_y_array)
    return [keypoint_x_min, keypoint_y_min], [keypoint_x_max, keypoint_y_max]


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

    moviefile = "haru_hinata_updown2.mov"
    # params["render_pose"] = 2  #レンダーにGPUを使用する
    # params["ip_camera"] = 'rtsp://root:P7CEqNF5ui8e@10.5.5.145:554/live1s1.sdp'
    params["video"] = "../examples/movie/" + moviefile  #ファイルパス

    # params["render_threshold"] = 0.05  #レンダーするキーポイントのスレッショルド
    params["write_video"] = "../examples/movie/output_tracking_" + moviefile
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
    person_h_array_original = []
    
    keypoint_judge =  {"鼻" : 0, "胸" : 1, "右肩" : 1, "右ひじ" : 1, "右手首" : 0, "左肩" : 1, "左ひじ" : 1, "左手首" : 0, "腰" : 1, "右腰" : 0, "右ひざ" : 1, "右足首" : 0, "左腰" : 0, "左ひざ" : 1, "左足首" : 0, "右目" : 0, "左目" : 0, "右耳" : 0, "左耳" : 0, "左親指" : 0, "左小指" : 0, "左かかと" : 0, "右親指" : 0, "右小指" : 0, "右かかと" : 0}
    while not userWantsToExit:
        person_h_array = []
        person_keypoints_array = []
        person_keypoints_min_array = []
        person_keypoints_max_array = []

        person_box_array = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255]] #B,G,Rを人数分
        # h_list_output.append("特徴量計算場所")
        # h_list_output.append(str(counter) + "枚目")
        print("-------------------------------------------------")
        
        # print("特徴量計算場所")

        print(counter, "枚目")
        # Pop frame
        datumProcessed = op.VectorDatum()
        if opWrapper.waitAndPop(datumProcessed):
            if not args[0].no_display:
                datum = datumProcessed[0]
                imageToProcess = datum.cvInputData
            
                h_array = []
                keypoints_array = []
                keypoints_min_array = []
                keypoints_max_array = []

                for d in range(len(datum.poseKeypoints)): #識別されている人数だけ回す
                    keypoints_list_output = []
                    h_list_output = []


                    list_tranpose_array_normalize = list_tranpose(datum.poseKeypoints[d]) #キーポイントデータを転置してlistにしている
                    keypoint_minmax_normalize = search_minmax(list_tranpose_array_normalize)
                    # # print(keypoint_minmax)
                    
                    # 首と腰のキーポイントを利用して画像をリサイズする。
                    keypoint_chest_x = int(datum.poseKeypoints[d][1][0])
                    keypoint_chest_y = int(datum.poseKeypoints[d][1][1])
                    keypoint_hip_x = int(datum.poseKeypoints[d][8][0])
                    keypoint_hip_y = int(datum.poseKeypoints[d][8][1])
                    
                    sekitui = math.sqrt((keypoint_hip_x - keypoint_chest_x) ** 2  + (keypoint_hip_y - keypoint_chest_y) ** 2)
                    print(d + 1, "人目脊椎:", sekitui)

                    if sekitui != 0:

                        person_image = imageToProcess[int(keypoint_minmax_normalize[0][1]) : int(keypoint_minmax_normalize[1][1]), int(keypoint_minmax_normalize[0][0]) : int(keypoint_minmax_normalize[1][0])]
                        
                        threshold = 200.0 #胸から腰までを80pxにするという基準値

                        resize_rate = round(threshold / sekitui, 2)
                        img_resize = cv2.resize(person_image, (int(person_image.shape[1] * resize_rate), int(person_image.shape[0] * resize_rate)))
                        print(resize_rate,"|",img_resize.shape[1], "|",img_resize.shape[0])
        
                        datum.cvInputData = img_resize



                    
                        for i, val in enumerate(keypoint_judge.values()): #keypointの数だけ回す。余裕があったらkeypoint_judgeの中で1になっている数だけ回す。そのindexを取得するようにしたら、1行下のif文がいらなくなる。
                            if val == 1: #keypointを使うかどうか

                                prob = int(datum.poseKeypoints[d][i][2] * 100)

                                width_r =25 #半径  一人ひとり試していた時はsekitui = 80px width,height_r = 10で試していたが、このコードではリサイズしていない、sekituiがデフォルトで200
                                height_r =25 #半径

                                if(prob == 0):
                                    keypoints_list_output.append([-1, -1])
                                    h_list_output.append(-1)


                                else:
                                    keypoint_x = int(datum.poseKeypoints[d][i][0] - keypoint_minmax_normalize[0][0])
                                    keypoint_y = int(datum.poseKeypoints[d][i][1] - keypoint_minmax_normalize[0][1])
                                    keypoints_list_output.append([keypoint_x, keypoint_y])
                                    # print(keypoint_x, keypoint_y,  keypoint_minmax_normalize[0][1], keypoint_minmax_normalize[1][1])
                                    img_area = person_image[keypoint_y - height_r : keypoint_y + height_r, keypoint_x - width_r : keypoint_x + width_r]
                                    # print("|||||",img_area.shape[1], "||||", img_area.shape[0])
                                    # img_area = imageToProcess[keypoint_y - height_r : keypoint_y + height_r, keypoint_x - width_r : keypoint_x + width_r]

                                    rsum,gsum,bsum = 0.0,0.0,0.0

                                    #画像の範囲指定をしたimg_areaからRGBをそれぞれ2次元配列で全px分取得。それをravelで1次元化してから平均を取得している
                                    ravg = np.ravel(img_area[:, :, 2]).mean() 
                                    gavg = np.ravel(img_area[:, :, 1]).mean()
                                    bavg = np.ravel(img_area[:, :, 0]).mean()
                                    # print(img_area[:, :, 2])
                                    # print(img_area[:, :, 1])
                                    # print(img_area[:, :, 0])


                                    hsv = cv2.cvtColor(np.array([[[bavg, gavg, ravg]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
                                    # print("H:", hsv[0])
                                    h_list_output.append(hsv[0])

                        list_tranpose_array = list_tranpose(datum.poseKeypoints[d]) #キーポイントデータを転置してlistにしている
                        
                        keypoint_minmax = search_minmax(list_tranpose_array)
                        # print(keypoint_minmax)

                        h_array.append(h_list_output)
                        keypoints_array.append(keypoints_list_output)
                        keypoints_min_array.append([int(keypoint_minmax[0][0]), int(keypoint_minmax[0][1])]) #float型をint型にしている
                        keypoints_max_array.append([int(keypoint_minmax[1][0]), int(keypoint_minmax[1][1])])




                person_h_array = h_array[:]
                person_keypoints_array = keypoints_array[:]
                person_keypoints_min_array = keypoints_min_array[:]
                person_keypoints_max_array = keypoints_max_array[:]

                # print(len(person_h_array), len(person_keypoints_array))
                print("---",len(person_keypoints_min_array), person_keypoints_min_array)


                print("現フレーム：",person_h_array)
                print("0フレーム：",person_h_array_original)
                if len(person_h_array_original) == 0: 
                    person_h_array_original = person_h_array[:]
        

                    for d in range(len(datum.poseKeypoints)): #識別されている人数だけ回す
                        
                        start_x = person_keypoints_min_array[d][0] #person_keypoints_min_array[何人目か][xかyか]
                        start_y = person_keypoints_min_array[d][1]
                        end_x = person_keypoints_max_array[d][0] #person_keypoints_max_array[何人目か][xかyか]
                        end_y =  person_keypoints_max_array[d][1]
                        cv2.rectangle(imageToProcess,(start_x, start_y), (end_x, end_y), (person_box_array[d][0], person_box_array[d][1], person_box_array[d][2]), 5) #person_keypoints_array[何人目か][どこのキーポイントか][xかyか]
                        cv2.putText(imageToProcess, "person_" + str(d + 1), (start_x, start_y), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (person_box_array[d][0], person_box_array[d][1], person_box_array[d][2]))


                else:
                    # person_h_array_list_output =[]
                    exclusion_array = [] #すでに選ばれた人の重複を避ける
                    for d in range(len(datum.poseKeypoints)): #識別されている人数だけ回す
                        print(d + 1, "人目")
                        sim_array = [] #現フレームの一人の人物に対して、0フレーム目の全員分の類似度を追加する
                        for h in person_h_array_original: #過去の人の人数分回す
                            sim_array.append(sim_distance(person_h_array[d], h)) #h:1人分の色相データ、person_h_array_original:0フレーム目の人の色相データ
                        print(d + 1,"人目の類似度：",sim_array)
                        exclusion_judge = False
                        while exclusion_judge == False:
                            print("while")
                            sim_max_index = sim_array.index(max(sim_array)) #過去の何人目の人との類似度が一番高いか
                            # print(exclusion_array, sim_max_index)
                            # print(sim_max_index in exclusion_array)
                            if((sim_max_index in exclusion_array) == False and sim_array[sim_max_index] >= 0.7): #すでに選ばれている人物かどうか、類似度が0.8以上あるか
                                exclusion_judge = True
                                exclusion_array.append(sim_max_index)
                                print("過去の", sim_max_index + 1, "番目の人と類似度が一番近い")

                            else:
                                sim_array[sim_max_index] = -1
                                print("過去の", sim_max_index + 1, "番目の人に一番類似していますが、すでに選ばれているか類似度が一定以下です")
                                if sim_array.count(-1) == len(sim_array): #過去のどの人物とも類似していなかった場合
                                    person_h_array_original.append(person_h_array[d]) #新しい人物として追加
                                    print("過去のどの人物とも類似しませんでした。", len(person_h_array_original), "人目として登録")

                                    sim_max_index = d 
                                    break
                        
                        start_x = person_keypoints_min_array[d][0]
                        start_y = person_keypoints_min_array[d][1]
                        end_x = person_keypoints_max_array[d][0]
                        end_y =  person_keypoints_max_array[d][1]
                        cv2.rectangle(imageToProcess,(start_x, start_y), (end_x, end_y), (person_box_array[sim_max_index][0], person_box_array[sim_max_index][1], person_box_array[sim_max_index][2]), 5)
                        cv2.putText(imageToProcess, "person_" + str(sim_max_index + 1), (start_x, start_y), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (person_box_array[sim_max_index][0], person_box_array[sim_max_index][1], person_box_array[sim_max_index][2]))
                        # person_h_array_list_output.append(person_h_array[sim_max_index])
                    
                    # person_h_array_original = person_h_array_list_output[:]
                cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", imageToProcess)

                key = cv2.waitKey(1)
                key == 27
            
        else:
            break
        counter += 1


except Exception as e:
    print(e)
    sys.exit(-1)
