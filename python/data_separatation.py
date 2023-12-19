import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans
import openpyxl
import math


def write_list_2d(sheet, l_2d, start_row, start_col):
    for y, row_data in enumerate(l_2d):
        for x, cell_data in enumerate(row_data):
            sheet.cell(row=start_row + y, column=start_col + x, value=l_2d[y][x])




def sim_distance(person_now, person_previous): #0フレーム目の人と色相を比較し類似度を算出
    sum_of_sqrt = 0
    for x in range(1, len(person_now)):
        if person_now[x] != -1 or person_previous != -1:
            calculation_source =  abs(int(person_now[x]) - int(person_previous[x]))
            ring_calculate = abs(calculation_source - 180) if round((calculation_source / 180)) * 180 > 90 else calculation_source
            # print(calculation_source, round((calculation_source / 180)) * 180, ring_calculate)
            sum_of_sqrt += (ring_calculate)**2
    sum_of_sqrt = math.sqrt(sum_of_sqrt)
    bottom = math.sqrt((90)**2 * len(person_now)) #色相の差分の最大値の2乗xキーポイントの数の平方根をとっている
    print(sum_of_sqrt, 1 - sum_of_sqrt/bottom)
    return 1 - sum_of_sqrt/ bottom


#input file name
input_file_name = '../examples/gym_data.xlsx'
#xls book Open (xls, xlsxのどちらでも可能)
input_book = pd.ExcelFile(input_file_name)
#sheet_namesメソッドでExcelブック内の各シートの名前をリストで取得できる
input_sheet_name = input_book.sheet_names
#lenでシートの総数を確認
num_sheet = len(input_sheet_name)
#シートの数とシートの名前のリストの表示
print ("Sheet の数:", num_sheet)
print (input_sheet_name)
input_sheet_df = input_book.parse(input_sheet_name[0])
# print(input_sheet_df.shape[0])

row_counter = 0
# Main loop
userWantsToExit = False
using_keypointname_array =  ["胸","右肩","右ひじ","左肩","左ひじ","腰", "右ひざ","左ひざ"]

person_first_array = []
person_second_array = []
person_third_array = []
person_fourth_array = []
person_fifth_array = []
judge_cou = 1

person_number_array = [person_first_array,person_second_array,person_third_array,person_fourth_array,person_fifth_array]
person_h_array_original = []

for ind in range(0, input_sheet_df.shape[0], 5):
    person_h_array = []
    h_array = []

    print(input_sheet_df["frame"][ind], judge_cou)
    counter = 0

    for fiv in range(5):
        print(ind+counter)

        h_list_output = []
        h_list_output.append((input_sheet_df["frame"][ind + counter]))

        for keypo in range(8):
            h_list_output.append(input_sheet_df[using_keypointname_array[keypo]][ind + counter])
        h_array.append(h_list_output)
        counter+=1
    judge_cou+=1
    person_h_array = h_array[:]
    
    print("現フレーム：",person_h_array)
    print("0フレーム：",person_h_array_original)
    if len(person_h_array_original) == 0: 
        person_h_array_original = person_h_array[:]
        for num in range(len(person_number_array)):
           person_number_array[num].append(person_h_array[num]) 

    else:
        # person_h_array_list_output =[]
        exclusion_array = [] #すでに選ばれた人の重複を避ける
        for d in range(5): #識別されている人数だけ回す
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
                if((sim_max_index in exclusion_array) == False): #すでに選ばれている人物かどうか
                    exclusion_judge = True
                    exclusion_array.append(sim_max_index)
                    print("過去の", sim_max_index + 1, "番目の人と類似度が一番近い")
                    person_number_array[sim_max_index].append(person_h_array[d])

                else:
                    sim_array[sim_max_index] = -1
                    print("過去の", sim_max_index + 1, "番目の人に一番類似していますが、すでに選ばれているか類似度が一定以下です")
                    if sim_array.count(-1) == len(sim_array): #過去のどの人物とも類似していなかった場合
                        person_h_array_original.append(person_h_array[d]) #新しい人物として追加
                        print("過去のどの人物とも類似しませんでした。", len(person_h_array_original), "人目として登録")

                        sim_max_index = d 
                        break


# excel_path = r"C:\Users\isapo\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\examples\gym_data_separatation.xlsx" #デスクトップ用パス
excel_path = r"C:\Users\isapo\Documents\tracking\examples\gym_data_separatation.xlsx" #ノートPC用パス

wb = openpyxl.load_workbook(excel_path)  
sheet = wb.worksheets[0]
row_step = 0
for i in person_number_array:
    print(len(i))
    write_list_2d(sheet, i, 2,1 + row_step) #行、列
    row_step += 10

wb.save(excel_path)
wb.close()

