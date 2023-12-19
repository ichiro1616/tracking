import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans
import math





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

sqrt_sumarray = []
sqrt_avearray = []
frame_steparray = []
step = 50 * 5
end = 100 * 5-5

for ind in range(1, input_sheet_df.shape[0], step):
# for ind in range(0, 1000, step):
    if ind+end > input_sheet_df.shape[0]:
        break
    print(ind, input_sheet_df["frame"][ind], input_sheet_df["frame"][ind + end])
    df_frame = pd.concat([input_sheet_df["frame"][ind:ind + end],input_sheet_df["frame"][ind:ind+end]], axis=0)
    df_chest =  pd.concat([input_sheet_df["胸"][ind:ind + end],input_sheet_df["胸"][ind:ind + end]], axis=0)
    df_hip =  pd.concat([input_sheet_df["腰"][ind:ind + end],input_sheet_df["腰"][ind:ind + end]], axis=0)
    df_right_sholder =  pd.concat([input_sheet_df["右肩"][ind:ind + end],input_sheet_df["右肩"][ind:ind + end]], axis=0)
    df_right_elbow = pd.concat([input_sheet_df["右ひじ"][ind:ind + end],input_sheet_df["右ひじ"][ind:ind + end]], axis=0)
    df_left_sholder = pd.concat([input_sheet_df["左肩"][ind:ind + end],input_sheet_df["左肩"][ind:ind + end]], axis=0)
    df_left_elbow = pd.concat([input_sheet_df["左ひじ"][ind:ind + end],input_sheet_df["左ひじ"][ind:ind + end]], axis=0)
    df_right_knee = pd.concat([input_sheet_df["右ひざ"][ind:ind + end],input_sheet_df["右ひざ"][ind:ind + end]], axis=0)
    df_left_knee = pd.concat([input_sheet_df["左ひざ"][ind:ind + end],input_sheet_df["左ひざ"][ind:ind + end]], axis=0)

    # print(df_right_sholder)
    X = pd.concat([df_chest, df_hip, df_right_sholder, df_right_elbow, df_left_sholder, df_left_elbow, df_right_knee, df_left_knee], axis=1)
    # X.dropna(how='any', subset=[2])
    # pd.set_option('display.max_rows', None)
    # print(X)

    sc = preprocessing.StandardScaler()
    sc.fit(X)
    X_norm = sc.transform(X)


    # クラスタリング
    cls = KMeans(n_clusters=5)
    result = cls.fit(X_norm)


    distortions = []

    for i  in range(1,11):                # 1~10クラスタまで一気に計算 
        km = KMeans(n_clusters=i,
                    init='k-means++',     # k-means++法によりクラスタ中心を選択
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X)                         # クラスタリングの計算を実行
        distortions.append(km.inertia_)   # km.fitするとkm.inertia_が得られる

    # plt.plot(range(1,11),distortions,marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Distortion')
    # plt.show()
    print(X_norm.shape)
    # 結果を出力
    # plt.scatter(X_norm[:,0],X_norm[:,1], c=result.labels_)

    centers = cls.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], s=100,
    #             facecolors='none', edgecolors='black')
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    # plt.grid()
    # print(centers)
    mean_x =0
    mean_y =0
    for j in range(len(centers)):
        mean_x += centers[j][0]
        mean_y += centers[j][1]
    mean_x = mean_x /len(centers)
    mean_y = mean_y /len(centers)
    # plt.scatter(mean_x, mean_y, s=100,
    #         facecolors='none', edgecolors='red')
    # plt.savefig("../examples/image_program_output/pattern2_chest_hip_plot_" + str(input_sheet_df["frame"][ind]) + "-" + str(input_sheet_df["frame"][ind + end]) + ".png")
    # plt.show()
    
    
    sqrt_sum =0.0
    sqrt_ave =0.0
    for k in range(len(centers)):
        sqrt_sum += math.sqrt((mean_x - centers[k][0]) ** 2 + (mean_y - centers[k][1]) ** 2)
    print("----------------------------------------------",sqrt_sum)
    sqrt_ave = sqrt_sum / len(centers)
    sqrt_sumarray.append(round(sqrt_sum, 1))
    sqrt_avearray.append(round(sqrt_ave, 1))
    print((input_sheet_df["frame"][ind] + input_sheet_df["frame"][ind + end]) / 2)
    frame_steparray.append((input_sheet_df["frame"][ind] + input_sheet_df["frame"][ind + end]) / 2)
    # frame_steparray.append((ind + (ind + end))/2)

# plt.ylim(0, 5)
# plt.grid()
plt.plot(frame_steparray,sqrt_sumarray,marker='o')
plt.xlabel('フレーム数', fontname="MS Gothic")
plt.ylabel('各クラスターの重心までの距離の総和', fontname="MS Gothic")
plt.ylim(2, 8)

plt.savefig("../examples/image_program_output/pattern1.png")

plt.show()


