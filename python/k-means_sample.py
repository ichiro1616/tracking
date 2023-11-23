import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans
import math



# datasetの読み込み
wine_data = datasets.load_wine()



# # # DataFrameに変換
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
pd.set_option('display.max_rows', None)

print(df[["alcohol", "color_intensity"]])
X = df[["alcohol","color_intensity"]]



sc = preprocessing.StandardScaler()
sc.fit(X)
X_norm = sc.transform(X)

# クラスタリング
cls = KMeans(n_clusters=3)
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

plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# # 結果を出力
plt.scatter(X_norm[:,0],X_norm[:,1], c=result.labels_)

plt.show()


# sqrt_sum =0.0
# sqrt_ave =0.0
# for k in range(len(centers)):
#     sqrt_sum += math.sqrt((mean_x - centers[k][0]) ** 2 + (mean_y - centers[k][1]) ** 2)
# print("----------------------------------------------",sqrt_sum)
# sqrt_ave = sqrt_sum / len(centers)
# sqrt_sumarray.append(round(sqrt_sum, 1))
# sqrt_avearray.append(round(sqrt_ave, 1))

# frame_steparray.append(str(ind) + "~" + str(ind + step))
# plt.plot(frame_steparray,sqrt_sumarray,marker='o')
# plt.xlabel('frame_step' + str(step))
# plt.ylabel('sqrt_sum')
# plt.show()


