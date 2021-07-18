# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 08:47:23 2021

@author: Umut
"""

"""
Faktör analizi benzeri bir yöntem kullanarak boyut indirgeme yapmak istedim.
Ancak kategorik veriler ve sayısal veriler arasındaki korelasyon nasıl hesaplanır bulamadım.
Sadece kategorik veriler veya sadece sayısal veriler arasındaki korelasyon hesabının yetersiz olacağını düşündüm.
"""

import numpy as np
import pandas as pd

#Verileri oku
df = pd.read_csv('voks.csv')
#İlk 4 sütun, tahminleme için gereksiz
#İlk 4 sütın çıkarıldı
df = df.iloc[:,4:]
#Aynı şekilde Tarih bilgisi de tahminleme için gereksiz olduğu için atıldı
df.drop(['Tarih'], inplace=True, axis=1)

#One Hot Encoding yapılacak, kategorik değişkenlere sahip sütunlar
labels_one_hot_encoded = ["Seri","Model","Renk","Vites","Yakıt","Sehir"]

#Dummy coding yapılacak sütunları gez
for label in labels_one_hot_encoded:
    #DataFrame'e ilgili sütunun dummy sütun değerlerini ekle
    df = pd.concat([df, pd.get_dummies(df[label])], axis=1)
    #İlgili sütunun kendisini çıkartabiliriz
    df.drop([label], inplace=True, axis=1)

#Fiyat sütunundaki string değişkenler float türüne dönüştürülüyor
df['Fiyat'] = df['Fiyat'].apply(lambda x:float(x.replace(' TL', '')))

#Modelin eğitimi ve tahminlemede kullanılacak sütunlar X'de tutuluyor
X = df.drop(['Fiyat'], axis=1).values

#Tahmin edilecek Fiyat değişkeni ayıklandı
y = df.loc[:, 'Fiyat'].values

#X--> Tahminleme yapılırken kullanılacak değerler
#Gerçek Fiyat değerleri


#Null verileri, bulunduğu sütunun ortalama değeri ile değiştir
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#İlk iki sütunda sayısal veriler var
imputer.fit(X[:, :2])
X[:, :2] = imputer.transform(X[:, :2])

#Verileri %75 eğitim, %25 test olarak ayır
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

modelSVR = SVR()
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['linear','poly']}  
grid = GridSearchCV(modelSVR, param_grid, refit = True, verbose = 3,n_jobs=-1) 
# Grid Serach için modelin uyumlu hale getirilmesi
grid.fit(X_train, y_train) 
# Tuning'den sonra en iyi parametlerleri yazdır
print(grid.best_params_) 
#Test verisi tahminlemesi
grid_predictions = grid.predict(X_test) 


#MSE ve RMSE değerleri yazdırılıyor
from sklearn import metrics
print('MSE: ', metrics.mean_squared_error(y_test, grid_predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, grid_predictions)))

#Modeli kaydetmek ve yüklemek için
import joblib
#Modeli diske kaydet
joblib.dump(grid.best_estimator_, 'model.pkl', compress = 1)

#Diskteki modeli yükle
grid = joblib.load('model.pkl')

