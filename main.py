# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 08:47:23 2021

@author: Umut
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

#sayısal veriler için min-max normalizasyonu
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
#Model olarak SUpport Vector Regression seçildi
regr = make_pipeline(MinMaxScaler(), SVR(kernel='linear', C=15, epsilon=0.2))
#Verilen eğitim verilerine göre SVM modelini uygun hale getir
regr.fit(X_train, y_train)
#Aytılan test verisi için fiyat tahminlerini yap
y_pred = regr.predict(X_test)

from sklearn import metrics
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

