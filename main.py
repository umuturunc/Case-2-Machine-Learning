# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 08:47:23 2021

@author: Umut
"""

import pandas as pd


# VERİ OKUMA
df = pd.read_csv("voks.csv")

#VERİ ÖNİŞLEME

# 0 ile 3 arasındaki sütunlar, tahminleme için gereksiz olan değerleri içerir.
# O yüzden bu sütunlar kaldırılıyor
df.drop(df.iloc[:, 0:3], inplace = True, axis = 1)

df_col_removed = df

# 1'den fazla farklı değer içermeyen sütunlar da tahminlemede işe yaramaz.
# tek bir kategori değeri varsa zaten attribute'un
for col in df.columns:
    print(col + " " + str(len(df[col].unique())) + " unique values")
    if(len(df[col].unique())==1):
        del df_col_removed[col]

#Tarih verisinin de tahminlemede etkisinini olmadığını düşündüğüm için o sütunu attım
del df_col_removed["Tarih"]

print(len(df_col_removed.columns))


df_one_hot_encoded = df_col_removed

from pandas.api.types import CategoricalDtype 
df_col_removed["Seri"].unique()
# modelin eğitildiği veriler ve test için kullanılan veriler rastgele seçilecek
# 
df_one_hot_encoded["Seri"] = df_col_removed["Seri"].astype(CategoricalDtype(df_col_removed["Seri"].unique()))

a = pd.get_dummies(df_one_hot_encoded["Seri"],prefix="Seri")

df_one_hot_encoded = pd.concat((df_one_hot_encoded,pd.get_dummies(df_one_hot_encoded["Seri"],prefix="Seri",dummy_na=True)),axis = 1)

labels_one_hot_encoded = {"Seri","Model","Yıl","Renk","Vites","Yakıt","Şehir"}

del df_one_hot_encoded["Seri"]  
