import numpy as np
import pandas as pd
import random


from sklearn.model_selection import train_test_split

df_mixed_data = pd.read_excel(r'penyakit_gejala_mixed.xlsx')

df_gejala = df_mixed_data['gejala']
df_jenis_penyakit = df_mixed_data['kategori']

X_train, X_test, y_train, y_test = train_test_split(
    df_gejala, df_jenis_penyakit, test_size=0.2, random_state=20)

df_train = pd.concat([y_train, X_train], axis=1)
df_test = pd.concat([y_test, X_test], axis=1)

df_train.to_csv("df_train.csv", index=False)
df_test.to_csv("df_test.csv", index=False)