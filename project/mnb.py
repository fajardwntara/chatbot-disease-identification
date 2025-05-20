# from preprocessing_input import text_preprocessing
import pandas as pd
import numpy as np
import itertools

# importing data
df = pd.read_excel(
    r"project/penyakit_gejala_mixed.xlsx", usecols=['kategori', 'penyakit'])

df_penyakit = df['penyakit']
df_jenis_penyakit = df['kategori']


class MultinomialNaiveBayes:

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def likelihood(self, nilai_tfidf_term, total_tfidf_kelas, data_total_idf):

        nilai_likelihood = 0

        nilai_likelihood = (nilai_tfidf_term + 1) / \
            (total_tfidf_kelas + data_total_idf)
        # print("pembilang:", nilai_tfidf_term + 1)
        # print("pembagi : ", total_tfidf_kelas + data_total_idf)

        return nilai_likelihood

    def predict(self, y_test):
        data_likelihood_prob = [[0.0]*len(y_test)
                                for i in range(self.X.shape[0])]

        data_tfidf = self.X.values.tolist()
        data_final = [0.0]*len(data_likelihood_prob)

        total_idf = self.y["total_idf"].sum()
        # print("total idf : ", total_idf)

        nilai_prior_prob = 7/self.X.shape[0]

        data_penyakit = list(df_penyakit.values)
        data_jenis_penyakit = list(df_jenis_penyakit.values)

        # print("data penyakit : ", data_penyakit)

        for i in range(len(self.X)):

            for j in range(len(y_test)):
               
                nilai_tfidf_term = data_tfidf[i][self.X.columns.get_loc(
                    y_test[j])]

                # print("term : ", y_test[j])
                # print("nilai_tfidf_term : ", nilai_tfidf_term)

                total_tfidf_kelas = data_tfidf[i][len(data_tfidf[0])-1]

                data_likelihood_prob[i][j] = self.likelihood(
                    nilai_tfidf_term, total_tfidf_kelas, total_idf)

                # print(data_likelihood_prob[i][j])

            # print("total_idf", total_idf)
            # print("data_likelihood_prob : ", data_likelihood_prob[i])
            data_final[i] = (np.prod(data_likelihood_prob[i]))*nilai_prior_prob

        # print("data final :", data_final)
        data_combine = [[t, x, y]
                        for t, x, y in zip(data_jenis_penyakit, data_penyakit, data_final)]

        # print(data_combine[0])
        # df_scores = pd.DataFrame(data_final, columns="nilai_mnb")
        # print(df_scores.to_string())

        data_combine_sort = sorted(
            data_combine, key=lambda l: l[2], reverse=True)

        get_nama_penyakit = [x for x in data_combine_sort]
        print("get_nama_penyakit", get_nama_penyakit[3][1])
        # sort_data_final = sorted(data_final,reverse=True)
        # print('get_nama_penyakit : ', get_nama_penyakit)
        # print('get_nama_penyakit idx : ', get_nama_penyakit[3][0])
        max_value = max(data_final)
        print(max_value)
        index_max_value = data_final.index(max_value)
        # print("index_max_value", index_max_value)
        # print("kategori :", df_jenis_penyakit.iloc[index_max_value])
        # print("data get_nama_penyakit : ", get_nama_penyakit)
        df_sort_penyakit = []

        def sorted_disease(nama_kategori):
            for x in range(len(get_nama_penyakit)):
                if nama_kategori == get_nama_penyakit[x][0]:
                    df_sort_penyakit.append(get_nama_penyakit[x][1])

        sorted_disease(df_jenis_penyakit.iloc[index_max_value])

        final_result = []
        final_result.append(df_jenis_penyakit.iloc[index_max_value])
        final_result.append(df_sort_penyakit[:5])
        print("get_nama_penyakit_fix : ", df_sort_penyakit[:5])

        return final_result
