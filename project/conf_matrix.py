import pandas as pd
import numpy as np


# importing data TF IDF
df_train_tfidf = pd.read_csv("df_train_tfidf.csv")
df_train_idf = pd.read_csv("df_train_idf.csv")

# importing data target
df = pd.read_csv("df_train.csv", usecols=['kategori'])

df_jenis_penyakit = df['kategori']

# Model

class MultinomialNaiveBayes:

    def __init__(self, X, y):
        self.X = X
        self.y = y
    # Perhitungan likelihood / probability menggunakan TF ID

    def likelihood(self, nilai_tfidf_term, total_tfidf_kelas, data_total_idf):

        nilai_likelihood = 0

        nilai_likelihood = (nilai_tfidf_term + 1) / \
            (total_tfidf_kelas + data_total_idf)

        return nilai_likelihood

    def predict(self, y_test):
        # print("y_test type : ", type(y_test))
        data_likelihood_prob = [[0.0]*len(y_test)
                                for i in range(self.X.shape[0])]

        data_tfidf = self.X.values.tolist()

        data_final = [0.0]*len(data_likelihood_prob)

        total_idf = self.y["total_idf"].sum()

        nilai_prior_prob = 7/self.X.shape[0]

        for i in range(len(self.X)):

            for j in range(len(y_test)):
                # print(y_test[j])
                try:
                    nilai_tfidf_term = data_tfidf[i][self.X.columns.get_loc(
                        y_test[j])]

                except:
                    nilai_tfidf_term = 0.0

                total_tfidf_kelas = data_tfidf[i][len(data_tfidf[0])-1]

                data_likelihood_prob[i][j] = self.likelihood(
                    nilai_tfidf_term, total_tfidf_kelas, total_idf)

            data_final[i] = (
                np.prod(data_likelihood_prob[i]))*nilai_prior_prob

        max_value = max(data_final)
        # print("max value : ", max_value)
        # print(max_value)
        index_max_value = data_final.index(max_value)

        return df_jenis_penyakit.iloc[index_max_value]


def text_preprocessing(dokumen):

    lower_dokumen = dokumen.lower()

    # tokenizing
    import string
    import re  # regex
    # import word_tokenize & FreqDist from NLTK
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize

    # menghapus spesial karakter
    def remove_text_special(text):
        # menghapus tab, new line, ans back slice
        text = text.replace('\\t', " ").replace(
            '\\n', " ").replace('\\u', " ").replace('\\', "")
        # menghapus non ASCII (emote, menghapus huruf non alphabet)
        text = text.encode('ascii', 'replace').decode('ascii')
        # menghapus mention, link, hashtag, dash, kata ulang yang memiliki makna sama
        text = ' '.join(sorted(set(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)|(\b[-']\b)|[\W_]",
                                          " ", text).split())))
        return text
    a = remove_text_special(lower_dokumen)

    # menghapus angka
    def remove_number(text):
        return re.sub(r"\d+", "", text)
    b = remove_number(a)

    # menghapus tanda baca
    def remove_punctuation(text):
        return text.translate(str.maketrans("", "", string.punctuation))
    c = remove_punctuation(b)

    # menghapus whitespace yang kosong didepan kalimat
    def remove_whitespace_LT(text):
        return text.strip()
    d = remove_whitespace_LT(c)

    # mengubah double whitespace ke single whitespace
    def remove_whitespace_multiple(text):
        return re.sub('\s+', ' ', text)
    e = remove_whitespace_multiple(d)

    # menghapus single karakter
    def remove_single_char(text):
        return re.sub(r"\b[a-zA-Z]\b", "", text)
    f = remove_single_char(e)

    # word tokenizing
    def word_tokenize_wrapper(text):
        return word_tokenize(text)
    gejala_tokens = word_tokenize_wrapper(f)

    # ----------------------- stopword -------------------------------
    from nltk.corpus import stopwords
    nltk.download('stopwords')

    # get stopword indonesia
    list_stopwords = stopwords.words('indonesian')

    # ---------------------------- menambah stopword yang tidak terdapat pada file txt  ------------------------------------
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo',
                           'kalo', 'amp', 'biar', 'bikin', 'bilang',
                           'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                           'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                           'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                           'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                           '&amp', 'yah', 'ko', 'dok', 'dokter', 'halo', 'hallo', 'kabar', 'siang', 'malam', 'pagi', 'sakit'])

    # ----------------------- tambah stopword dari file txt ------------------------------------
    # baca txt stopword using pandas
    txt_stopword = pd.read_csv("stopwords.txt", names=[
                               "stopwords"], header=None)

    # konversi stopword string ke list & menambahkan stopword
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

    # konversi list ke dictionary
    list_stopwords = set(list_stopwords)

    # remove stopword pada list token
    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]
    gejala_tokens_WSW = stopwords_removal(gejala_tokens)

    # stemmer
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_list = []

    for term in gejala_tokens_WSW:
        term_list.append(stemmed_wrapper(term))

    return term_list


def confusion_matrix():

    # import data test
    df_test = pd.read_csv("df_test_text_preprocessing.csv", usecols=[
                          'kategori', 'gejala_tokens_stemmed'])
    df_test.columns = ['kategori', 'gejala']

    df_test_jenis_penyakit = df_test['kategori']
    df_test_gejala = df_test['gejala']

    # len(df_test_jenis_penyakit)

    matrix_confusion = [[0]*7 for i in range(7)]

    # print(type(df_test_gejala))
    # df_test_gejala = df_test_gejala.apply(lambda x: x.strip('()').split(','))
    for i in range(len(df_test_jenis_penyakit)):
        model = MultinomialNaiveBayes(df_train_tfidf, df_train_idf)
        y_pred = model.predict(text_preprocessing(df_test_gejala[i]))
        y_true = df_test_jenis_penyakit.iloc[i]
        # print("data ke : ", i+1, "\nGT : ", y_true, "\nPredict : ", y_pred)

        # print("y_pred : ",y_pred)
        # print("y_true : ",y_true)

        if y_pred == y_true:
            # print("data ke -", i, " > predict : ", y_pred, " > Text Processing : ", df_test_gejala[i])

            if y_pred == "infeksi":
                matrix_confusion[0][0] += 1
                # print("data ke : ", i+1, " = TP\n")

            elif y_pred == "jantung":
                matrix_confusion[1][1] += 1
                # print("data ke : ", i+1, " = TP\n")

            elif y_pred == "kanker":
                matrix_confusion[2][2] += 1
                # print("data ke : ", i+1, " = TP\n")

            elif y_pred == "kepala":
                matrix_confusion[3][3] += 1
                # print("data ke : ", i+1, " = TP\n")

            elif y_pred == "kulit dan kelamin":
                matrix_confusion[4][4] += 1
                # print("data ke : ", i+1, " = TP\n")

            elif y_pred == "pernapasan":
                matrix_confusion[5][5] += 1
                # print("data ke : ", i+1, " = TP\n")

            elif y_pred == "perut":
                matrix_confusion[6][6] += 1
                # print("data ke : ", i+1, " = TP\n")

        else:
            # print("data ke -", i, " > predict : ", y_pred, " > Text Processing : ", df_test_gejala[i])
            print("data ke -", i, " > predict : ", y_pred)
            print("data ke -", i, " > actual : ", y_true)

            if y_true == "infeksi" and y_pred == "jantung":
                matrix_confusion[0][1] += 1
            elif y_true == "infeksi" and y_pred == "kanker":
                matrix_confusion[0][2] += 1
            elif y_true == "infeksi" and y_pred == "kepala":
                matrix_confusion[0][3] += 1
            elif y_true == "infeksi" and y_pred == "kulit dan kelamin":
                matrix_confusion[0][4] += 1
            elif y_true == "infeksi" and y_pred == "pernapasan":
                matrix_confusion[0][5] += 1
            elif y_true == "infeksi" and y_pred == "perut":
                matrix_confusion[0][6] += 1

            # jantung
            elif y_true == "jantung" and y_pred == "infeksi":
                matrix_confusion[1][0] += 1
            elif y_true == "jantung" and y_pred == "kanker":
                matrix_confusion[1][2] += 1
            elif y_true == "jantung" and y_pred == "kepala":
                matrix_confusion[1][3] += 1
            elif y_true == "jantung" and y_pred == "kulit dan kelamin":
                matrix_confusion[1][4] += 1
            elif y_true == "jantung" and y_pred == "pernapasan":
                matrix_confusion[1][5] += 1
            elif y_true == "jantung" and y_pred == "perut":
                matrix_confusion[1][6] += 1

            # kanker
            elif y_true == "kanker" and y_pred == "infeksi":
                matrix_confusion[2][0] += 1
            elif y_true == "kanker" and y_pred == "jantung":
                matrix_confusion[2][1] += 1
            elif y_true == "kanker" and y_pred == "kepala":
                matrix_confusion[2][3] += 1
            elif y_true == "kanker" and y_pred == "kulit dan kelamin":
                matrix_confusion[2][4] += 1
            elif y_true == "kanker" and y_pred == "pernapasan":
                matrix_confusion[2][5] += 1
            elif y_true == "kanker" and y_pred == "perut":
                matrix_confusion[2][6] += 1

            # kepala
            elif y_true == "kepala" and y_pred == "infeksi":
                matrix_confusion[3][0] += 1
            elif y_true == "kepala" and y_pred == "jantung":
                matrix_confusion[3][1] += 1
            elif y_true == "kepala" and y_pred == "kanker":
                matrix_confusion[3][2] += 1
            elif y_true == "kepala" and y_pred == "kulit dan kelamin":
                matrix_confusion[3][4] += 1
            elif y_true == "kepala" and y_pred == "pernapasan":
                matrix_confusion[3][5] += 1
            elif y_true == "kepala" and y_pred == "perut":
                matrix_confusion[3][6] += 1

            # kulit dan kelamin
            elif y_true == "kulit dan kelamin" and y_pred == "infeksi":
                matrix_confusion[4][0] += 1
            elif y_true == "kulit dan kelamin" and y_pred == "jantung":
                matrix_confusion[4][1] += 1
            elif y_true == "kulit dan kelamin" and y_pred == "kanker":
                matrix_confusion[4][2] += 1
            elif y_true == "kulit dan kelamin" and y_pred == "kepala":
                matrix_confusion[4][3] += 1
            elif y_true == "kulit dan kelamin" and y_pred == "pernapasan":
                matrix_confusion[4][5] += 1
            elif y_true == "kulit dan kelamin" and y_pred == "perut":
                matrix_confusion[4][6] += 1

            # pernapasan
            elif y_true == "pernapasan" and y_pred == "infeksi":
                matrix_confusion[5][0] += 1
            elif y_true == "pernapasan" and y_pred == "jantung":
                matrix_confusion[5][1] += 1
            elif y_true == "pernapasan" and y_pred == "kanker":
                matrix_confusion[5][2] += 1
            elif y_true == "pernapasan" and y_pred == "kepala":
                matrix_confusion[5][3] += 1
            elif y_true == "pernapasan" and y_pred == "kulit dan kelamin":
                matrix_confusion[5][4] += 1
            elif y_true == "pernapasan" and y_pred == "perut":
                matrix_confusion[5][6] += 1

            # perut
            elif y_true == "perut" and y_pred == "infeksi":
                matrix_confusion[6][0] += 1
            elif y_true == "perut" and y_pred == "jantung":
                matrix_confusion[6][1] += 1
            elif y_true == "perut" and y_pred == "kanker":
                matrix_confusion[6][2] += 1
            elif y_true == "perut" and y_pred == "kepala":
                matrix_confusion[6][3] += 1
            elif y_true == "perut" and y_pred == "kulit dan kelamin":
                matrix_confusion[6][4] += 1
            elif y_true == "perut" and y_pred == "pernapasan":
                matrix_confusion[6][5] += 1

    # length
    # print("matrix col : ", len(matrix_confusion))
    # print("matrix rows : ", len(matrix_confusion))

    # tp
    tp_infeksi = 0.0
    tp_jantung = 0.0
    tp_kanker = 0.0
    tp_kepala = 0.0
    tp_kulit_dan_kelamin = 0.0
    tp_pernapasan = 0.0
    tp_perut = 0.0

    # tn
    tn_infeksi = 0.0
    tn_jantung = 0.0
    tn_kanker = 0.0
    tn_kepala = 0.0
    tn_kulit_dan_kelamin = 0.0
    tn_pernapasan = 0.0
    tn_perut = 0.0

    # fn
    fn_infeksi = 0.0
    fn_jantung = 0.0
    fn_kanker = 0.0
    fn_kepala = 0.0
    fn_kulit_dan_kelamin = 0.0
    fn_pernapasan = 0.0
    fn_perut = 0.0

    # fp
    fp_infeksi = 0.0
    fp_jantung = 0.0
    fp_kanker = 0.0
    fp_kepala = 0.0
    fp_kulit_dan_kelamin = 0.0
    fp_pernapasan = 0.0
    fp_perut = 0.0

    for col in range(len(matrix_confusion)):
        for rows in range(len(matrix_confusion[0])):
            # class infeksi
            if col == 0:
                if rows == 0:
                    tp_infeksi += matrix_confusion[col][rows]
                else:
                    fn_infeksi += matrix_confusion[col][rows]
            else:
                if rows == 0:
                    fp_infeksi += matrix_confusion[col][rows]
                else:
                    tn_infeksi += matrix_confusion[col][rows]

            # class jantung
            if col == 1:
                if rows == 1:
                    tp_jantung += matrix_confusion[col][rows]
                else:
                    fn_jantung += matrix_confusion[col][rows]
            else:
                if rows == 1:
                    fp_jantung += matrix_confusion[col][rows]
                else:
                    tn_jantung += matrix_confusion[col][rows]

            # class kanker
            if col == 2:
                if rows == 2:
                    tp_kanker += matrix_confusion[col][rows]
                else:
                    fn_kanker += matrix_confusion[col][rows]
            else:
                if rows == 2:
                    fp_kanker += matrix_confusion[col][rows]
                else:
                    tn_kanker += matrix_confusion[col][rows]

            # class kepala
            if col == 3:
                if rows == 3:
                    tp_kepala += matrix_confusion[col][rows]
                else:
                    fn_kepala += matrix_confusion[col][rows]
            else:
                if rows == 3:
                    fp_kepala += matrix_confusion[col][rows]
                else:
                    tn_kepala += matrix_confusion[col][rows]

            # class kulit dan kelamin
            if col == 4:
                if rows == 4:
                    tp_kulit_dan_kelamin += matrix_confusion[col][rows]
                else:
                    fn_kulit_dan_kelamin += matrix_confusion[col][rows]
            else:
                if rows == 4:
                    fp_kulit_dan_kelamin += matrix_confusion[col][rows]
                else:
                    tn_kulit_dan_kelamin += matrix_confusion[col][rows]

            # class pernapasan
            if col == 5:
                if rows == 5:
                    tp_pernapasan += matrix_confusion[col][rows]
                else:
                    fn_pernapasan += matrix_confusion[col][rows]
            else:
                if rows == 5:
                    fp_pernapasan += matrix_confusion[col][rows]
                else:
                    tn_pernapasan += matrix_confusion[col][rows]

            # class perut
            if col == 6:
                if rows == 6:
                    tp_perut += matrix_confusion[col][rows]
                else:
                    fn_perut += matrix_confusion[col][rows]
            else:
                if rows == 6:
                    fp_perut += matrix_confusion[col][rows]
                else:
                    tn_perut += matrix_confusion[col][rows]

    # total conf matrix
    tp_total = tp_infeksi + tp_jantung + tp_kanker + tp_kepala +\
        tp_kulit_dan_kelamin + tp_pernapasan + tp_perut
    tn_total = tn_infeksi + tn_jantung + tn_kanker + tn_kepala +\
        tn_kulit_dan_kelamin + tn_pernapasan + tn_perut
    fn_total = fn_infeksi + fn_jantung + fn_kanker + fn_kepala +\
        fn_kulit_dan_kelamin + fn_pernapasan + fn_perut
    fp_total = fp_infeksi + fp_jantung + fp_kanker + fp_kepala +\
        fp_kulit_dan_kelamin + fp_pernapasan + fp_perut

    # hitung accuracy
    acc = (tp_total+tn_total) / (tp_total+tn_total+fp_total+fn_total)

    # hitung precision
    precc = (tp_total)/(tp_total+fp_total)

    # hitung recall
    recc = (tp_total)/(tp_total+fn_total)

    # hitung f1-measure
    f_measure = (2*precc*recc) / (precc+recc)

    # hitung precision
    # precc = (
    #     (tp_infeksi/(tp_infeksi+fp_infeksi)) + (tp_jantung/(tp_jantung + fp_jantung)) + (tp_kanker/(tp_kanker + fp_kanker)) + (tp_kepala/(tp_kepala + fp_kepala)) +
    #     (tp_kulit_dan_kelamin/(tp_kulit_dan_kelamin+fp_kulit_dan_kelamin)) +
    #     (tp_pernapasan/(tp_pernapasan+fp_pernapasan)) +
    #     (tp_perut/(tp_perut+fp_perut))
    # )/7

    # recall = (
    #     (tp_infeksi/(tp_infeksi+fn_infeksi)) + (tp_jantung/(tp_jantung + fn_jantung)) + (tp_kanker/(tp_kanker + fn_kanker)) + (tp_kepala/(tp_kepala + fn_kepala)) +
    #     (tp_kulit_dan_kelamin/(tp_kulit_dan_kelamin+fn_kulit_dan_kelamin)) +
    #     (tp_pernapasan/(tp_pernapasan+fn_pernapasan)) +
    #     (tp_perut/(tp_perut+fn_perut))
    # )/7

    # show data
    # print("Total data : ", len(df_test_jenis_penyakit))
    print("matrix : \n",
          matrix_confusion[0], "\n",
          matrix_confusion[1], "\n",
          matrix_confusion[2], "\n",
          matrix_confusion[3], "\n",
          matrix_confusion[4], "\n",
          matrix_confusion[5], "\n",
          matrix_confusion[6]
          )
    print("TP total : ", tp_total)
    print("TN total : ", tn_total)
    print("FN total : ", fn_total)
    print("FP total : ", fp_total)

    # show measure
    print("Accuracy : ", acc)
    print("Precision : ", precc)
    print("Recall : ", recc)
    print("F1-Measure : ", f_measure)


confusion_matrix()
