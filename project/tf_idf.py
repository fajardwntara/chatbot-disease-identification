import pandas as pd 
import numpy as np
import random
from random import randrange
from sklearn.model_selection import train_test_split

# df = pd.read_csv("text_preprocessing.csv", usecols=["penyakit", "gejala_tokens_stemmed"])
df = pd.read_csv("text_preprocessing.csv", usecols=["kategori", "gejala_tokens_stemmed"])
df.columns = ["kategori", "gejala"]

import ast

def convert_text_list(texts):
    texts = ast.literal_eval(texts)
    return [text for text in texts]

df["gejala_list"] = df["gejala"].apply(convert_text_list)

def calc_TF(document):
    # menghitung banyaknya jumlah kata yang muncul pada pertanyaan
    TF_dict = {}
    for term in document:
        if term in TF_dict:
            TF_dict[term] += 1
        else:
            TF_dict[term] = 1
    # menghitung TF tiap kata
    for term in TF_dict:
        TF_dict[term] = TF_dict[term] / len(document)
    return TF_dict

df["TF_dict"] = df['gejala_list'].apply(calc_TF)


# Check TF result
# index = 585
# print('%20s' % "term", "\t", "TF\n")
# for key in df["TF_dict"][index]:
#     print('%20s' % key, "\t", df["TF_dict"][index][key])

def calc_DF(tfDict):
    count_DF = {}
    # Run through each document's tf dictionary and increment countDict's (term, doc) pair
    for document in tfDict:
        for term in document:
            if term in count_DF:
                count_DF[term] += 1
            else:
                count_DF[term] = 1
    return count_DF

DF = calc_DF(df["TF_dict"])
print("DF : " , DF)
# print("log 6/ 2 : " , np.log10(6/2))

n_document = len(df)

def calc_IDF(__n_document, __DF):
    IDF_Dict = {}
    for term in __DF:
        IDF_Dict[term] = np.log10(__n_document / (__DF[term] + 1))
    return IDF_Dict

#Stores the idf dictionary
IDF = calc_IDF(n_document, DF)

# print("IDF : " ,IDF)
#calc TF-IDF
def calc_TF_IDF(TF):
    TF_IDF_Dict = {}
    
    #For each word in the review, we multiply its tf and its idf.
    for key in TF:
        TF_IDF_Dict[key] = TF[key] * IDF[key]
    
    return TF_IDF_Dict

def total_idf(TF):
    IDF_Dict = {}
    #For each word in the review, we multiply its tf and its idf.
    for key in TF:
        IDF_Dict[key] = IDF[key]
    return IDF_Dict

#Stores the TF-IDF Series
df["TF-IDF_dict"] = df["TF_dict"].apply(calc_TF_IDF)
df["IDF_dict"] = df["TF_dict"].apply(total_idf)


# #Check TF-IDF result
# index = 5
# print('%20s' % "term", "\t", '%10s' % "TF", "\t", '%20s' % "TF-IDF\n")
# for key in df["TF-IDF_dict"][index]:
#     print('%20s' % key, "\t", df["TF_dict"][index][key] ,"\t" , df["TF-IDF_dict"][index][key])


# sort descending by value for DF dictionary 
sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)

# Create a list of unique words from sorted dictionay `sorted_DF`
unique_term = [item[0] for item in sorted_DF]
unique_term.append("total_idf")


def calc_TF_IDF_Vec(__TF_IDF_Dict):
    TF_IDF_vector = [0.0] * len(unique_term)
    sum_tf_idf = 0.0
    # For each unique word, if it is in the review, store its TF-IDF value.
    for i, term in enumerate(unique_term):
        if term in __TF_IDF_Dict:
            TF_IDF_vector[i] = __TF_IDF_Dict[term]
            # print("TF_IDF_vector : ", TF_IDF_vector[i])
            sum_tf_idf += TF_IDF_vector[i]
        TF_IDF_vector[len(unique_term)-1] = sum_tf_idf
    # print(TF_IDF_vector)
    return TF_IDF_vector

def calc_IDF_Vec(_IDF_Dict):
    IDF_vector = [0.0] * len(unique_term)
    sum_idf = 0.0
    # For each unique word, if it is in the review, store its TF-IDF value.
    for i, term in enumerate(unique_term):
        if term in _IDF_Dict:
            IDF_vector[i] = _IDF_Dict[term]
            sum_idf += IDF_vector[i]
        IDF_vector[len(unique_term)-1] = sum_idf
    return IDF_vector

df["TF_IDF_Vec"] = df["TF-IDF_dict"].apply(calc_TF_IDF_Vec)
df["IDF_Vec"] = df["IDF_dict"].apply(calc_IDF_Vec)


# Convert Series to List
TF_IDF_Vec_List = np.array(df["TF_IDF_Vec"].to_list())
IDF_Vec_List = np.array(df["IDF_Vec"].to_list())

tf_idf = pd.DataFrame(TF_IDF_Vec_List, columns=[unique_term])
idf = pd.DataFrame(IDF_Vec_List, columns=[unique_term])

# Save tf-idf 
# idf.to_csv("df_train_idf.csv", index=False)
# tf_idf.to_csv("df_train_tfidf.csv", index=False)

