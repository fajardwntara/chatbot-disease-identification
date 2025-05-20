import numpy as np
import pandas as pd
import random

# import data excel
# df = pd.read_excel(r'penyakit_gejala_mixed.xlsx')
df = pd.read_csv("df_test.csv")
print(df['gejala'])
# print(df)
# case folding 

df['gejala'] = df['gejala'].str.lower()


# print("Case folding : \n ")
# print(type(df['gejala']))

# tokenizing
import string 
import re #regex
# import word_tokenize & FreqDist from NLTK
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

# menghapus spesial karakter
def remove_text_special(text):
    # menghapus tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # menghapus non ASCII (emote, menghapus huruf non alphabet)
    text = text.encode('ascii', 'replace').decode('ascii')
    # menghapus mention, link, hashtag, dash, kata ulang yang memiliki makna sama
    text = ' '.join(sorted(set(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)|(\b[-']\b)|[\W_]",
    " ", text).split())))
    return text    
    
df['gejala'] = df['gejala'].apply(remove_text_special)

# menghapus angka
def remove_number(text):
    return  re.sub(r"\d+", "", text)
df['gejala'] = df['gejala'].apply(remove_number)

# menghapus tanda baca
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))
df['gejala'] = df['gejala'].apply(remove_punctuation)

# menghapus whitespace yang kosong didepan kalimat
def remove_whitespace_LT(text):
    return text.strip()
df['gejala'] = df['gejala'].apply(remove_whitespace_LT)

# mengubah double whitespace ke single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)
df['gejala'] = df['gejala'].apply(remove_whitespace_multiple)

# menghapus single karakter
def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)
df['gejala'] = df['gejala'].apply(remove_single_char)

# word tokenizing 
def word_tokenize_wrapper(text):
    return word_tokenize(text)
df['gejala_tokens'] = df['gejala'].apply(word_tokenize_wrapper)

# print('Tokenizing Result : \n') 
# print(df['gejala_tokens'])
# print('\n\n\n')

# NLTK calc frequency distribution
def freqDist_wrapper(text):
    return FreqDist(text)

df['gejala_tokens_fdist'] = df['gejala_tokens'].apply(freqDist_wrapper)

# print('Frequency Tokens : \n') 
# print(df['gejala_tokens_fdist'].head().apply(lambda x : x.most_common()))

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
                        '&amp', 'yah', 'ko', 'dok', 'dokter', 'sakit'])

# ----------------------- tambah stopword dari file txt ------------------------------------
# baca txt stopword menggunakan pandas
txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)

# konversi stopword string ke list & menambahkan stopword
list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

# konversi list ke dictionary
list_stopwords = set(list_stopwords)

#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

df['gejala_tokens_WSW'] = df['gejala_tokens'].apply(stopwords_removal) 

# print(df['gejala_tokens_WSW'])

# # normalization
# normalizad_word = pd.read_excel("normalization.xlsx")

# normalizad_word_dict = {}

# for index, row in normalizad_word.iterrows():
#     if row[0] not in normalizad_word_dict:
#         normalizad_word_dict[row[0]] = row[1] 

# def normalized_term(document):
#     return [normalizad_word_dict[term] if term in normalizad_word_dict 
#     else term for term in document]

# df['question_normalized'] = df['question_tokens_WSW'].apply(normalized_term)

# df['question_normalized'].head(10)


# stemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)


term_dict = {}

for document in df["gejala_tokens_WSW"]:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    # print(term,":" ,term_dict[term])
    

# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

df['gejala_tokens_stemmed'] = df['gejala_tokens_WSW'].swifter.apply(get_stemmed_term)

# print(df['gejala_tokens_stemmed'])


# # save to csv
df.to_csv("df_test_text_preprocessing.csv")

# def data_test_preprocessing(dokumen):
#     remove_text_special(dokumen)
#     remove_number(dokumen)
#     remove_punctuation(dokumen)
#     remove_whitespace_LT(dokumen)
#     remove_whitespace_multiple(dokumen)
#     remove_single_char(dokumen)
#     word_tokenize_wrapper(dokumen)
#     freqDist_wrapper(dokumen)

