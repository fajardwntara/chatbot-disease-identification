import numpy as np
import pandas as pd
import random

def text_preprocessing(dokumen):
        
    lower_dokumen = dokumen.lower()

    # tokenizing
    import string 
    import re #regex
    # import word_tokenize & FreqDist from NLTK
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize 

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
    a = remove_text_special(lower_dokumen)

    # menghapus angka
    def remove_number(text):
        return  re.sub(r"\d+", "", text)
    b = remove_number(a)

    # menghapus tanda baca
    def remove_punctuation(text):
        return text.translate(str.maketrans("","",string.punctuation))
    c = remove_punctuation(b)

    # menghapus whitespace yang kosong didepan kalimat
    def remove_whitespace_LT(text):
        return text.strip()
    d = remove_whitespace_LT(c)

    # mengubah double whitespace ke single whitespace
    def remove_whitespace_multiple(text):
        return re.sub('\s+',' ',text)
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
    txt_stopword = pd.read_csv("project/stopwords.txt", names= ["stopwords"], header = None)

    # konversi stopword string ke list & menambahkan stopword
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

    # konversi list ke dictionary
    list_stopwords = set(list_stopwords)

    #remove stopword pada list token
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
