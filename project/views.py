
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd

from project.preprocessing_input import text_preprocessing
from project.mnb import MultinomialNaiveBayes


def home(request):

    return render(request, 'public/home.html', context={})


def createProcess(request):
    hasil = ""
    if request.method == 'POST':

        try:
            X = pd.read_csv("project/tf_idf.csv")
            y = pd.read_csv("project/idf.csv")

            gejala = request.POST['gejala_teks']

            teks = text_preprocessing(gejala)
            print(len(teks))
            if len(teks) >= 2:

                model = MultinomialNaiveBayes(X, y)
                hasil = model.predict(teks)

                print(hasil)

                return JsonResponse({"jenis": hasil[0], "gejala": hasil[1:]})
            else:
                return JsonResponse({"error": "Gejala anda kurang jelas, tolong lebih spesifik."})
        except Exception as e:
            print("Error : ", e)
            return JsonResponse({"error": "Gejala anda kurang jelas, tolong lebih spesifik."})

    return JsonResponse({"gejala": hasil})
