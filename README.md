# name-gender-prediction-ml
Machine Learning Gender Prediction by Name

Indonesia Gender prediction using name as feature. The names trained are Indonesian names. The training data is retrieved from KPU, this data consists of names of voter and its gender.

This python program uses Scikit-Learn library for learning pipeline. It provides 3 machine learning algoriothm options (Naive Bays, Logistic Regresion, Random Forest).

This program accepts name from command line parameter and names batch in CSV file.

# Run the program

```

python jenis-kelamin.py -h
usage: jenis-kelamin.py [-h] [-ml {NB,LG,RF}] [-t TRAIN] nama

Menentukan jenis kelamin berdasarkan nama Bahasa Indoensia

positional arguments:
  nama                  Nama

optional arguments:
  -h, --help            show this help message and exit
  -ml {NB,LG,RF}        NB=Naive Bayes(default); LG=Logistic Regression;
                        RF=Random Forest
  -t TRAIN, --train TRAIN
                        Training ulang dengan dataset yang ditentukan

```
