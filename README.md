# name-gender-prediction-ml
Machine Learning Gender Prediction by Name

Indonesia Gender prediction using name as feature. The names trained are Indonesian names. The training data is retrieved from KPU, this data consists of names of voter and its gender.

This python program uses Scikit-Learn library for learning pipeline. It provides 3 machine learning algoriothm options (Naive Bays, Logistic Regresion, Random Forest).

This program accepts name from command line parameter and names batch in CSV file.

# Run the program

```

python gender-predictor.py -h
usage: gender-predictor.py [-h] [-ml {NB,LG,RF}] [-t TRAIN] name

Gender prediction by Indonesian name

positional arguments:
  name                  Name to predict

optional arguments:
  -h, --help            show this help message and exit
  -n nama, --name nama  The name to predict
  -ml {NB,LG,RF}        NB=Naive Bayes(default); LG=Logistic Regression;
                        RF=Random Forest
  -t TRAIN, --train TRAIN
                        Retrain using specific dataset
  -f FEATURE, --feature FEATURE
                        Feature column names for prediction
  -p PREDICT, --predict PREDICT
                        Predicted class names for prediction
  -i INPUT, --input INPUT
                        Input filename for batch prediction
  -o OUTPUT, --output OUTPUT
                        Output filename for batch prediction

```
