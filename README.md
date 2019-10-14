# Indonesian name-gender-prediction-ml
Machine Learning Gender Prediction by Name

Indonesia Gender prediction using name as feature. The names trained are Indonesian names. The training data is retrieved from KPU, this data consists of names of voter and its gender.

This python program uses Scikit-Learn library for learning pipeline. It provides 3 machine learning algoriothm options (Naive Bays, Logistic Regresion, Random Forest).

This program accepts name from command line parameter and names batch in CSV file.

# Setup the environment
1. Clone this repository git clone git@github.com:miftahul-huda/name-gender-prediction-ml.git
2. Open project directory : cd name-gender-prediction-ml
3. Create Python virtual environment : python3 -m venv venv
4. Activate the virtual environment : source venv/bin/activate
5. Install dependency : pip3 install -r requirements.txt

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
Predict gender for a name

```
python gender-predictor.py -ml LG -n "Miftahul Huda"
Gender prediction using  Logistic Regression .
Miftahul Huda  :  Male

```
Predict gender using batch

```
python gender-predictor.py -ml LG -i ./data/voters.DKIJAKARTA.JAKARTASELATAN.PANCORAN.CIKOKO.1.csv -o ./data/output_test.csv
```
Retrain the model and predict
```
python gender-predictor.py -ml LG -t ./data/data-pemilih-kpu.csv -f "name" -p "gender"  -n "Amanda Husein"
Accuracy : 93.51351351351352 %
Gender prediction using  Logistic Regression .
Amanda Husein  :  Female

```

