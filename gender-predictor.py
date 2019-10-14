import sys, argparse, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import json

# main
def main(args):
    if args.feature is None :
       args.feature = "name"
       
    if args.predict is None :
       args.predict = "gender"

    if args.input is None :

        result = predict(args)
        
        print ("Gender prediction using ", result[1], ".")
        print(args.name, ' : ', result_to_label(result[0]))
           
    else:
        s = "name,gender\r\n"
        inputfile = args.input
        df = pd.read_csv(inputfile, encoding = 'utf-8-sig')
        df = df.dropna(how='all')
        names = df[args.feature].values

        for name in names:
           args.name = name.replace("\n", "")
           result = predict(args)
           s = s + args.name + "," + result_to_label(result[0]) + "\r\n"
           
       
        f= open(args.output,"w+")
        f.write(s)
        f.close()

# change result to label
def result_to_label(result):
    jk_label = {1:"Male", 0:"Female"}
    r = jk_label[result]
    return r
    
    
# predict
def predict(args):
    if(args.ml == 'LG'):
       result = predict_lg(args.name, args.train, args.feature, args.predict)
       ml_type = 'Logistic Regression'
    elif(args.ml == 'RF'):
       result = predict_rf(args.name, args.train, args.feature, args.predict)
       ml_type = 'Random Forest'
    else:
       result = predict_nb(args.name, args.train, args.feature, args.predict)
       ml_type = 'Naive Bayes'
    
    arr = [result, ml_type]
    return arr

# load dataset
def load_data(dataset="./data/data-pemilih-kpu.csv", feature_colnames="name", predicted_colnames="gender"):
    df = pd.read_csv(dataset, encoding = 'utf-8-sig')
    df = df.dropna(how='all')
    
    jk_map = {"Male" : 1, "Female" : 0}
    df["gender"] = df["gender"].map(jk_map)

    feature_col_names = [feature_colnames]
    predicted_class_names = [predicted_colnames]
    X = df[feature_col_names].values     
    y = df[predicted_class_names].values 
    
    #split train:test data 70:30
    split_test_size = 0.30
    text_train, text_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, stratify=y, random_state=42) 
    
    return (text_train, text_test, y_train, y_test)

# Naive Bayes implementation
def predict_nb(name, dataset, feature_colnames, predicted_colnames):
    if os.path.isfile("./data/pipe_nb.pkl") and dataset is None:        
        file_nb = open('./data/pipe_nb.pkl', 'rb')
        pipe_nb = pickle.load(file_nb)
    else:
        file_nb = open('./data/pipe_nb.pkl', 'wb')
        pipe_nb = Pipeline([('vect', CountVectorizer(analyzer = 'char_wb', ngram_range=(2,6))),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultinomialNB())])       
        #train and dump to file                     
        dataset = load_data(dataset)
        pipe_nb = pipe_nb.fit(dataset[0].ravel(), dataset[2].ravel())
        pickle.dump(pipe_nb, file_nb)
        
        #Akurasi
        predicted = pipe_nb.predict(dataset[1].ravel())
        Akurasi = np.mean(predicted == dataset[3].ravel())*100
        print("Accuracy :", Akurasi, "%")
    
    allnameresult = pipe_lg.predict([name])[0]
    names = name.split(' ')
    l = len(names)
    if l > 1:
        lastname = names[l - 1]
        reslastname = pipe_lg.predict([lastname])[0]
        names.pop(l - 1)
        firstnames = "".join(names)
        resfirstnames = pipe_lg.predict([firstnames])[0]
    	
        if reslastname == 1 and resfirstnames == 0:
           return resfirstnames
        else:
           return allnameresult
    else:
        return allnameresult

# Logistic Regression implementation
def predict_lg(name, dataset, feature_colnames, predicted_colnames):
    if os.path.isfile("./data/pipe_lg.pkl") and dataset is None:        
        file_lg = open('./data/pipe_lg.pkl', 'rb')
        pipe_lg = pickle.load(file_lg)
    else:
        file_lg = open('./data/pipe_lg.pkl', 'wb')
        pipe_lg = Pipeline([('vect', CountVectorizer(analyzer = 'char_wb', ngram_range=(2,6))),
                            ('tfidf', TfidfTransformer()),
                            ('clf', LogisticRegression())])        
        dataset = load_data(dataset)
        pipe_lg = pipe_lg.fit(dataset[0].ravel(), dataset[2].ravel())
        pickle.dump(pipe_lg, file_lg)

        #Akurasi
        predicted = pipe_lg.predict(dataset[1].ravel())
        Akurasi = np.mean(predicted == dataset[3].ravel())*100
        print("Accuracy :", Akurasi, "%")
    
    allnameresult = pipe_lg.predict([name])[0]
    names = name.split(' ')
    l = len(names)
    if l > 1:
        lastname = names[l - 1]
        reslastname = pipe_lg.predict([lastname])[0]
        names.pop(l - 1)
        firstnames = "".join(names)
        resfirstnames = pipe_lg.predict([firstnames])[0]
    	
        if reslastname == 1 and resfirstnames == 0:
           return resfirstnames
        else:
           return allnameresult
    else:
        return allnameresult
    		

# Random Forest implementation
def predict_rf(name, dataset, feature_colnames, predicted_colnames):
    if os.path.isfile("./data/pipe_rf.pkl") and dataset is None:         
        file_rf = open('./data/pipe_rf.pkl', 'rb')
        pipe_rf = pickle.load(file_rf)
    else:
        file_rf = open('./data/pipe_rf.pkl', 'wb')
        pipe_rf = Pipeline([('vect', CountVectorizer(analyzer = 'char_wb', ngram_range=(2,6))),
                            ('tfidf', TfidfTransformer()),
                            ('clf', RandomForestClassifier(n_estimators=10, n_jobs=-1))])        
        dataset = load_data(dataset)
        pipe_rf = pipe_rf.fit(dataset[0].ravel(), dataset[2].ravel())
        pickle.dump(pipe_rf, file_rf)

        #Akurasi
        predicted = pipe_rf.predict(dataset[1].ravel())
        Akurasi = np.mean(predicted == dataset[3].ravel())*100
        print("Accuracy :", Akurasi, "%")
    
    allnameresult = pipe_lg.predict([name])[0]
    names = name.split(' ')
    l = len(names)
    if l > 1:
        lastname = names[l - 1]
        reslastname = pipe_lg.predict([lastname])[0]
        names.pop(l - 1)
        firstnames = "".join(names)
        resfirstnames = pipe_lg.predict([firstnames])[0]
    	
        if reslastname == 1 and resfirstnames == 0:
           return resfirstnames
        else:
           return allnameresult
    else:
        return allnameresult

# args setting
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "Gender prediction by Indonesian name")
 
  parser.add_argument(
                      "-n",
                      "--name",
                      help = "The name to predict",
                      metavar='nama'
                      )
  parser.add_argument(
                      "-ml",
                      help = "NB=Naive Bayes(default); LG=Logistic Regression; RF=Random Forest",
                      choices=["NB", "LG", "RF"]
                      )
  parser.add_argument(
                      "-t",
                      "--train",
                      help="Retrain using specific dataset")
  parser.add_argument(
                      "-f",
                      "--feature",
                      help = "Feature column names "
                      )
  parser.add_argument(
                      "-p",
                      "--predict",
                      help = "Predicted class names"
                      )
  parser.add_argument(
                      "-i",
                      "--input",
                      help = "Input filename for batch prediction"
                      )
  parser.add_argument(
                      "-o",
                      "--output",
                      help = "Output filename for batch prediction"
                      )
  args = parser.parse_args()
  
  main(args)