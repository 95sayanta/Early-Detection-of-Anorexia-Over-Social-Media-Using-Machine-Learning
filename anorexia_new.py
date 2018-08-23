#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:47:32 2018

@author: sayanta
"""

import os
import pandas as pd
import numpy as np
from scipy import sparse
import xml.etree.ElementTree as ET
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import grid_search
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest, SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostClassifier




def main():
    prefix='/home/sayanta/eRisk2018/eRisk 2018_anorexia_training'
    anorexia =[] ##empty list
    anorexia_test12345678910 = []
      
    pos_folder='%s/selected_pos' % prefix
    neg_folder='%s/selected_neg' % prefix
    test12345678910_folder = '%s/test_12345678910' % prefix
    
    listfiles=os.listdir(pos_folder)
    listfiles2=os.listdir(neg_folder)
    listfiles3 = os.listdir(test12345678910_folder)
    
    anoPos = []  
    a=[]
    idp = []
    for file in listfiles:
      filepath=os.path.join(pos_folder,file)  #print(filepath)
      tree = ET.parse(filepath)
      root = tree.getroot() #prints ID
      pos_txt='' 
      for child in root:
          if child.tag!='ID':
              pos_txt= pos_txt+child[0].text+child[3].text     
      anoPos.append(pos_txt)  
      a.append(root[0].text+'\n'+pos_txt)
      idp.append(root[0].text)
    
    anoPosWithClassLabels = []
    for i in anoPos:
        row = []
        row.append(i)
        row.append(1)
        anoPosWithClassLabels.append(row)
    
    anoNeg =[]
    b=[]
    idn=[]
    for file in listfiles2:
      filepath=os.path.join(neg_folder,file)
      try:
        tree = ET.parse(filepath)
      except Exception:
        continue
      root = tree.getroot()
      neg_txt='' 
      for child in root:
        if child.tag!='ID':
           neg_txt= neg_txt+child[0].text+child[3].text
      anoNeg.append(neg_txt)
      b.append(root[0].text+'\n'+pos_txt)
      idn.append(root[0].text)
    
    anoNegWithClassLabels = []
    for i in anoNeg:
        row = []
        row.append(i)
        row.append(2)
        anoNegWithClassLabels.append(row)
        
       
        anorexia = anoPosWithClassLabels + anoNegWithClassLabels
        anorexia = pd.DataFrame(anorexia)
        #anorx=a+b
          
    test12345678910 = []  
    c=[]
    idt=[]
    for file in listfiles3:
      filepath=os.path.join(test12345678910_folder,file)  #print(filepath)
      tree = ET.parse(filepath)
      root = tree.getroot() #prints ID
      test_txt='' 
      for child in root:
          if child.tag!='ID':
              test_txt= test_txt+child[0].text+child[3].text     
      test12345678910.append(test_txt)  
      c.append([[root[0].text],[test_txt]])
      idt.append(root[0].text) 
    #zer=pd.DataFrame(np.zeros((960,1)))
        
    anorexia = anoPosWithClassLabels + anoNegWithClassLabels
    anorexia = pd.DataFrame(anorexia)
    anorexia_test12345678910 = pd.DataFrame(c)
    anorexia_train_data = list(anorexia[0])
    anorexia_train_class = list(anorexia[1])
    anorexia_test_data = list(anorexia_test12345678910[1])
    anorexia_test_subject_names = list(anorexia_test12345678910[0])
    def idf_text(doc):
        subject_id,text = doc[0][0],doc[1][0]   
        return subject_id,text
split_data = sorted(map(idf_text,c))
split_data = pd.DataFrame(split_data)
type(split_data)  
data = split_data[1]
data = pd.DataFrame(data)
data.to_csv("anorexia_test2.csv")# index=False,index_label=False)
    #data=anorexia_train_data + anorexia_test_data
    #data = pd.DataFrame(data)
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)
    anorexia_train_data = pd.DataFrame(anorexia_train_data)
    anorexia_test_data = pd.DataFrame(anorexia_test_data)
    tfidf = vectorizer.fit_transform([item[0] for item in anorexia_train_data.values.tolist()])
    tfidf1 = vectorizer.transform([item[0] for item in anorexia_test_data.values.tolist()])
    term_freq1=tfidf.toarray()
    term_freq2=tfidf1.toarray()
    tfidf_trn=term_freq1[0:len(anorexia_train_data)]
    tfidf_tst=term_freq2[0:len(anorexia_test_data)]
    final_tfidf_trn=sparse.csr_matrix(tfidf_trn)        
    final_tfidf_tst=sparse.csr_matrix(tfidf_tst)
    
        
            
    print(" 1.SVM \n 2.Multinomial Naive Bayse \n 3.Logistic Regression \n 4.Multilayer Perceptron\n 5.Random Forest\n 6.AdaBoostClassifier")
    choice = input(" Enter your Choice : ")
    usr_choice(choice,final_tfidf_trn,final_tfidf_tst,anorexia_test_subject_names,anorexia_train_class)
            
    return;

def OUTPUT(predicted,anorexia_test_subject_names):
    file = open("RKMVERI_13.txt",'w')
    for i in range(len(anorexia_test_subject_names)):
        anorexia_test_subject_names = pd.DataFrame(anorexia_test_subject_names)
        file.write(str(anorexia_test_subject_names[0].values[i])+'\t\t'+str(predicted[i])+'\n')
    file.close()
    return;
 
def GridSearch(final_tfidf_trn,final_tfidf_tst,anorexia_train_class,names,parameters,clf):
    pipeline2 = Pipeline([   
        ('feature_selection', SelectKBest(chi2, k= 2000)),    
        ('clf', clf),
    ])
    grid = grid_search.GridSearchCV(pipeline2,parameters,cv=10)          
    grid.fit(final_tfidf_trn,anorexia_train_class)    
    clf = grid.best_estimator_                   # Best grid
    print('\n The best grid is as follows: \n')
    print(grid.best_estimator_)
    # Classification of the test samples
    predicted = clf.predict(final_tfidf_tst)     
    predicted = list(predicted)
    OUTPUT(predicted,names)
    return;

def usr_choice(choice,final_tfidf_trn,final_tfidf_tst,names,anorexia_train_class):
   if choice=="1":
        print("SVM is running...")
        clf = svm.LinearSVC(class_weight='balanced', random_state=None, loss='hinge',penalty='l2')  
        parameters = {
        'clf__C':(0.1,0.5,0.9,1,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10),
        }
   elif choice=="2":
        print("NaiveBayes is running...")
        clf = MultinomialNB()  
        parameters = {
       'clf__alpha':(0,1),
        }       
   elif choice=="3":
        print("Logistic Regression is running...")
        clf = LogisticRegression(solver='liblinear',class_weight='balanced', penalty='l1') 
        parameters = {
        'clf__random_state':(0,10),
        }
   elif choice=="4":
        print("MLP is running...")
        clf = MLPClassifier(alpha=0.0001,hidden_layer_sizes=(5, 2), random_state=1)
        parameters = {
        'clf__solver':('lbfgs','sgd','adam'),        
        'clf__activation':('identity', 'logistic', 'tanh', 'relu'),
        'clf__random_state':(0,1,4,7,9,10),
        }
   elif choice=="5":
        print("Random Forest is running...")
        clf = RandomForestClassifier(criterion='gini',max_features='auto',class_weight='balanced')
        parameters = {
        'clf__n_estimators':(100,200,500),
        'clf__max_depth':(10,20),
        } 
   elif choice=="6":
        print("AdaBoostClassifier is running...")
        clf = AdaBoostClassifier(base_estimator=None, random_state=None)
        parameters = {
        'clf__n_estimators':(50,100),
        }
   else:
        print("Wrong choice. Try Again Later")
   GridSearch(final_tfidf_trn,final_tfidf_tst,anorexia_train_class,names, parameters, clf)
   return;
if __name__ == "__main__":
    main()

