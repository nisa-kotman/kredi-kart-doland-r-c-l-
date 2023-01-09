# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 02:46:25 2023

@author: Nisa
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('creditcard.csv',sep=',')

# Veri setini Minimize ettik aynı zamanda veri seti dağılımını düzenledik
newData = data.drop(data[data.Class.eq(0)].sample(frac=0.998).index);
# Class sütunundaki değerleri Y atadık
Y=newData["Class"]
# veri setinin geri kalan kısmı X atadık
X = newData.drop('Class',axis=1)
print(X.shape)

# veri setimizi %20 test, %80 eğitim olarak böldük
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


######################## DEĞERLENDİRME ÖLÇÜTLERİ ########################

def mymodel(model):
    #model.fit(X_train.loc[:,solution],Y_train)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    
    tr=model.score(X_train,Y_train)
    te=model.score(X_test,Y_test)
    
    # print('Accuracy: ',accuracy_score(Y_test,ypred),\
    #       '\nConfusion Matrix: \n',confusion_matrix(Y_test,ypred))
    
    print(str(model)[:-2],'Accuracy: ',accuracy_score(Y_test,Y_pred),\
          '\nConfusion Matrix: \n',confusion_matrix(Y_test,Y_pred))
    print("Classification report: \n",classification_report(Y_test,Y_pred))
    print(f'Training Accuracy: {tr}\nTesting Accuracy: {te}')
    print()
    
    cnf = ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred, cmap = 'Blues')
    #confusion = confusion_matrix(Y_test, Y_pred)
    cnf.plot(cmap = 'Blues')
    
    plt.colorbar(cnf, cmap='binary')

    # Plot the confusion matrix
    #plt.pyplot(cnf, cmap='binary')
    plt.show()
#################################  SVM  ###################################    

model = SVC()
model.fit(X_train,Y_train)
print("normal svm")
mymodel(model)

# SVM ALGORİTMASI

def objective_function(solution):
  model = SVC()
  if(sum(solution)==0):
    return 0
  model.fit(X_train.loc[:,solution],Y_train)
  mymodel(model);


#################################  KNN  ###################################  

model = KNeighborsClassifier()
model.fit(X_train,Y_train)
mymodel(model); 

# KNN ALGORİTMASI
def objective_function(solution):
  model = KNeighborsClassifier(3)
  if(sum(solution)==0):
    return 0
  model.fit(X_train.loc[:,solution],Y_train)
  mymodel(model); 

###########################  KARAR AĞACI  ################################

model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
mymodel(model);

def objective_function(solution):
  model = DecisionTreeClassifier()
  if(sum(solution)==0):
      return 0
  model.fit(X_train.loc[:,solution],Y_train)
  mymodel(model);


#######################  NEİGHBORHOOD FUNCTİON  ###########################


def neighborhood_function(solution, obj_val):
    neighbors = []
    for i in range(len(solution)):
        temp_sol = solution.copy()
        temp_sol[i] = ~temp_sol[i]
        if ( objective_function(temp_sol) < obj_val ):
            neighbors.append(temp_sol)
    if len(neighbors) == 0:
        return None
    rand_ind = np.random.randint(0, len(neighbors))
    return neighbors[rand_ind]
    


initial_state=1
#cost_function=1
max_iterations=100
solution = np.random.rand(30)>0.5
obj_val = objective_function(solution)

# best_solution = solution.copy()
# best_val = obj_val

convergence = []

#############################  HİLL CLİMBİNG  ##############################

def hill_climbing( cost_function):
        
    solution = initial_state
    current_cost = cost_function(solution)
    for i in range(max_iterations):
      next_state = neighborhood_function(solution, current_cost)
      next_cost = cost_function(next_state)
      if next_cost > current_cost:
        solution = next_state
        current_cost = next_cost
      else:
        # We have reached a local maximum, so stop
        break
    return solution


############################  SİMULATED ANNEALİNG  ############################
temperature_schedule=0.1
def simulated_annealing(temperature_schedule):
  # Initialize the current state with a random subset of features
  num_features = X_train.shape[1]
  solution = [np.random.randint(0, 1) for _ in range(num_features)]
  current_cost = objective_function(solution, X_train, Y_train)

  for t in range(max_iterations):
    temperature = temperature_schedule(t)
    next_state = neighborhood_function(solution,obj_val)
    next_cost = objective_function(next_state, X_train, Y_train)
    if next_cost > current_cost or math.exp((next_cost - current_cost) / temperature) > np.random.random():
      solution = next_state
      current_cost = next_cost
  return solution

# Use the simulated annealing algorithm to find the optimal subset of features
