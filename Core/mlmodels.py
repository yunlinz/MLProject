import nltk
import business
from business import Business
import json
import xgboost
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


def one_vs_rest(x_train,y_train):
    one_v_rest = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x_train, y_train)
    return one_v_rest


