import time

import datetime
from sklearn import preprocessing
import pandas as pd
from feauture_extraction.Feature_Extraction import Ngrams_Rules_Features
from training_clf.Training_classifier import Training_Classifiers
import os, errno
def create_Folder(file_path):
  '''Def: To create folder
  Args: pass folder path with full name to create it
  Ret: null'''
  try:
      os.makedirs(file_path)
      print('Folder, \" ' + file_path + "\" is created successfully.")
  except OSError as e:
      print('Folder, \" ' + file_path + "\" might already exists.")
      if e.errno != errno.EEXIST:
          raise


script_start = time.time()  # script start point

# Name of dataset
fnameTrainingDataset = "../datasets/preprocessed_binary_label_dataset.csv"
# fnameTrainingDataset = "../datasets/preprocessed_multi_label_dataset.csv"


df = pd.read_csv(fnameTrainingDataset)
X = df['tweet-text']
y = df['tweet-class']
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
feat_extract1 = Ngrams_Rules_Features()
file_path = '../simple_vectorizer/'
create_Folder(file_path)
features1 = feat_extract1.generate_Features(X,'simple', 1, 1.0, 'no', pathToVectFolder=file_path)
tc1 = Training_Classifiers()
file_path_res = '../simple_vect_results/'
create_Folder(file_path_res)
file_path_clf = '../simple_vect_clfModels/'
create_Folder(file_path_clf)
tc1.training_clfs(features1, y, k_fold = 5, pathToResFolder= file_path_res, pathToClfFolder=file_path_clf)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
file_path = '../tfidf_vectorizer/'
create_Folder(file_path)
feat_extract2 = Ngrams_Rules_Features()
features2 = feat_extract2.generate_Features(X,'tfidf', 1, 1.0, 'no', pathToVectFolder=file_path)
tc2 = Training_Classifiers()
file_path_res = '../tfidf_results/'
create_Folder(file_path_res)
file_path_clf = '../tifidf_clfModels/'
create_Folder(file_path_clf)
tc2.training_clfs(features2, y, k_fold = 5, pathToResFolder= file_path_res, pathToClfFolder=file_path_clf)


print("\nMain script completed")
Total_time = time.time() - script_start
print("\nTotal time for script completion :" + str(datetime.timedelta(seconds=int(Total_time))))