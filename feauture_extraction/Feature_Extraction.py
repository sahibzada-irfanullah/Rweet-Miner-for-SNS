
# coding: utf-8

# In[44]:

import warnings
warnings.filterwarnings('ignore')#replace ignore with default for enabling the warning)
import pandas as pd
from utilities.my_progressBar import My_progressBar
import scipy
import time
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from utilities.my_save_load_model import Save_Load_Model

class Ngrams_Rules_Features:
  '''Generating Ngrams and rules features'''


  def apply_regex(self, text, regex):
    match_found = (re.search(regex ,text) !=None)
    match_found = int(match_found == True)
    return match_found

  def gen_rules_features(self, X_data_series): # , X_data_dtm, features_arg):
    '''sparse matrix and series matrices should be converted to dataframe for applying rules and treating
    it as features...
    I wrote two functions i.e.,sparse_matrix_to_DataFrame() and series_DataFrame()
      for changing datatypes'''
    X_data_DF = self.series_to_DataFrame(X_data_series)
    regexes = [
      re.compile(r'\b(I|we)\b.*\b(am|are|will be)\b.*\b(bringing|giving|helping|raising|donating|auctioning)\b', re.I|re.M),
      re.compile(r'\b(I\'m)\b.*\b(bringing|giving|helping|raising|donating|auctioning)\b', re.I|re.M),
      re.compile(r'\b(we\'re)\b.*\b(bringing|giving|helping|raising|donating|auctioning)\b', re.I|re.M),
      re.compile(r'\b(I|we)\b.*\b(will|would like to)\b.*\b(bring|give|help|raise|donate|auction)\b', re.I|re.M),
      re.compile(r'\b(I|we)\b.*\b(will|would like to)\b.*\b(work|volunteer|assist)\b', re.I|re.M),
      re.compile(r'\b(we\'ll)\b.*\b(bring|give|help|raise|donate|auction)\b', re.I|re.M),
      re.compile(r'\b(I|we)\b.*\b(ready|prepared)\b.*\b(bring|give|help|raise|donate|auction)\b', re.I|re.M),
      re.compile(r'\b(where)\b.*\b(can I|can we)\b.*\b(bring|give|help|raise|donate)\b', re.I|re.M),
      re.compile(r'\b(where)\b.*\b(can I|can we)\b.*\b(work|volunteer|assist)\b', re.I|re.M),
      re.compile(r'\b(I|we)\b.*\b(like|want)\b.*\bto\b.*\b(bring|give|help|raise|donate)\b', re.I|re.M),
      re.compile(r'\b(I|we)\b.*\b(like|want)\b.*\bto\b.*\b(work|volunteer|assist)\b', re.I|re.M),
      re.compile(r'\b(will be)\b.*\b(brought|given|raised|donated|auctioned)\b', re.I|re.M),
      re.compile(r'\b\w*\s*\b\?', re.I|re.M),
      re.compile(r'\b(you|u).*(can|could|should|want to)\b', re.I|re.M),
      re.compile(r'\b(can|could|should).*(you|u)\b', re.I|re.M),
      re.compile(r'\b(like|want)\b.*\bto\b.*\b(bring|give|help|raise|donate)\b', re.I|re.M),
      re.compile(r'\b(how)\b.*\b(can I|can we)\b.*\b(bring|give|help|raise|donate)\b', re.I|re.M),
      re.compile(r'\b(how)\b.*\b(can I|can we)\b.*\b(work|volunteer|assist)\b', re.I|re.M)

    ]
    temp = pd.DataFrame()
    features_arg = []
    for i, regex in zip(range(len(regexes)), regexes):
      columnName = "RegEx_" + str(i + 1)
      features_arg.append(columnName)
      temp[columnName] = X_data_DF['tweet-text'].apply(lambda text: self.apply_regex(text, regex))
    temp_sparse = scipy.sparse.csr_matrix(temp.values)
    return temp_sparse, features_arg

  def concat_sparse_matrices_h(self, data_X_dtm, data_Rules_dtm, features_X, features_Rules):
    combined_features = features_X + features_Rules
    concat_sparse = scipy.sparse.hstack([data_X_dtm, data_Rules_dtm], format='csr')
    return concat_sparse, combined_features

  def gen_Ngrams(self, X_train, X_test, lower, higher):
    Vect = CountVectorizer(ngram_range=(lower, higher))
    X_train_dtm = Vect.fit_transform(X_train)
    X_test_dtm = Vect.transform(X_test)
    return X_train_dtm, X_test_dtm, Vect.get_feature_names()

  def sparse_matrix_to_DataFrame(self, X_data_dtm, features):
    X_data_dtm = pd.DataFrame(X_data_dtm.toarray(), columns=features)
    return X_data_dtm

  def series_to_DataFrame(self, X_data):
    X_data = X_data.to_frame()
    return X_data


  def apply_gen_rules_features(self, X, X_train_WoR_dtm, X_train_WoR_Features):
    X_Rules_dtm, features_Rules = self.gen_rules_features(X)
    X_train_WR_dtm, combined_features = self.concat_sparse_matrices_h(X_train_WoR_dtm, X_Rules_dtm, X_train_WoR_Features, features_Rules)
    return  X_train_WR_dtm, combined_features


  def simple_CountVect(self, X, lower, higher, minDF, maxDF, include_rules):
    Vect = CountVectorizer(ngram_range=(lower, higher), min_df = minDF, max_df = maxDF)
    # X_train_WoR_dtm = Vect.fit_transform(X)
    Vect.fit(X)
    X_train_WoR_dtm = Vect.transform(X)
    X_train_Features = Vect.get_feature_names()
    if include_rules == 'yes':
      X_train_dtm, X_train_Features  = self.apply_gen_rules_features(X, X_train_WoR_dtm, X_train_Features)
    else:
      X_train_dtm = X_train_WoR_dtm
    return X_train_dtm, X_train_Features, Vect, X_train_dtm.shape

  def setPathToVectFolder(self, pathToVectFolder = '../vect_phase2/'):
    '''Def: set path to the folder containing vectorizers
    Args: pass path to the folder to folder containing vectorizers as an argument (Default: '../vectorizer/')
     Ret: none'''
    self.pathToVectFolder = pathToVectFolder

  def tfIdf_Vect(self, X, lower, higher, minDF, maxDF, include_rules):
    Vect = TfidfVectorizer(ngram_range=(lower, higher), min_df=minDF, max_df=maxDF)
    Vect.fit(X)
    X_train_WoR_dtm = Vect.transform(X)
    X_train_Features = Vect.get_feature_names()
    if include_rules == 'yes':
      X_train_dtm, X_train_Features  = self.apply_gen_rules_features(X, X_train_WoR_dtm, X_train_Features)
    else:
      X_train_dtm = X_train_WoR_dtm
    return X_train_dtm, X_train_Features, Vect, X_train_dtm.shape

  def attach_date(self, fname):
    created_at = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    fname = created_at + fname
    return fname


  def generate_Features(self, X, vect_type = 'simple', minDF = 1, maxDF = 1.0, include_rules = 'yes', pathToVectFolder = None):
    '''Generating Features from data
    -Takes attributes,responses, vect_type(simple by default), minDF, maxDF'''
    #generating name for the dataset file
    if pathToVectFolder == None:
      print("Please Provide the path to the folder for saving vectorizers")
      exit(0)
    self.pathToVectFolder= pathToVectFolder
    save_vect = Save_Load_Model(self.pathToVectFolder)
    if vect_type == 'simple':
      temp_label = 'CountVect'
      func_handler = self.simple_CountVect
    else:
      temp_label = 'TfIdfVect'
      func_handler = self.tfIdf_Vect
    temp_label = temp_label +'_minDF-' + str(minDF) + '_maxDF-' + str(maxDF)

    if include_rules == 'yes':
      temp_label = temp_label + '_WRul'
    else:
      temp_label = temp_label + '_WoRul'
    pBar = My_progressBar('Generating Features:',6)
    for i in range(1, 4):
      for j in range(i, 4):
        start_time = time.time()
        if i == j == 1:
          X_train_dtm_uni, X_train_Features_uni, vect, dim =               \
            func_handler(X, i, j, minDF, maxDF, include_rules)  #temp_label represents vectorizer type i.e., simple or tfidf
          uniLabel = '_Unigrams_' + temp_label + '_Freq-' + str(dim[1])
          uniLabel = self.attach_date(uniLabel)
          save_vect.save_Model(vect, uniLabel)
          # print('feat')
          # print(uniLabel)
          # exit(0)
        elif (i == 1) & (j == 2):
          X_train_dtm_uniBi, X_train_Features_uniBi, vect, dim =               \
            func_handler(X, i, j, minDF, maxDF, include_rules)
          uniBiLabel = "_UniAndBigrams_" + temp_label + '_Freq-' + str(dim[1])
          uniBiLabel =  self.attach_date(uniBiLabel)
          save_vect.save_Model(vect, uniBiLabel)
        elif (i == 1) & (j == 3):
          X_train_dtm_uniBiTri, X_train_Features_uniBiTri, vect, dim =               \
            func_handler(X, i, j, minDF, maxDF, include_rules)
          uniBiTriLabel = "_UniBiAndTrigrams_" + temp_label + '_Freq-' + str(dim[1])
          uniBiTriLabel = self.attach_date(uniBiTriLabel)
          save_vect.save_Model(vect, uniBiTriLabel)
        elif (i == 2) & (j == 2):
          X_train_dtm_bi, X_train_Features_bi, vect, dim =               \
            func_handler(X, i, j, minDF, maxDF, include_rules)
          biLabel =  "_Bigrams_" + temp_label + '_Freq-' + str(dim[1])
          biLabel = self.attach_date(biLabel)
          save_vect.save_Model(vect, biLabel)
        elif (i == 2) & (j == 3):
          X_train_dtm_biTri, X_train_Features_biTri, vect, dim =               \
            func_handler(X, i, j, minDF, maxDF, include_rules)
          biTriLabel =  "_BiTrigrams_" + temp_label + '_Freq-' + str(dim[1])
          biTriLabel = self.attach_date(biTriLabel)
          save_vect.save_Model(vect, biTriLabel)
        elif (i == 3) & (j == 3):
          X_train_dtm_tri, X_train_Features_tri, vect, dim =               \
            func_handler(X, i, j, minDF, maxDF, include_rules)
          triLabel = "_Trigrams_" + temp_label + '_Freq-' + str(dim[1])
          triLabel =  self.attach_date(triLabel)
          save_vect.save_Model(vect, triLabel)
        pBar.call_to_progress(start_time)
    encap = {'unigrams':{'dtm':X_train_dtm_uni, 'header':X_train_Features_uni, 'specs': uniLabel},          'uniBigrams':{'dtm':X_train_dtm_uniBi, 'header':X_train_Features_uniBi, 'specs': uniBiLabel},          'uniBiTrigrams':{'dtm':X_train_dtm_uniBiTri, 'header':X_train_Features_uniBiTri, 'specs': uniBiTriLabel},          'bigrams':{'dtm':X_train_dtm_bi, 'header':X_train_Features_bi, 'specs': biLabel},          'biTrigrams':{'dtm':X_train_dtm_biTri, 'header':X_train_Features_biTri, 'specs': biTriLabel},          'trigrams':{'dtm':X_train_dtm_tri, 'header':X_train_Features_tri, 'specs': triLabel}}
    return encap









script_start = time.time() #script start point
# fnameTrainingDataset = "/home/sahibzada/Desktop/ipythonNB/ThImp/Datasets/specific labelled/preprocessed_dataset_specific.csv"
# fnameTrainingDataseet = "/home/sahibzada/Desktop/ipythonNB/ThImp/Datasets/TestingDatasets/preprocessed_dataset.csv"



# df = pd.read_csv(fnameTrainingDataseet)
# X = df['tweet-text']
# y = df['tweet-class']
# le = preprocessing.LabelEncoder()
# y = le.fit_transform(y)
# feat_extract = Ngrams_Rules_Features()
# features = feat_extract.generate_Features(X,'simple', 1, 1.0, 'no')


print("\nscritp completed")
Total_time = time.time() - script_start
print("\nTotal time for script completion :" + str(datetime.timedelta(seconds=int(Total_time))))

