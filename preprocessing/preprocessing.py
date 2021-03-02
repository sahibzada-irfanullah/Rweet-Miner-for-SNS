import warnings
warnings.filterwarnings('ignore')#replace ignore with default for enabling the warning)
import pandas as pd
import time
import re
import datetime
import nltk
from pywsd.utils import lemmatize_sentence
import sys
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
script_start = time.time()
global steps
global dataset_Size
global count

# # Showing progress <--------------
def call_to_progress(start_time):
  global count
  global elapsed_time
  count = count + 1
  elapsed_time = time.time() - start_time
  if count % 500 == 0:
    progress(elapsed_time)
  if count == dataset_Size:
    progress(elapsed_time)
  
def init_prgoress_para():
  global count
  global steps
  global dataset_Size
  count = 0
  steps = dataset_Size//500
def progress(cal_time):
  bar_len = 50
  time = cal_time * (dataset_Size-count)
  status = str(datetime.timedelta(seconds=int(time)))
  filled_len = int(round(bar_len * count / float(dataset_Size)))
  percents = round(100.0 * count / float(dataset_Size), 1)
  bar = '=' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] [%s%s] [eta:%s]\r' % (bar, percents, '%', status))
  sys.stdout.flush()

# Showing progress -------------->

# ****************************************************************
# Removing Punctuations <--------------
def remove_punc(text):
  start_time = time.time()
  text = text.lower()
  tweet_tokenizer = TweetTokenizer()
  punctuation = list(string.punctuation)
  punctuation = punctuation + list(["."*2,"."*3, "."*4, "?"*2,"?"*3,"?"*4, "!"*2, "!"*3, "!"*4])
  tokens = tweet_tokenizer.tokenize(text)
  clean_tokens = []
  for tok in tokens:
    if tok not in punctuation:
      clean_tokens.append(tok)
  text = token_str(clean_tokens)
  call_to_progress(start_time)
  return text
# Removing Punctuations -------------->

#****************************************************************
# stemming <------------
def stemming(text):
  start_time = time.time()
  tweet_tokenizer = TweetTokenizer()
  stemmer = PorterStemmer() 
  result = [stemmer.stem(i.lower()) for i in tweet_tokenizer.tokenize(text)]
  call_to_progress(start_time)
  return token_str(result)
# stemming ------------>
#****************************************************************


# converting tokens to string <------------
def token_str(tokens = []):
  return ' '.join(tokens)
# converting tokens to string ------------>
#****************************************************************

# converting list string <------------
def list_str(list1):
  return ' '.join(list1)
# converting list string  ------------>

#****************************************************************

# removing Non-ascii characters <-----------
def removeNonAscii(s):
  start_time = time.time()
  chunks = []
  for i in s: 
    if ord(i)<128:
      chunks.append(i)
  text=''.join(chunks)
  call_to_progress(start_time)
  return text
# removing Non-ascii characters ----------->

#****************************************************************

# removing stopwords <------------
def removeStopWords(text):
  #porter = PorterStemmer()
  start_time = time.time()
  stopword_list = set(stopwords.words('english'))
  tweet_tokenizer = TweetTokenizer()
  # result = [spell(porter.stem(i.lower())) for i in tweet_tokenizer.tokenize(doc) if i.lower() not in stopword_list]
  result = [i.lower() for i in tweet_tokenizer.tokenize(text) if i.lower() not in stopword_list]
  call_to_progress(start_time)
  return token_str(result)
# removing stopwords ------------->

#****************************************************************
# Num, URL, Menition, RT replacement <------------
def replace_Num_Url_Mention_RT(text):
  start_time = time.time()
  result = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)','_NUM_', text)
  result = re.sub(r'(?:(RT|rt) @ ?[\w_]+:?)','_RT_', result)
  result = re.sub(r'(?:@ ?[\w_]+)','_MENTIOM_', result)
  result = re.sub(r'http[s]? ?: ?//(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
                  '_URL_', result)
  call_to_progress(start_time)
  return result
# Num, URL, Menition, RT replacement ------------->
#****************************************************************

#****************************************************************
# Remove Non-English tweets <------------
ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS

STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}


def get_language(text):
  words = set(nltk.wordpunct_tokenize(text.lower()))
  return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key=lambda x: x[1])[0]


def is_english(text):
  text = text.lower()
  words = set(nltk.wordpunct_tokenize(text))
  if len(words & ENGLISH_STOPWORDS) > len(words & NON_ENGLISH_STOPWORDS):
    return 'eng'
  else:
    return 'non-eng'
#****************************************************************
# Remove Non-English tweets <------------

#****************************************************************
# lower case conversion <------------
def lower_case_conversion(text):
  return text.lower()
#****************************************************************
# lower case conversion <------------

#****************************************************************
# removing null or uni lenght tweets <----------
def remove_null_unilenght(text):
  return len(text)
#****************************************************************
# removing null or uni lenght tweets <----------

tweet_tokenizer = TweetTokenizer()


# dataset

# fname = "multi_label_dataset.csv"

fname = "binary_label_dataset.csv"

path = "../datasets/"
dataset = pd.read_csv(path + fname, index_col= False)
# print("Dataset Loaded Successfully")
dataset_Size = len(dataset)
# print('size :', dataset_Size)

print("\nRemoving non-ascii Characters...")
init_prgoress_para()
dataset['tweet-text'] = dataset['tweet-text'].apply(removeNonAscii)
# dataset.to_csv("dataset/nonascii.csv")
print("\nNon-Ascii Characters Removed!")

print("\nRemoving Non-English tweets...")
dataset['lang']= dataset['tweet-text'].apply(is_english) #if dataset already contains lang info then no need to run the this statement
dataset = dataset[dataset['lang'] == 'eng']
dataset = dataset.drop('lang', axis = 1)
# dataset.to_csv("dataset/noneng.csv")
print("\nNon-English tweets Removed!")

print("\n Lower Coversion started...")
dataset['tweet-text'] = dataset['tweet-text'].apply(lower_case_conversion)
# dataset.to_csv("dataset/lwconversion.csv")
print("\n Lower Coversion finished...")

print("\nRemoving punctuation tweets...")
dataset['tweet-text'] = dataset['tweet-text'].apply(remove_punc)
# dataset.to_csv("dataset/remov_punc.csv")
print("Punctuations Removed")


print("\nRemoving Stopwords...")
init_prgoress_para()
dataset['tweet-text']= dataset['tweet-text'].apply(removeStopWords)
# dataset.to_csv("dataset/remov_stopwords.csv")
print("\nStop Words Removed!")


print("\nRemoving tweets having null or uni length tweets...")
dataset['length']= dataset['tweet-text'].apply(remove_null_unilenght)
dataset['length'] = dataset[dataset['length'] < 2]
dataset = dataset.drop('length', axis = 1)
# dataset.to_csv("dataset/remove_null_single.csv")
print("\nTweets having null or uni length tweets removed...")


print("\nURLs, Mentions, Retweets, Numbers replacement started...")
init_prgoress_para()
dataset['tweet-text'] = dataset['tweet-text'].apply(replace_Num_Url_Mention_RT)
dataset = dataset.drop_duplicates(subset='tweet-text', keep='first')
# dataset.to_csv("dataset/general_tags.csv")
print("\nURLs, Mentions, Retweets, Numbers replacement completed!")


print("\nLemmatization started...")
init_prgoress_para()
dataset['tweet-text']= dataset['tweet-text'].apply(lemmatize_sentence)
dataset['tweet-text']= dataset['tweet-text'].apply(list_str)
# dataset.to_csv("dataset/lemmatized_dataset.csv")
print("\nLemmatization completed!")


print("\nRemoving duplicate tweets...")
dataset = dataset.drop_duplicates(subset='tweet-text', keep='first')
dataset.to_csv(path + "preprocessed_" + fname)
print('Duplicate tweets are removed:')
# print('size :', len(dataset))

Total_time = time.time() - script_start
print("Total time for script completion" + str(datetime.timedelta(seconds=int(Total_time))))
