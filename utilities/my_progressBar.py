
# coding: utf-8

# In[19]:


import time
import sys
import datetime

class My_progressBar:
  '''used to show progress bar in console. '''
  # # Showing progress <--------------
  def __init__(self, message="started:", task_Size=None):
    '''initialize task_Size, steps to take and initialize count to zero'''
    if task_Size is None:
      print("size of the task should be passed")
      self.task_Size = 20
    else:
      self.task_Size = task_Size
    print(message)
    self.count = 0
    self.steps = self.task_Size // 2
  def call_to_progress(self,start_time):
    '''this is called with providing start_time of the task, which then calculate the elapased
    time and called the progress() with passing elapsed time as argement'''
    self.count = self.count + 1
    self.elapsed_time = time.time() - start_time
    if self.count % 2 == 0:
      self.progress(self.elapsed_time)
    if self.count == self.task_Size:
      self.progress(self.elapsed_time)
  def progress(self, cal_time):
    '''this update the loading bar of the in run time'''
    bar_len = 50
    time = cal_time * (self.task_Size - self.count)
    status = str(datetime.timedelta(seconds=int(time)))
    filled_len = int(round(bar_len * self.count / float(self.task_Size)))
    percents = round(100.0 * self.count / float(self.task_Size), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] [%s%s] [eta:%s]\r' % (bar, percents, '%', status))
    sys.stdout.flush()

  # Showing progress -------------->

  # ****************************************************************

# def folding_code(k):
#   num_Clf = 3
  
#   for j in range(0, k):
#     test = My_progressBar("\n\n Running "+str(j+1)+" out of "+str(k)+' fold(s):',num_Clf)
#     start_time = time.time()
#     time.sleep(0.1)
#     test.call_to_progress(start_time)
#     start_time = time.time()
#     time.sleep(1)
#     test.call_to_progress(start_time)
#     start_time = time.time()
#     time.sleep(3)
#     test.call_to_progress(start_time)
    
# k=5

# folding_code(k)
# folding_code(k)
      

# for i in range(0, 20):
#   start_time = time.time()
#   time.sleep(0.3)
#   test.call_to_progress(start_time)


# In[11]:


# global steps
# global dataset_Size
# global count

# # # Showing progress <--------------
# def call_to_progress(start_time):
#   global count
#   global elapsed_time
#   count = count + 1
#   elapsed_time = time.time() - start_time
#   if count % 500 == 0:
#     progress(elapsed_time)
#   if count == dataset_Size:
#     progress(elapsed_time)
  
# def init_prgoress_para():
#   global count
#   global steps
#   global dataset_Size
#   count = 0
#   steps = dataset_Size//500
# def progress(cal_time):
#   bar_len = 50
#   time = cal_time * (dataset_Size-count)
#   status = str(datetime.timedelta(seconds=int(time)))
#   filled_len = int(round(bar_len * count / float(dataset_Size)))
#   percents = round(100.0 * count / float(dataset_Size), 1)
#   bar = '=' * filled_len + '-' * (bar_len - filled_len)
#   sys.stdout.write('[%s] [%s%s] [eta:%s]\r' % (bar, percents, '%', status))
#   sys.stdout.flush()

# # Showing progress -------------->


# # In[14]:


# get_ipython().system('pip3.6 install xlsxwriter')


# # In[ ]:




