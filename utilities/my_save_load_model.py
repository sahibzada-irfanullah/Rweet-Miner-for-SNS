import pickle
class Save_Load_Model:
  '''For saving and loading model'''
  def __init__(self, folderPath = None):
    '''Take path to the folder for saving and loading model as an argement. Default is 'saved _models'''
    if folderPath is None:
      print("Please, provide path to folder.")
      exit(0)
    else:
      self.folderPath = folderPath

  def save_Model(self, model, filename):
    '''-To save a model
      + take model and complete filename as argements
    '''
    self.filename  = self.folderPath + filename
    self.model = model
    with open(self.filename, 'wb') as fid:
      pickle.dump(self.model, fid)
      print("Model saved successfully")

  def load_Model(self, filename):
    '''To load a model
      + take complete filename as an argement
      -return model
    '''
    self.filename  = self.folderPath + filename
    with open(self.filename, 'rb') as fid:
      self.model = pickle.load(fid)
      print("model loaded succesfully")
    return self.model