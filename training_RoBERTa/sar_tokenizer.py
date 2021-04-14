class MyTokenizer:
  vocab_size = 0
  vocab = []
  id_to_token = {}
  token_to_id = {}

  def __init__(self):
    vocabfile = open('training_RoBERTa/roberta_decoder.vocab','rb')
    self.vocab = pickle.load(vocabfile)
    vocabfile.close()
    self.vocab_size = len(self.token_to_id) 
    # Special tokens: <s><pad></s><unk>
    self.add_token('<s>')
    self.add_token('<pad>')
    self.add_token('</s>')
    self.add_token('<unk>')
    for v in self.vocab:
      self.add_token(v)
    self.add_token('None')

  def tokenize(self, s):
    return s.split()

  def add_token(self, s):
    if s not in self.token_to_id:
      self.id_to_token[self.vocab_size] = s
      self.token_to_id[s] = self.vocab_size
      self.vocab_size+=1

  def convert_string_to_ids(self, s):
    tokens = s.split()
    ids = []
    for token in tokens:
      ids.append(self.token_to_id[token])
    return ids

  def decode(self, ids):
    text = ""
    for id in ids:
      text += self.id_to_token[id] + " "
    return text[:-1]