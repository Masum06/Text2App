import nltk
nltk.download('all')
from nltk.tokenize import sent_tokenize, word_tokenize
from PyDictionary import PyDictionary
import re

class Text2App:
  NL = ""
  SAR = ""
  literal_dict = {}
  best_model_dir = 'model_checkpoints/model_step_30000.pt'

  def format_text(self, NL):
    NL = ' '.join(word_tokenize(NL))
    text_num_dict = {}
    NL = NL.replace("'", '"')
    strings_list = []
    strings_list = re.findall(r'\"(.+?)\"', NL)

    if len(strings_list) != 0:
      for i, strng in enumerate(strings_list):
        key = "string" + str(i)
        text_num_dict[key] = strng
        to_replace = '"' + strng + '"'
        NL = NL.replace(to_replace, key)  
    
    numbers_list = []
    tokens = NL.split()
    for token in tokens:
      if is_Number(token):
        numbers_list.append(token)
    
    if len(numbers_list) != 0:
      for i, number in enumerate(numbers_list):
        key = "number" + str(i)
        text_num_dict[key] = str(number)
        to_replace = str(number)
        NL = NL.replace(to_replace, key)
  
    for comp in ["button", "switch", "label"]:
      for i in range(1, 4):
        key = comp + str(i)
        text_num_dict[key] = key
    
    text_num_dict["random_player_source"] = "DanceMonkey.mp3"
    text_num_dict["random_video_player_source"] = "text2app_live.mp4"

    return NL, text_num_dict

  def tranlsate(self, NL):
    nl_file = open("single_test.txt", "w")
    nl_file.write(NL)
    nl_file.close()
    s = "python OpenNMT-py-legacy/translate.py -model {} -src single_test.txt -output sar_to_compile.txt -replace_unk -verbose -beam_size 1".format(self.best_model_dir)
    subprocess.call(s.split())
    f = open("sar_to_compile.txt", 'r')
    sar = f.read().strip()
    f.close()
    subprocess.call('rm single_test.txt'.split())
    subprocess.call('rm sar_to_compile.txt'.split())
    return sar

  def __init__(self, NL):
    self.NL, self.literal_dict = self.format_text(NL)
    self.SAR = self.translate(NL)
