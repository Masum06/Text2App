import nltk, re, subprocess
nltk.download('all')
from nltk.tokenize import sent_tokenize, word_tokenize
from PyDictionary import PyDictionary
import re
import random
import os

class Text2App:
  NL = ""
  SAR = ""
  literal_dict = {}
  best_model_dir = 'model_checkpoints/PointerNet/model_step_14500.pt'

  def is_Number(self, test):
    is_number = True
    try:
      float(test)
    except ValueError:
      is_number = False
    return is_number

  def format_text(self, NL):
    NL = ' '.join(word_tokenize(NL)).replace('`` ', '"').replace(" ''", '"').replace(" ' ", "' ")

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
      if self.is_Number(token):
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
    
    text_num_dict["random_player_source"] = random.choice(os.listdir("./Media/Music"))
    text_num_dict["random_video_player_source"] = random.choice(os.listdir("./Media/Videos"))

    

    return NL, text_num_dict

  def translate(self, NL):
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
    self.NL, self.literal_dict = self.format_text(NL.lower())
    self.SAR = self.translate(self.NL)




###### SAR PARSER ##########



import re
import os
import shutil
import subprocess

"""# Static Dictionaries"""

vis_comp_dict = {
    "<textbox>" : """{"$Name":"TextBox<|number|>","$Type":"TextBox","$Version":"6","Hint":"Hint for TextBox<|number|>","Uuid":"<|negativeuuid|>"}""",
    "<button>" : """{"$Name":"Button<|number|>","$Type":"Button","$Version":"6","Text":"<|buttontext|>","Uuid":"<|negativeuuid|>"}""",
    "<text2speech>": """{"$Name":"TextToSpeech<|number|>","$Type":"TextToSpeech","$Version":"5","Uuid":"<|negativeuuid|>"}""",
    "<canvas>" : """{"$Name":"Canvas<|canvas_number|>","$Type":"Canvas","$Version":"12","Height":"-2","Width":"-2","Uuid":"<|positive_uuid|>","$Components":<|canvas_components|>}""",
    "<accelerometer>" : """{"$Name":"AccelerometerSensor<|number|>","$Type":"AccelerometerSensor","$Version":"4","Uuid":"<|negative_uuid|>"}""",
    "<video_player>" : """{"$Name":"VideoPlayer<|number|>","$Type":"VideoPlayer","$Version":"6","Source":"<|source|>","Uuid":"<|negative_uuid|>"}""",
    "<switch>" : """{"$Name":"Switch<|number|>","$Type":"Switch","$Version":"1","Text":"<|switch_text|>","Uuid":"<|positive_uuid|>"}""",
    "<player>" :"""{"$Name":"Player<|number|>","$Type":"Player","$Version":"6","Source":"<|source|>","Uuid":"<|negative_uuid|>"}""",
    "<label>" : """{"$Name":"Label<|number|>","$Type":"Label","$Version":"5","Text":"<|label_text|>","Uuid":"<|positive_uuid|>"}""",
    "<datepicker>" : """"{"$Name":"DatePicker<|number|>","$Type":"DatePicker","$Version":"3","Text":"Choose a Date","Uuid":"<|negative_uuid|>"}""",
    "<timepicker>" : """{"$Name":"TimePicker<|number|>","$Type":"TimePicker","$Version":"3","Text":"Choose a Time","Uuid":"<|negative_uuid|>"}""",
    "<passwordtextbox>" : """{"$Name":"PasswordTextBox<|number|>","$Type":"PasswordTextBox","$Version":"4","Uuid":"<|negative_uuid|>"}"""
}

canvas_comp_dict = {
    "<ball>" : """{"$Name":"Ball<|ball_number|>","$Type":"Ball","$Version":"6","Radius":"10","Uuid":"<|positive_uuid|>","X":"132","Y":"147"}"""
}

#textbox text edited on December 8. "<value name=VALUE>" block removed.
color_dic = {
    "<red>" : "#ff0000",
    "<green>" : "#00ff00",
    "<black>" : "#000000",
    "<cyan>" : "#00ffff",
    "<pink>" : "#ffafaf",
    "<magenta>" : "#ff00ff",
    "<blue>" : "#0000ff",
    "<light_gray>" : "#cccccc",
    "<orange>" : "#ffc800",
    "<yellow>" : "#ffff00", 
    "<dark_gray>" : "#444444",
    "<gray>" : "#888888"
  
}

logic_dic = {
	"<code>" : """<xml xmlns="http://www.w3.org/1999/xhtml">""",

	"</code>" :   """<yacodeblocks ya-version="208" language-version="33"></yacodeblocks>
	</xml>""",

	"<button_click>" : """<block type="component_event" id="<|string_id|>" x="-717" y="-541">
	    <mutation component_type="Button" is_generic="false" instance_name="Button<|number|>" event_name="Click"></mutation>
	    <field name="COMPONENT_SELECTOR">Button<|number|></field>
	    <statement name="DO">""",

	"</button_click>" : """</statement>
  </block>""",

  "<textbox_text>" : """<block type="component_set_get" id="<|string_id|>">
            <mutation component_type="TextBox" set_or_get="get" property_name="Text" is_generic="false" instance_name="TextBox<|textbox_number|>"></mutation>
            <field name="COMPONENT_SELECTOR">TextBox<|textbox_number|></field>
            <field name="PROP">Text</field>
            </block> """,

  "<text2speech>" : """ <block type="component_method" id="<|string_id|>">
        <mutation component_type="TextToSpeech" method_name="Speak" is_generic="false" instance_name="TextToSpeech<|text2speech_number|>"></mutation>
        <field name="COMPONENT_SELECTOR">TextToSpeech<|text2speech_number|></field>
        <value name="ARG0"> """,
          
  "</text2speech>" : """  </value>
      </block>  """,

  "<ball_flung>" : """<block type="component_event" id="<|string_id|>" x="-381" y="-151">
    <mutation component_type="Ball" is_generic="false" instance_name="Ball<|ball_number|>" event_name="Flung"></mutation>
    <field name="COMPONENT_SELECTOR">Ball<|ball_number|></field>
    <statement name="DO"> """,

  "</ball_flung>" : """ </statement>
  </block>""",

  "<ball_set_heading>" : """<block type="component_set_get" id="<|string_id|>">
        <mutation component_type="Ball" set_or_get="set" property_name="Heading" is_generic="false" instance_name="Ball<|ball_number|>"></mutation>
        <field name="COMPONENT_SELECTOR">Ball<|ball_number|></field>
        <field name="PROP">Heading</field>""",
  
  "</ball_set_heading>" : """</block>""",

  "<ball1_get_heading>" : """<value name="VALUE">
  <block type="lexical_variable_get" id="<|string_id|>">
            <mutation>
              <eventparam name="heading"></eventparam>
            </mutation>
            <field name="VAR">heading</field>
          </block>
          </value>""",

    "<next>" : """<next>""",

    "<ball_set_speed>" : """<block type="component_set_get" id="<|string_id|>">
            <mutation component_type="Ball" set_or_get="set" property_name="Speed" is_generic="false" instance_name="Ball<|ball_number|>"></mutation>
            <field name="COMPONENT_SELECTOR">Ball<|ball_number|></field>
            <field name="PROP">Speed</field>""",
             
    "<ball1_get_speed>" : """<value name="VALUE">
    <block type="lexical_variable_get" id="<|string_id|>">
                <mutation>
                  <eventparam name="speed"></eventparam>
                </mutation>
                <field name="VAR">speed</field>
              </block>
              </value>""",
      
   "</ball_set_speed>" : """</block>""",

   "</next>" : """</next>""",

  "<ball_edge_reached>" : """<block type="component_event" id="<|string_id|>" x="-372" y="4">
		<mutation component_type="Ball" is_generic="false" instance_name="Ball<|ball_number|>" event_name="EdgeReached"/>
		<field name="COMPONENT_SELECTOR">Ball<|ball_number|></field>
		<statement name="DO">""",

  "</ball_edge_reached>" : """</statement>
	</block>""",

  "<ball_bounce>" : """<block type="component_method" id="<|string_id|>">
				<mutation component_type="Ball" method_name="Bounce" is_generic="false" instance_name="Ball<|ball_number|>"/>
				<field name="COMPONENT_SELECTOR">Ball<|ball_number|></field>
				<value name="ARG0">""",
    
  "</ball_bounce>" : """</value>
			</block>""",

  "<get_edge>" : """<block type="lexical_variable_get" id="<|string_id|>">
						<mutation>
							<eventparam name="edge"/>
						</mutation>
						<field name="VAR">edge</field>
					</block>""", 
          
  "<number>" : """<value name="VALUE">
  <block type="math_number" id="<|string_id|>">
            <field name="NUM"><|number|></field>
          </block>
        </value>""",

  "<text>" : """<block type="text" id="<|string_id|>">
      <field name="TEXT"><|text|></field>
    </block>""",

  "<ball_set_color>" : """<block type="component_set_get" id="<|string_id|>">
        <mutation component_type="Ball" set_or_get="set" property_name="PaintColor" is_generic="false" instance_name="Ball<|ball_number|>"></mutation>
        <field name="COMPONENT_SELECTOR">Ball<|ball_number|></field>
        <field name="PROP">PaintColor</field>""",
  
  "</ball_set_color>" : """</block>""",

  "color" : """<value name="VALUE">
          <block type="color_<|color|>" id="<|string_id|>">
            <field name="COLOR"><|color_code|></field>
          </block>
        </value>""",

    "<ball_set_radius>" : """<block type="component_set_get" id="<|string_id|>">
        <mutation component_type="Ball" set_or_get="set" property_name="Radius" is_generic="false" instance_name="Ball<|ball_number|>"></mutation>
        <field name="COMPONENT_SELECTOR">Ball<|ball_number|></field>
        <field name="PROP">Radius</field>""",
    
    "</ball_set_radius>" : """</block>""",

    "<accelerometer1shaken>" : """<block type="component_event" id="<|string_id|>" x="-281" y="-97">
    <mutation component_type="AccelerometerSensor" is_generic="false" instance_name="AccelerometerSensor1" event_name="Shaking"></mutation> 
    <field name="COMPONENT_SELECTOR">AccelerometerSensor1</field>
    <statement name="DO">""",

    "</accelerometer1shaken>" : """</statement>
  </block>""",

  "<start_stop_video>" : """<block type="component_method" id="<|string_id|>">
        <mutation component_type="VideoPlayer" method_name="<|method|>" is_generic="false" instance_name="VideoPlayer<|number|>"></mutation>
        <field name="COMPONENT_SELECTOR">VideoPlayer<|number|></field>
      </block>""",

  "<switch_change>" : """<block type="component_event" id="<|string_id|>" x="-301" y="-137">
    <mutation component_type="Switch" is_generic="false" instance_name="Switch<|number|>" event_name="Changed"></mutation>
    <field name="COMPONENT_SELECTOR">Switch<|number|></field>
    <statement name="DO">""",
  
  "</switch_change>" : """</statement>
  </block>""",

  "<start_stop_player>" : """<block type="component_method" id="<|string_id|>">
        <mutation component_type="Player" method_name="<|method|>" is_generic="false" instance_name="Player<|number|>"></mutation>
        <field name="COMPONENT_SELECTOR">Player<|number|></field>
      </block>""",

  "<Screen>" : """<block type="controls_openAnotherScreen" id="<|string_id1|>">
        <value name="SCREEN">
          <block type="text" id="<|string_id2|>">
            <field name="TEXT">Screen<|number|></field>
          </block>
        </value>
      </block>""",

    "<time>" : """<block type="text_join" id="<|string_id|>1">
            <mutation items="2"></mutation>
            <value name="ADD0">
              <block type="text_join" id="<|string_id|>2">
                <mutation items="2"></mutation>
                <value name="ADD0">
                  <block type="component_set_get" id="<|string_id|>3">
                    <mutation component_type="TimePicker" set_or_get="get" property_name="Hour" is_generic="false" instance_name="TimePicker<|number|>"></mutation>
                    <field name="COMPONENT_SELECTOR">TimePicker<|number|></field>
                    <field name="PROP">Hour</field>
                  </block>
                </value>
                <value name="ADD1">
                  <block type="text" id="<|string_id|>4">
                    <field name="TEXT"> hours </field>
                  </block>
                </value>
              </block>
            </value>
            <value name="ADD1">
              <block type="text_join" id="<|string_id|>5">
                <mutation items="2"></mutation>
                <value name="ADD0">
                  <block type="component_set_get" id="<|string_id|>6">
                    <mutation component_type="TimePicker" set_or_get="get" property_name="Minute" is_generic="false" instance_name="TimePicker<|number|>"></mutation>
                    <field name="COMPONENT_SELECTOR">TimePicker<|number|></field>
                    <field name="PROP">Minute</field>
                  </block>
                </value>
                <value name="ADD1">
                  <block type="text" id="<|string_id|>7">
                    <field name="TEXT"> minutes</field>
                  </block>
                </value>
              </block>
            </value>
          </block>""",

      "<label>" : """<block type="component_set_get" id="<|string_id|>">
        <mutation component_type="Label" set_or_get="set" property_name="Text" is_generic="false" instance_name="Label<|number|>"></mutation>
        <field name="COMPONENT_SELECTOR">Label<|number|></field>
        <field name="PROP">Text</field>
        <value name="VALUE">""",

      "</label>" : """</value>
      </block>"""

}

text_and_number_dict = {}

"""# Generate UUIDs"""

import random
import string

neg_uuid_queue = [] #list
def negative_uuid_generator():
  for x in range(10):
    rand = (random.randint(100000000, 300000000))
    uuid = -rand
    #print (uuid)
    neg_uuid_queue.append(uuid)

pos_uuid_queue = []
def positive_uuid_generator():
  for x in range(10):
    rand = (random.randint(100000000, 300000000))
    uuid = rand
    pos_uuid_queue.append(uuid)

string_id_queue = []
def generateStringID(stringLength):
  lettersAndDigits = string.ascii_letters + string.digits
  for j in range(10):
    string_id = ''.join(random.choice(lettersAndDigits) for i in range(stringLength))
    string_id_queue.append(string_id)

"""# Utility Functions

"""

def get_visual_components_texts (start, end, vis_tokens):
  texts_list = []
  idx = -1
  for token in vis_tokens:
    idx += 1
    idx_end = idx
    if token == start:
      for item in vis_tokens[idx:]:
        idx_end += 1
        if item == end:
          break
      text = vis_tokens[idx+1:idx_end-1]
      text_str = text[0]
      if len(text) > 1:
        for word in text[1:]:
          text_str = text_str + " " + word
      texts_list.append(text_str)
  return texts_list

def modify_canvas_code(item, item_number):
  item_code = canvas_comp_dict[item]
  print(item_code)
  
  if len(neg_uuid_queue) == 0:
    negative_uuid_generator()
  if len(pos_uuid_queue) == 0:
    positive_uuid_generator()
  
  if item == "<ball>":
    uuid = pos_uuid_queue.pop(0)
    item_code = item_code.replace("<|ball_number|>", str(item_number)).replace("<|positive_uuid|>" , str(uuid))
  
  return item_code

def modify_vis_code(token, token_number, texts_dict, canvas_components):
  generic_code = vis_comp_dict[token]

  #make the texts lists here
  if texts_dict.get("button_texts") != None:
    button_texts = texts_dict["button_texts"]
  if texts_dict.get("switch_texts") != None:
    switch_texts = texts_dict["switch_texts"]
  if texts_dict.get("label_texts") != None:
    label_texts = texts_dict["label_texts"]
  if texts_dict.get("vid_srcs") != None:
    vid_srcs = texts_dict["vid_srcs"]
  if texts_dict.get("player_srcs") != None:
    player_srcs = texts_dict["player_srcs"]

  if len(neg_uuid_queue) == 0:
    negative_uuid_generator()
  if len(pos_uuid_queue) == 0:
    positive_uuid_generator()

  negative_uuid = neg_uuid_queue.pop(0)
  positive_uuid = pos_uuid_queue.pop(0)
  final_code = ""
  if token == "<text2speech>" or token == "<accelerometer>" or token == "<datepicker>" or token == "<timepicker>" or token == "<passwordtextbox>":
    final_code = (generic_code.replace("<|number|>", str(token_number))).replace("<|negative_uuid|>" , str(negative_uuid))
  elif token == "<textbox>":
    final_code = (generic_code.replace("<|number|>", str(token_number))).replace("<|negativeuuid|>" , str(negative_uuid))  
  elif token == "<video_player>":
    final_code = (generic_code.replace("<|number|>", str(token_number))).replace("<|negative_uuid|>" , str(negative_uuid)).replace("<|source|>", vid_srcs[token_number - 1])
  elif token == "<player>":
    final_code = (generic_code.replace("<|number|>", str(token_number))).replace("<|negative_uuid|>" , str(negative_uuid)).replace("<|source|>", player_srcs[token_number - 1])
  elif token == "<button>":
    final_code = (generic_code.replace("<|number|>" , str(token_number))).replace("<|negativeuuid|>" , str(negative_uuid)).replace("<|buttontext|>", button_texts[token_number - 1])
  elif token == "<switch>":
    final_code = (generic_code.replace("<|number|>" , str(token_number))).replace("<|positive_uuid|>" , str(positive_uuid)).replace("<|switch_text|>", switch_texts[token_number - 1])
  elif token == "<label>":
    final_code = (generic_code.replace("<|number|>" , str(token_number))).replace("<|positive_uuid|>" , str(positive_uuid)).replace("<|label_text|>", label_texts[token_number - 1])
  elif token == "<canvas>":
    semi_final_code = generic_code.replace("<|canvas_number|>" , str(token_number)).replace("<|positive_uuid|>", str(positive_uuid))
    #Get the codes for the components inside the canvas
    canvas_components_list = []
    ball_number = 0
    for item in canvas_components:
      if item == "<ball>":
        ball_number += 1
        item_number = ball_number
        canvas_component_code = modify_canvas_code(item, item_number)
        canvas_components_list.append(canvas_component_code)
    canvas_components_code_final = str(canvas_components_list).replace("'" , "")
    #Get the final vis_comp code (entry in scm) for the canvas
    final_code = semi_final_code.replace("<|canvas_components|>" , canvas_components_code_final)
    print("Final code of canvas components:" + final_code)
  return final_code

def modifyLogicCode(token, token_number=None, method=None, text_or_number=None):
  code = logic_dic[token]
  if len(string_id_queue) == 0:
      generateStringID(20)
  string_id = string_id_queue.pop(0)
  if token == "<button_click>" or token == "<switch_change>" or token == "<label>":
    code = code.replace("<|string_id|>", string_id).replace("<|number|>" , str(token_number))
  elif token == "<text>" or token == "<number>":
    code = code.replace("<|string_id|>", string_id).replace("<|text|>", str(text_or_number)).replace("<|number|>", str(text_or_number))
  elif token == "<text2speech>":
    code = code.replace("<|string_id|>" , string_id).replace("<|text2speech_number|>" , str(token_number))
  elif token == "<textbox_text>":
    code = code.replace("<|string_id|>" , string_id).replace("<|textbox_number|>", str(token_number))
  elif token == "<ball_flung>":
    code = code.replace("<|string_id|>" , string_id).replace("<|ball_number|>", str(token_number))
  elif token == "<ball_set_heading>":
    code = code.replace("<|string_id|>" , string_id).replace("<|ball_number|>", str(token_number))
  elif token == "<ball_get_heading>":
    code = code.replace("<|string_id|>" , string_id)
  elif token == "<ball_set_speed>":
    code = code.replace("<|string_id|>" , string_id).replace("<|ball_number|>" , str(token_number))
  elif token == "<ball_get_speed>" or token == "<get_edge>":
    code = code.replace("<|string_id|>" , string_id)
  elif token == "<ball_edge_reached>" or token == "<ball_bounce>" or token == "<ball_set_color>" or token == "<ball_set_radius>":
    code = code.replace("<|string_id|>" , string_id).replace("<|ball_number|>" , str(token_number))
  elif token == "color":
    code = code.replace("<|string_id|>", string_id)
  elif token == "<accelerometer1shaken>":
    code = code.replace("<|string_id|>", string_id)
  elif token == "<start_stop_video>" or token == "<start_stop_player>":
    #print("Method inside modify: " + method)
    code = code.replace("<|string_id|>", string_id).replace("<|number|>", str(token_number)).replace("<|method|>", method)
  elif token == "<Screen>":
    string_id1 = string_id
    if len(string_id_queue) == 0:
      generateStringID(20)
    string_id2 = string_id_queue.pop(0)
    code = code.replace("<|string_id1|>", string_id1).replace("<|string_id2|>", string_id2).replace("<|number|>", str(token_number))
  elif token == "<time>":
    generateStringID(20)
    
    for i in range(7):
      to_replace = "<|string_id|>" + str(i + 1)
      code = code.replace(to_replace, string_id_queue.pop(0))
    
    code = code.replace("<|number|>", str(token_number))

  return code

def is_Number(test):
  is_number = True
  try:
    float(test)
  except ValueError:
    is_number = False
  return is_number

def handle_ball(code_tokens, start_index, ball_number, bky, event):
  #print("Handling code tokens of ball number: " + str(ball_number))

  event_end = event[:1] + "/" + event[1:]
  for i in range(start_index, len(code_tokens)):
    if code_tokens[i] == event_end:
      event_end_idx = i
      break
   

  ball_start = code_tokens[start_index]
  ball_end = "</ball" + str(ball_number) + ">"

  next_required_1 = False
  next_required_2 = False
  do_implicit_bounce = False
  implicit_bounce_iter = None
  implicit_bounce_ball_number = None

  i = start_index + 1
  token = code_tokens[i]
  while token != ball_end:
    #print("  Ball token: " + token)
    if token == "<motion>" or token == "<bounce>":
      #set speed to 10
      #set direction to 10
      bky += modifyLogicCode("<ball_set_heading>", ball_number)
      bky += "\n"
      bky += modifyLogicCode("<number>", text_or_number="10")
      bky += "\n"
      bky += logic_dic["<next>"]
      bky += "\n"
      bky += modifyLogicCode("<ball_set_speed>", ball_number)
      bky += "\n"
      bky += modifyLogicCode("<number>", text_or_number="10")
      bky += "\n"
      bky += modifyLogicCode("</ball_set_speed>", ball_number)
      bky += "\n"
      bky += logic_dic["</next>"]
      bky += "\n"
      bky += modifyLogicCode("</ball_set_heading>", ball_number)
      bky += "\n"

      if token == "<bounce>":
        do_implicit_bounce = True
        implicit_bounce_iter = event_end_idx + 1
        implicit_bounce_ball_number = ball_number

    elif  token == "<heading>":
      if next_required_1 and next_required_2:
        bky += logic_dic["<next>"]
        bky += "\n"
      bky += modifyLogicCode("<ball_set_heading>", ball_number)
      bky += "\n"
      next_required_1 = True
    elif  token == "</heading>":
      bky += modifyLogicCode("</ball_set_heading>", ball_number)
      bky += "\n"
      if next_required_1 and next_required_2:
        bky += logic_dic["</next>"]
        bky += "\n"
      next_required_2 = True
    elif "number" in token:
      number = str(text_and_number_dict[token])
      bky += modifyLogicCode("<number>", text_or_number=number)
      bky += "\n"
    elif token == "<speed>":
      if next_required_1 and next_required_2:
        bky += logic_dic["<next>"]
        bky += "\n"
      bky += modifyLogicCode("<ball_set_speed>", ball_number)
      bky += "\n"
      next_required_1 = True
    elif token == "</speed>":
      bky += modifyLogicCode("</ball_set_speed>", ball_number)
      bky += "\n"
      if next_required_1 and next_required_2:
        bky += logic_dic["</next>"]
        bky += "\n"
      next_required_2 = True
    
    elif token == "<radius>":
      if next_required_1 and next_required_2:
        bky += logic_dic["<next>"]
        bky += "\n"
      bky += modifyLogicCode("<ball_set_radius>", ball_number)
      bky += "\n"
      next_required_1 = True
    
    elif token == "</radius>":
      bky += modifyLogicCode("</ball_set_radius>", ball_number)
      bky += "\n"
      if next_required_1 and next_required_2:
        bky += logic_dic["</next>"]
        bky += "\n"
      next_required_2 = True

    elif token == "<color>":
      if next_required_1 and next_required_2:
        bky += logic_dic["<next>"]
        bky += "\n"
      bky += modifyLogicCode("<ball_set_color>", ball_number)
      bky += "\n"
      next_required_1 = True
    
    elif token == "</color>":
      bky += modifyLogicCode("</ball_set_color>", ball_number)
      bky += "\n"
      if next_required_1 and next_required_2:
        bky += logic_dic["</next>"]
        bky += "\n"
      next_required_2 = True
    
    elif token in color_dic.keys():
      temp_code = modifyLogicCode("color")
      color_code = color_dic[token]
      color = token[1:-1]
      temp_code = temp_code.replace("<|color|>", color).replace("<|color_code|>", str(color_code))
      bky += temp_code
      bky += "\n"
    
    i = i + 1
    token = code_tokens[i]
  return bky, i, do_implicit_bounce, implicit_bounce_iter, implicit_bounce_ball_number

def compile_scm_bky(tokens, screen_number, username="ksmehrab", project_name="test"):
  vis_tokens = tokens[1:tokens.index("</complist>")]
  #print(vis_tokens)

  number_of_canvas = vis_tokens.count("<canvas>")
  number_of_balls = vis_tokens.count("<ball>")

  #Get the middle texts here
  texts_dict = {}
  button_texts = []
  if "<button>" in vis_tokens:
    button_text_identifiers = get_visual_components_texts("<button>", "</button>", vis_tokens) 
    for button_text_identifier in button_text_identifiers:
      button_texts.append(text_and_number_dict[button_text_identifier]) 
  if len(button_texts) != 0:
    texts_dict["button_texts"] = button_texts

  switch_texts = []
  if "<switch>" in vis_tokens:
    switch_text_identifiers = get_visual_components_texts("<switch>", "</switch>", vis_tokens) 
    for switch_text_identifier in switch_text_identifiers:
      switch_texts.append(text_and_number_dict[switch_text_identifier]) 
  if len(switch_texts) != 0:
    texts_dict["switch_texts"] = switch_texts


  label_texts = []
  if "<label>" in vis_tokens:
    label_text_identifiers = get_visual_components_texts("<label>", "</label>", vis_tokens) 
    for label_text_identifier in label_text_identifiers:
      label_texts.append(text_and_number_dict[label_text_identifier]) 
  if len(label_texts) != 0:
    texts_dict["label_texts"] = label_texts


  vid_src_list = []
  if "<video_player>" in vis_tokens:
    vid_src_identifier_list = get_visual_components_texts("<video_player>", "</video_player>", vis_tokens)
    for vid_src_identifier in vid_src_identifier_list:
      vid_src_list.append(text_and_number_dict[vid_src_identifier])
  if (len(vid_src_list) != 0):
    texts_dict["vid_srcs"] = vid_src_list

  
  player_src_list = []
  if "<player>" in vis_tokens:
    player_src_identifier_list = get_visual_components_texts("<player>", "</player>", vis_tokens)
    for player_src_identifier in player_src_identifier_list:
      player_src_list.append(text_and_number_dict[player_src_identifier])
  if (len(player_src_list) != 0):
    texts_dict["player_srcs"] = player_src_list

  idx = -1
  #For canvas and ball, get the canvas components
  canvas_components = []
  for token in vis_tokens:
    idx += 1
    start_canvas = "<canvas>"
    end_canvas = "</canvas>"
    idx_end = idx
    if token == start_canvas:
      for item in vis_tokens[idx:]:
        idx_end += 1
        if item == end_canvas:
          break
      canvas_components = vis_tokens[idx+1:idx_end-1]
  
  need_assets = {
      "<video_player>" : False,
      "<player>" : False
  }
  items_with_assets = ["<video_player>", "<player>"]
  for token in items_with_assets:
    if token in vis_tokens:
      need_assets[token] = True
  vis_comp_number = {
      "<textbox>": 0,
      "<button>": 0,
      "<text2speech>": 0,
      "<accelerometer>": 0,
      "<canvas>": 0,
      "<video_player>": 0,
      "<player>": 0,
      "<switch>": 0,
      "<label>": 0,
      "<datepicker>": 0,
      "<timepicker>": 0,
      "<passwordtextbox>": 0
  }

  #Generating vis comp code starts here
  vis_components = []

  for token in vis_tokens:
    #print("Handling component token: " + token)
    if vis_comp_number.get(token) != None:
      vis_comp_number[token] += 1
      component = modify_vis_code(token, vis_comp_number[token], texts_dict, canvas_components)
      vis_components.append(component)

  vis_components_code = str(vis_components) 
  vis_components_code_final = vis_components_code.replace("'" , "")
  #print("Final components code:\n" + vis_components_code_final)
  scm = """#|
$JSON
{"authURL":["ai2.appinventor.mit.edu"],"YaVersion":"208","Source":"Form","Properties":{"$Name":"Screen<|screen_number|>","$Type":"Form","$Version":"27","AppName":"<|app_name|>","Title":"Screen<|screen_number|>","Uuid":"0","$Components":<|components|>
}}
|#"""
  scm = scm.replace("<|components|>", vis_components_code_final).replace("<|app_name|>" , project_name).replace("<|screen_number|>", str(screen_number))
  #print("Final SCM:\n" + scm)
###########################################################################################################################################################################################
  #generating block (bky) codes starts here
  if "<code>" in tokens:
    code_tokens = tokens[tokens.index("<code>"):tokens.index("</code>")+1]

    button_number = 0
    text2speech_number = 0
    textbox_number = 0

    button_click_regex = re.compile(r'<button\dclicked>')
    button_click_end_regex = re.compile(r'</button\dclicked>')

    switch_change_regex = re.compile(r'<switch\dflipped>')
    switch_change_end_regex = re.compile(r'</switch\dflipped>')


    ball_edge_reached_regex = re.compile(r'<ball\dreach_edge>')
    ball_edge_reached_end_regex = re.compile(r'</ball\dreach_edge>')

    ball_flung_regex = re.compile(r'<ball\dflung>')
    ball_flung_end_regex = re.compile(r'</ball\dflung>')

    ball_regex = re.compile(r'<ball\d>')

    screen_change_regex = re.compile(r'<Screen\d>')
    
    video_player_regex = re.compile(r'<video_player\d>')
    player_regex = re.compile(r'<player\d>')
    time_regex = re.compile(r'<time\d>')

    label_regex = re.compile(r'<label\d>')
    label_end_regex = re.compile(r'</label\d>')

    textbox_text_regex = re.compile(r'<textboxtext\d>')

    text2speech_regex = re.compile(r'<text2speech\d>')
    text2speech_end_regex = re.compile(r'</text2speech\d>')

    bky = ""
    
    #for token in code_tokens:
    do_implicit_bounce = False
    implicit_bounce_iter = None
    implicit_bounce_ball_number = None
    i = 0
    while (i < len(code_tokens)):
      if do_implicit_bounce and (i == implicit_bounce_iter) and (implicit_bounce_ball_number != None):
        number = implicit_bounce_ball_number
        bky += modifyLogicCode("<ball_edge_reached>", number)
        bky += "\n"
        bky +=  modifyLogicCode("<ball_bounce>", number)
        bky += "\n"
        bky += modifyLogicCode("<get_edge>", number)
        bky += "\n"
        bky +=  modifyLogicCode("</ball_bounce>", number)
        bky += "\n"
        bky += modifyLogicCode("</ball_edge_reached>", number)
        bky += "\n"   

        do_implicit_bounce = False
        implicit_bounce_iter = None
        implicit_bounce_ball_number = None     

      token = code_tokens[i]

      #print("Handling code token: " + token)
    
      if token[0] == "<": #So it's is a normal token
        if token == "<button_click>":
          button_number += 1
          bky += modifyLogicCode(token, button_number)
          bky += "\n"

        elif ball_regex.match(token) != None:
          br = ball_regex.match(token)
          br = str(br.group())
          number = br.split("ball")[1][0]

          bky, i, do_implicit_bounce, implicit_bounce_iter, implicit_bounce_ball_number = handle_ball(code_tokens, i, number, bky, code_tokens[i - 1])
          
        elif ball_flung_regex.match(token) != None:
          br = ball_flung_regex.match(token)
          br = str(br.group())
          ball_number = br.split("ball")[1][0]

          dict_token = "<ball_flung>"
          bky += modifyLogicCode(dict_token, ball_number)
          bky += "\n"

        elif ball_flung_end_regex.match(token) != None:
          bky += logic_dic["</ball_flung>"]
          bky += "\n"

        elif ball_edge_reached_regex.match(token) != None:
          br = ball_edge_reached_regex.match(token)
          br = str(br.group())
          ball_number = br.split("ball")[1][0]

          dict_token = "<ball_edge_reached>"
          bky += modifyLogicCode(dict_token, ball_number)
          bky += "\n"

        elif ball_edge_reached_end_regex.match(token) != None:
          bky += logic_dic["</ball_edge_reached>"]
          bky += "\n"

        elif text2speech_regex.match(token) != None:
          br = text2speech_regex.match(token)
          br = str(br.group())
          text2speech_number = br.split("text2speech")[1][0]

          dict_token = "<text2speech>"
          bky += modifyLogicCode(dict_token, text2speech_number)
          bky += "\n"
        
        elif text2speech_end_regex.match(token) != None:
          bky += logic_dic["</text2speech>"]
          bky += "\n"

        elif token == "<textbox_text>":
          textbox_number += 1
          bky += modifyLogicCode(token, textbox_number)
          bky += "\n"

        elif button_click_regex.match(token) != None:
          br = button_click_regex.match(token)
          br = str(br.group())
          button_number = br.split("button")[1][0]

          dict_token = "<button_click>"
          bky += modifyLogicCode(dict_token , button_number)
          bky += "\n" 

        elif button_click_end_regex.match(token) != None:
          dict_token = "</button_click>"
          bky += modifyLogicCode(dict_token , button_number)
          bky += "\n"
        
        elif switch_change_regex.match(token) != None:
          br = switch_change_regex.match(token)
          br = str(br.group())
          switch_number = br.split("switch")[1][0]

          dict_token = "<switch_change>"
          bky += modifyLogicCode(dict_token , switch_number)
          bky += "\n"

        elif switch_change_end_regex.match(token) != None:
          bky += logic_dic["</switch_change>"]
          bky += "\n"

        elif textbox_text_regex.match(token) != None:
          br = textbox_text_regex.match(token)
          br = str(br.group())
          textbox_number = br.split("textboxtext")[1][0]

          dict_token = "<textbox_text>"
          bky += modifyLogicCode(dict_token , textbox_number)
          bky += "\n"


        elif screen_change_regex.match(token) != None:
          br = screen_change_regex.match(token)
          br = str(br.group())
          number = br.split("Screen")[1][0]

          dict_token = "<Screen>"
          bky += modifyLogicCode(dict_token, number)
          bky += "\n"

        elif token == "<get_edge>":
          bky += modifyLogicCode(token, 0)
          bky += "\n"
        
        elif token ==  "<accelerometer1shaken>":
          bky += modifyLogicCode(token, 0)
          bky += "\n"

        elif video_player_regex.match(token) != None:
          br = video_player_regex.match(token)
          br = str(br.group())
          number = br.split("video_player")[1][0]
          dict_token = "<start_stop_video>"
          method = code_tokens[i + 1][1:-1].capitalize()
          #print("Method: " + method)
          bky += modifyLogicCode(dict_token, number, method)
          bky += "\n"

          i = i + 2
        
        elif player_regex.match(token) != None:
          br = player_regex.match(token)
          br = str(br.group())
          number = br.split("player")[1][0]
          dict_token = "<start_stop_player>"
          method = code_tokens[i + 1][1:-1].capitalize()
          bky += modifyLogicCode(dict_token, number, method)
          bky += "\n"

          i = i + 2
        
        elif time_regex.match(token) != None:
          br = time_regex.match(token)
          br = str(br.group())
          number = br.split("time")[1][0]
          dict_token = "<time>"
  
          bky += modifyLogicCode(dict_token, number)
          bky += "\n"

        elif label_regex.match(token) != None:
          br = label_regex.match(token)
          br = str(br.group())
          number = br.split("label")[1][0]
          dict_token = "<label>"
  
          bky += modifyLogicCode(dict_token, number)
          bky += "\n"
        
        elif label_end_regex.match(token) != None:  
          bky += logic_dic["</label>"]
          bky += "\n"
      
        else:
          bky += logic_dic[token]
          bky += "\n"

      else: #It is either a number or a text alone
        if "number" in token:
          number = str(text_and_number_dict[token])
          bky += modifyLogicCode("<number>", text_or_number=number)
          bky += "\n"
        else:
          text = text_and_number_dict[token]
          bky += modifyLogicCode("<text>", text_or_number=text)
          bky += "\n"
      i = i + 1
  else:
    bky=""

  #print("Final BKY:\n" + bky)

  if not os.path.exists('./myapp/src/appinventor/ai_{0}/{1}'.format(username, project_name)):
    os.makedirs('./myapp/src/appinventor/ai_{0}/{1}'.format(username, project_name))

  if not os.path.exists('./myapp/youngandroidproject'):
    os.makedirs('./myapp/youngandroidproject')

  if need_assets["<video_player>"]:
    os.makedirs('./myapp/assets')
    for vid_src_str in vid_src_list:
      if os.path.exists("./Media/Videos/" + vid_src_str):
        shutil.copy("./Media/Videos/" + vid_src_str, "./myapp/assets")
      else:
        #raise error
        print("Video Asset " + vid_src_str + " not found!")
      
  if need_assets["<player>"]:
    if not os.path.exists('./myapp/assets'):
      os.makedirs('./myapp/assets')
    for player_src_str in player_src_list:
      if os.path.exists("./Media/Music/" + player_src_str):
        shutil.copy("./Media/Music/" + player_src_str, "./myapp/assets")
      else:
        #raise error
        print("Audio Asset " + player_src_str + " not found!")

  bky_file = open('./myapp/src/appinventor/ai_{0}/{1}/Screen{2}.bky'.format(username, project_name, str(screen_number)), 'w+')
  scm_file = open('./myapp/src/appinventor/ai_{0}/{1}/Screen{2}.scm'.format(username, project_name, str(screen_number)), 'w+')

  bky_file.write(bky)
  scm_file.write(scm)

  bky_file.close()
  scm_file.close()

def enclose_with_canvas(SAR):
  comps_inside_canvas = ["<ball>"]
  tokens = SAR.split()

  for comp in comps_inside_canvas:
    if comp not in tokens:
      return SAR

  canvas_comps = []
 
  start_comp_idx = tokens.index("<complist>")
  end_comp_idx = tokens.index("</complist>")
  for i in range (start_comp_idx, end_comp_idx):
    if tokens[i] in comps_inside_canvas:
      canvas_comps.append(tokens[i])
      tokens[i] = "<ignore>"

  canvas_start_idx = start_comp_idx + 1
  canvas_end_idx = canvas_start_idx + len(canvas_comps) + 1;
  tokens.insert(canvas_start_idx, "<canvas>")
  for i in range(len(canvas_comps)):
    comp = canvas_comps[i]
    tokens.insert(canvas_start_idx + 1, comp)
  tokens.insert(canvas_end_idx, "</canvas>")
  #print(tokens)

  modified_SAR = ""
  for token in tokens:
    if token != "<ignore>":
      modified_SAR += token
      modified_SAR += " "

  if modified_SAR[-1] == " ":
    modified_SAR = modified_SAR[:-1]

  return modified_SAR

def sar_to_aia(t2a, username="anonymuser", project_name="test"):
  original_SAR, text_num_dict = t2a.SAR, t2a.literal_dict
  global text_and_number_dict
  text_and_number_dict = text_num_dict

  if os.path.exists('./myapp'):
    subprocess.call("rm -r myapp", shell=True)

  #original_SAR = "<complist> <player> string6 </player> <switch> string1 </switch> <textbox> <video_player> string8 </video_player> <accelerometer> <switch> string3 </switch> </complist> <code> <switch1flipped> <video_player1> <stop> </video_player1> </switch1flipped> <accelerometer1shaken> <player1> <start> </player1> </accelerometer1shaken> </code>"
  screen_number = 1
  if "<screen>" in original_SAR:
    SARs = original_SAR.split(" <screen> ")
    for SAR in SARs:
      #modify SAR by enclosing with canvas if necessary
      #The necessity check is performed INSIDE the following function
      SAR = enclose_with_canvas(SAR)
      #print("Modified SAR with canvas: " + SAR)
      tokens = SAR.split()
      compile_scm_bky(tokens, screen_number, username, project_name)
      screen_number += 1
  else:
    SAR = original_SAR
    SAR = enclose_with_canvas(SAR)
    #print("Modified SAR with canvas: " + SAR)
    tokens = SAR.split()
    compile_scm_bky(tokens, 1, username, project_name)

  base_properties = """main=appinventor.ai_{0}.{1}.Screen1
  name={1}
  assets=../assets
  source=../src
  build=../build
  versioncode=1
  versionname=1.0
  useslocation=False
  aname={1}
  sizing=Responsive
  showlistsasjson=True
  actionbar=False
  theme=Classic
  color.primary=&HFF3F51B5
  color.primary.dark=&HFF303F9F
  color.accent=&HFFFF4081
  """

  project_properties = base_properties.format(username, project_name)
  properties = open('./myapp/youngandroidproject/project.properties', 'w+')
  properties.write(project_properties)
  properties.close()


  os.chdir("./myapp")
  subprocess.call('zip -r myapp.zip *', shell=True)
  subprocess.call('mv myapp.zip ..', shell=True)
  os.chdir("..")
  instruction = "mv myapp.zip " + project_name + ".aia"
  subprocess.call(instruction, shell=True)

