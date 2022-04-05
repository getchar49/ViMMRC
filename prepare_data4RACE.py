# -*- coding: utf-8 -*-
import json
import random

from transformers import AutoTokenizer

from utils import *

tokenizer = None

import os


def create_obj(data_path,model_type):
  #global tokenizer
  data = json.load(open(data_path))
  tail = data_path.split('/')[-1].split('.')[0]
  data_path = '/'.join(data_path.split('/')[:-1])
  result = []
  tokenizer.model_max_length = 512
  for one in tqdm(data):
    alternatives = one['alternatives'].split('|')
    alternatives = one['alternatives'].split('|')
    if len(alternatives) != 4:
          continue
    label = int(one['answer'])
    query = one['query']
    text_a = one['passage']
    choices_inputs = []
    for ending in alternatives:
        if query.find("_") != -1:
            text_b = query.replace("_", ending)
        else:
            text_b = query + " " + ending
        inputs = {}
        inputs["input_ids"] =  [tokenizer.cls_token_id] \
                            +  tokenizer.encode(text_b,add_special_tokens=False,max_length=128,truncation=True) \
                            +  [tokenizer.sep_token_id]
        inputs["token_type_ids"] = [0] * len(inputs["input_ids"])
        inputs["input_ids"] += tokenizer.encode(text_a, max_length=tokenizer.model_max_length - len(inputs["input_ids"])-1,truncation=True,add_special_tokens=False) +  [tokenizer.sep_token_id]
        inputs["attention_mask"] = [1] * len(inputs["input_ids"])
        inputs["token_type_ids"] += [1] * (len(inputs["input_ids"]) - len(inputs["token_type_ids"]))
  #        inputs = tokenizer(
  #            text_b,
  #            text_a,
  #            add_special_tokens=True,
  #            max_length=tokenizer.max_len,
  #            #padding="max_length",
  #            truncation=True,
  #            return_overflowing_tokens=True,
  #        )
        
        choices_inputs.append(inputs)
    input_ids = [x["input_ids"] for x in choices_inputs]
    attention_mask = (
        [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None)
    token_type_ids = (
        [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None)
    
    global_attention_mask = None 
    result.append([input_ids, label, attention_mask, token_type_ids,global_attention_mask])
  f = open(os.path.join(data_path,tail+'.{}.obj'.format(model_type.replace('/', '.'))), 'wb')
  pickle.dump(result, f)
  
  
def prepare_bert_data(model_type,data_path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if not os.path.exists(os.path.join(data_path,'test.{}.obj'.format(model_type.replace('/', '.')))):
      create_obj(os.path.join(data_path,'test.json'),model_type)
    if not os.path.exists(os.path.join(data_path,'dev.{}.obj'.format(model_type.replace('/', '.')))):
      create_obj(os.path.join(data_path,'dev.json'),model_type)
    if not os.path.exists(os.path.join(data_path,'train.{}.obj'.format(model_type.replace('/', '.')))):
      create_obj(os.path.join(data_path,'train.json'),model_type)
