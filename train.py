import argparse
import torch

from transformers import AutoTokenizer, AutoModel

from utils import *
import torch.distributed as dist
import numpy as np
import logging
import gc
import math
from tqdm import tqdm
import os
import sys
from evaluate_metrics import evaluate
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class_num = {
    "ReCO":3,
    "RACE":4,
    "book_review":2,
    "lcqmc":2,
    "ag_news":4,
    "dbpedia":14
    }
    
record = {'loss':[],'avg_loss':[],'val_acc':[],'optim':None,'sche':None,'epoch':None,'total_loss':0,'best_acc':0.24}

torch.manual_seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=2.0e-5)
parser.add_argument("--max_grad_norm", type=float, default=0.2)
parser.add_argument("--model_type", type=str, default="voidful/albert_chinese_base")
parser.add_argument("--data_path", type=str)
parser.add_argument(
    "--fp16",
    action="store_true",
#    default=True,
)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
) 
parser.add_argument(
    "--output_dir",
    type=str,
    help="save path of models"
)
parser.add_argument(
    "--alpha",
    type=float,
    required=True
)
parser.add_argument(
    "--warmup_proportion",
    type=float,
    default = None
)
parser.add_argument(
    "--split_layer",
    type=int,
    default = 12
)


args = parser.parse_args()
logging.info(args)
data_path = args.data_path
model_type = args.model_type
local_rank = args.local_rank

dataset_type = "RACE"
# import the data process code
exec("from prepare_data4{} import prepare_bert_data".format(dataset_type))



prepare_bert_data(model_type, data_path)

"""
data = load_file(os.path.join(data_path,'train.{}.obj'.format(model_type.replace('/', '.'))))
valid_data = load_file(os.path.join(data_path,'valid.{}.obj'.format(model_type.replace('/', '.'))))
valid_data = sorted(valid_data, key=lambda x: len(x[0]))
"""
with open(os.path.join(data_path,'test.{}.obj'.format(model_type.replace('/', '.'))),"rb") as f:
  test_data = pickle.load(f)
with open(os.path.join(data_path,'train.{}.obj'.format(model_type.replace('/', '.'))),"rb") as f:
  data = pickle.load(f)
with open(os.path.join(data_path,'dev.{}.obj'.format(model_type.replace('/', '.'))),"rb") as f:
  valid_data = pickle.load(f)
batch_size = args.batch_size
#62444
total_size = (len(data)+1)//batch_size
#print("Total size: ",total_size)
#total_size = 31222
tokenizer = AutoTokenizer.from_pretrained(model_type)
tokenizer.model_max_length = 512
total_loss = 0


from model import Bert4RACE as bertmodel
model = bertmodel(model_type, layers=args.split_layer).cuda() 

layers = model.encoder.config.num_hidden_layers

optimizer = optim(layers,args.lr,model,args.alpha,split_layers=args.split_layer)

if args.warmup_proportion is not None and args.warmup_proportion!=0.0:
    total_steps = len(data) * args.epoch // (args.gradient_accumulation_steps * args.batch_size)
    #print(total_steps)
    #total_steps = 39027
    scheduler = get_linear_schedule_with_warmup(optimizer,int(args.warmup_proportion*total_steps),total_steps)

def get_shuffle_data():
    pool = {}
    for one in data:
        length = len(one[0]) // 5
        if length not in pool:
            pool[length] = []
        pool[length].append(one)
    for one in pool:
        np.random.shuffle(pool[one])
    length_lst = list(pool.keys())
    np.random.shuffle(length_lst)
    whole_data = [x for y in length_lst for x in pool[y]]
    return whole_data

def train(epoch,offset=0):
    global total_loss
    #print(total_loss)
    #print(total_loss)
    model.train()
    train_data = get_shuffle_data()
    total = len(train_data)
    step = offset
    losses = []
    avg_losses = []
    for i in range(0,total,batch_size):
        seq = [x[0] for x in train_data[i:i + batch_size]]
        label = [x[1] for x in train_data[i:i + batch_size]]
        
        attention_mask = [x[2] for x in train_data[i:i + batch_size]]
       
        if "roberta" in model_type.lower() or "longformer" in model_type.lower():
            token_type_ids = None
        else:
            token_type_ids = [x[3] for x in train_data[i:i + batch_size]]
        
        seq, attention_mask,token_type_ids = \
            RACE_padding(seq, pads=tokenizer.pad_token_id, max_len=512,attention_mask = attention_mask, token_type_ids = token_type_ids)
        seq = torch.LongTensor(seq).cuda()
        attention_mask = torch.LongTensor(attention_mask).cuda() if attention_mask is not None else None
        token_type_ids = torch.LongTensor(token_type_ids).cuda() if token_type_ids is not None else None
        label = torch.LongTensor(label).cuda()
        
        loss = model([seq, label, attention_mask, token_type_ids])

        loss = loss / args.gradient_accumulation_steps

        loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            if args.warmup_proportion is not None and args.warmup_proportion!=0.0:
                scheduler.step()
        step += 1
        #print(loss)
        total_loss += loss
        avg_loss = total_loss/step
        if (step%50==0):
            losses.append(loss)
            avg_losses.append(avg_loss)
            print('Epoch {}: Step {}/{}'.format(epoch,step,total_size))
            print("loss: {}, avg loss: {}".format(loss,avg_loss))
    record['loss'].append(losses)
    record['avg_loss'].append(avg_losses)
    
def evaluation(epoch):
    model.eval()
    total = len(valid_data)
    
    all_prediction = []
    all_predictions = []
    for i in range(args.split_layer):
        all_predictions.append([])
    labels = [x[1] for x in valid_data]
    with torch.no_grad():
        for i in iter_printer(total, epoch):
            seq = [x[0] for x in valid_data[i:i + batch_size]]
            
            attention_mask = [x[2] for x in valid_data[i:i + batch_size]]
#            token_type_ids = [x[3] for x in valid_data[i:i + batch_size]]

            if "roberta" in model.encoder.config.model_type.lower() or "longformer" in model_type.lower():
                token_type_ids = None
            else:
                token_type_ids = [x[3] for x in valid_data[i:i + batch_size]]            

            seq, attention_mask,token_type_ids = \
                RACE_padding(seq, pads=tokenizer.pad_token_id, max_len=512,attention_mask = attention_mask, token_type_ids = token_type_ids)
            seq = torch.LongTensor(seq).cuda()
            attention_mask = torch.LongTensor(attention_mask).cuda() if attention_mask is not None else None
            token_type_ids = torch.LongTensor(token_type_ids).cuda() if token_type_ids is not None else None
            
            predictions,probs = model([seq, None, attention_mask, token_type_ids])
            
            predictions = [prediction.cpu().tolist() for prediction in predictions]
            probs = [prob.cpu().numpy() for prob in probs]
            
            for j in range(probs[0].shape[0]):
                ans = -1
                max_prob = 0
                for k in range(len(probs)):
                    if max(probs[k][j]) >= max_prob:
                        ans = predictions[k][j]
                        max_prob = max(probs[k][j])
                all_prediction.append(ans)
            for k,prediction in enumerate(predictions):
                all_predictions[k].extend(prediction)
    rights,right = evaluate(all_prediction,all_predictions,labels,"acc") 
    print('epoch {} eval acc is {}'.format(epoch, right))
    for k in range(len(probs)):
        print('layer {} eval acc is {}'.format(k,rights[k]))
    global record
    record['val_acc'].append(rights[:3])
    return right
cur_epo = 0
cur_i = 0
def load_saved_state():
    global record
    record = torch.load(os.path.join(args.output_dir,'log2.pt'))
    scheduler.load_state_dict(record['sche'])
    optimizer.load_state_dict(record['optim'])
    best_acc = record['best_acc']
    state_dict = torch.load(os.path.join(args.output_dir,'tmp_model.th'))
    model.load_state_dict(state_dict)
    global cur_i
    global cur_epo
    cur_i = record['cur_i']
    cur_epo = record['cur_epo']
    #print(cur_i,cur_epo)

#best_acc = evaluation(-1)
best_acc = 0.24
c = 4000
num = len(data)//c+1
#num = 16
#model.load_state_dict(torch.load(os.path.join(args.output_dir,'tmp_model.th')))
print("Training...")
#print(scheduler.state_dict())
#load_saved_state()
print(cur_i,cur_epo)
#cur_epo = 5
#cur_i = 14
#cur_i = record['cur_i']
total_loss = record['total_loss']
#total_loss.requires_grad = False
print(total_loss)
for epo in range(cur_epo-1,args.epoch):
    for i in range(cur_i,num):
      
        data = load_file(os.path.join(data_path,'train.{}.obj'.format(model_type.replace('/', '.'))))[c*i:c*(i+1)]
        train(epo+1,c*i//batch_size)
        state_dict = model.state_dict()
        torch.save(state_dict,os.path.join(args.output_dir,'tmp_model.th'))
        record['optim'] = optimizer.state_dict()
        record['sche'] = scheduler.state_dict()
        record['cur_i'] = i+1
        record['cur_epo'] = epo + 1
        record['total_loss'] = total_loss
        #print(record)
        torch.save(record,os.path.join(args.output_dir,'log2.pt'))
    record['cur_i'] = 0
    record['cur_epo'] = epo + 2
    if local_rank == -1 or local_rank == 0:
        accuracy = evaluation(epo)
        #record['val_acc'].append(accuracy)
        record['optim']=optimizer.state_dict()
        record['sche']=scheduler.state_dict()
        #record['cur_i'] = cur_i
        #record['cur_epo'] = cur_epo
        torch.save(record,os.path.join(args.output_dir,'log2.pt'))
        if accuracy > best_acc:
            best_acc = accuracy
            record['best_acc'] = best_acc
            torch.save(record,os.path.join(args.output_dir,'log2.pt'))
            print('---- new best_acc = {}'.format(best_acc))
            with open(os.path.join(args.output_dir,'checkpoint.{}.th'.format(model_type.replace('/', '.'))), 'wb') as f:
                state_dict = model.module.state_dict() if args.fp16 else model.state_dict()
                torch.save(state_dict, f)
    cur_i = 0
    total_loss = 0