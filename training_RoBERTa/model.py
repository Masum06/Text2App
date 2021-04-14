# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import os 
import subprocess
import pickle
import pandas as pd

import sys
import training_RoBERTa.bleu
import pickle
import torch
import json
import random
import logging
# import argparse
import numpy as np
import pandas as pd
from io import open
from itertools import cycle
import torch.nn as nn
# from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder,config,decoder_tokenizer,beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, decoder_tokenizer.vocab_size, bias=False) #config.vocab_size
        self.lsm = nn.LogSoftmax(dim=-1)
        # self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,args=None):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        if target_ids is not None:  
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous() ## MH: Problmatic
            out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss*active_loss.sum(), active_loss.sum()
            # print("Inside forward, outputs:", outputs)
            return outputs
        else:
            #Predict 
            preds=[]       
            zero=torch.cuda.LongTensor(1).fill_(0)     
            for i in range(source_ids.shape[0]):
                context=encoder_output[:,i:i+1]
                context_mask=source_mask[i:i+1,:]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState()
                context=context.repeat(1, self.beam_size,1)
                context_mask=context_mask.repeat(self.beam_size,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)
            # print("Inside Forward, preds: ", preds)
            return preds   
        
        

class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        
##########################################3

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""




class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]

    ## open DF, convert to json, iterate one by one
    # with open(filename,encoding="utf-8") as f:
    #     for idx, line in enumerate(f):
    #         line=line.strip()
    #         js=json.loads(line)
    #         if 'idx' not in js:
    #             js['idx']=idx
    #         sar=' '.join(js['SAR']).replace('\n',' ')
    #         sar=' '.join(code.strip().split())
    #         nl=' '.join(js['NL']).replace('\n','')
    #         nl=' '.join(nl.strip().split())  

    df = pd.read_csv(filename)
    data_list = list(df.T.to_dict().values())
    for idx, data in enumerate(data_list):
      nl = data['NL']
      sar = data['SAR']
      examples.append(
          Example(
                  idx = idx,
                  source=nl,
                  target=sar,
                  ) 
      )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        


def convert_examples_to_features(examples, tokenizer, args,stage=None): # MH: make it encoder_tokenizer, decoder_tokenizer
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length

        #target
        if stage=="test":
            target_tokens = ['None'] #tokenizer.tokenize("None")
        else:
            # target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
            target_tokens = example.target.split()[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        # target_ids = tokenizer.convert_tokens_to_ids(target_tokens) # MH: decoder
        target_ids = decoder_tokenizer.convert_string_to_ids(' '.join(target_tokens))
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   

        if example_index < 5:
            if stage=='train':
                print("*** Example ***")
                print("idx: {}".format(example.idx))

                print("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                print("source_ids: {}".format(' '.join(map(str, source_ids))))
                print("source_mask: {}".format(' '.join(map(str, source_mask))))
                
                print("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                print("target_ids: {}".format(' '.join(map(str, target_ids))))
                print("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def main(args):

    print(args)

    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
## model was here

    if args.do_train:
        print("Inside TRAIN")
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train') # MH: 2 tokenizers
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
        train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total*0.1),
                                                    num_training_steps=t_total)
    
        #Start training
        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples))
        print("  Batch size = %d", args.train_batch_size)
        print("  Num epoch = %d", args.num_train_epochs)
        

        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,target_ids,target_mask = batch
                loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval:
                print("Inside EVAL")
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)      
                    eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)   
                    dev_dataset['dev_loss']=eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                print("\n***** Running evaluation *****")
                print("  Num examples = %d", len(eval_examples))
                print("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,target_ids,target_mask = batch                  

                    with torch.no_grad():
                        _,loss,num = model(source_ids=source_ids,source_mask=source_mask,
                                           target_ids=target_ids,target_mask=target_mask)     
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                #Pring loss of dev dataset    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    print("  %s = %s", key, str(result[key]))
                print("  "+"*"*20)   

                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)                    
                if eval_loss<best_loss:
                    print("  Best ppl:%s",round(np.exp(eval_loss),5))
                    print("  "+"*"*20)
                    best_loss=eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)  


                #Calculate bleu  
                print("Calculating BLEU")
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
                    eval_data = TensorDataset(all_source_ids,all_source_mask)   
                    dev_dataset['dev_bleu']=eval_examples,eval_data



                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask= batch                  
                    with torch.no_grad():
                        preds = model(source_ids=source_ids,source_mask=source_mask)  
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            # text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            text = decoder_tokenizer.decode(t) # MH: decoder
                            p.append(text)
                model.train()
                predictions=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                print("  %s = %s "%("bleu-4",str(dev_bleu)))
                print("  "+"*"*20)    
                if dev_bleu>best_bleu:
                    print("  Best bleu:%s",dev_bleu)
                    print("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
               
    if args.do_test:
        # print("Inside TEST")
        # files=[]
        # if args.dev_filename is not None:
        #     files.append(args.dev_filename)
        # if args.test_filename is not None:
        #     files.append(args.test_filename)
        # for idx,file in enumerate(files):   
        idx = 0
        file = args.test_filename # Change it to test file later
        print("Test file: {}".format(file))
        eval_examples = read_examples(file)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
        eval_data = TensorDataset(all_source_ids,all_source_mask)   

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval() 
        p=[]
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask= batch                  
            with torch.no_grad():
                preds = model(source_ids=source_ids,source_mask=source_mask) 
                for pred in preds:
                    t=pred[0].cpu().numpy()
                    t=list(t)
                    if 0 in t:
                        t=t[:t.index(0)]
                    # text = tokenizer.decode(t,clean_up_tokenization_spaces=False) # MH: decoder
                    text = decoder_tokenizer.decode(t)
                    p.append(text)
        model.train()
        predictions=[]
        with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
            for ref,gold in zip(p,eval_examples):
                predictions.append(str(gold.idx)+'\t'+ref)
                f.write(str(gold.idx)+'\t'+ref+'\n')
                f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

        (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, args.output_name+"test_{}.gold".format(idx))) 
        dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
        print("  %s = %s "%("bleu-4",str(dev_bleu)))
        print("  "+"*"*20)    


##==================================================

def single_example_to_feature(example, tokenizer, decoder_tokenizer,args): # MH: make it encoder_tokenizer, decoder_tokenizer
    features = []
    source_tokens = tokenizer.tokenize(example)[:args.max_source_length-2]
    source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
    source_mask = [1] * (len(source_tokens))
    padding_length = args.max_source_length - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    source_mask+=[0]*padding_length
 
    target_tokens = ['None']
    target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
    # target_ids = tokenizer.convert_tokens_to_ids(target_tokens) # MH: decoder
    target_ids = decoder_tokenizer.convert_string_to_ids(' '.join(target_tokens))
    target_mask = [1] *len(target_ids)
    padding_length = args.max_target_length - len(target_ids)
    target_ids+=[tokenizer.pad_token_id]*padding_length
    target_mask+=[0]*padding_length   

    features.append(
        InputFeatures(
              0,
              source_ids,
              target_ids,
              source_mask,
              target_mask,
        )
    )
    return features


def get_sar(eval_example, model, args):
  device = args.device
  tokenizer = args.encoder_tokenizer
  decoder_tokenizer = args.decoder_tokenizer
  eval_features = single_example_to_feature(eval_example, tokenizer, decoder_tokenizer, args)
  all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long).to(device)
  all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long).to(device)
  model.eval() 
  preds = model(source_ids=all_source_ids, source_mask=all_source_mask)  
  pred_list = list(preds[0][0].cpu().numpy())
  predicted_text = decoder_tokenizer.decode(pred_list[:pred_list.index(0)])
  return predicted_text


#===================================================
class MyTokenizer:
  vocab_size = 0
  vocab = []
  id_to_token = {}
  token_to_id = {}

  def __init__(self):
    infile = open('training_RoBERTa/roberta_decoder.vocab','rb')
    self.vocab = pickle.load(infile)
    infile.close()
    self.vocab.sort()
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

#=========================================================
class Arguments:
    pass


output_dir="training_RoBERTa/" # Colab + Drive
# output_dir="model/" # Local
data_dir = 'synthesized_data/'
train_file=data_dir+'nl_sar_train.csv'
dev_file=data_dir+'nl_sar_valid.csv'
test_file=data_dir+'nl_sar_test.csv'
pretrained_model= 'roberta-base' #'microsoft/codebert-base' #'roberta-base'

args2 = Arguments()

## Required parameters
args2.model_type='roberta'
args2.model_name_or_path=pretrained_model
args2.output_dir=output_dir
args2.load_model_path=None #output_dir+"/checkpoint-best-bleu/pytorch_model.bin"
## Other parameters
args2.train_filename=train_file
args2.dev_filename=dev_file
args2.test_filename=test_file
args2.output_name = ""
args2.config_name=""
args2.tokenizer_name=""
args2.gradient_accumulation_steps=1
args2.weight_decay=0.0
args2.adam_epsilon=1e-8
args2.max_grad_norm=1.0
args2.max_steps=-1
args2.eval_steps=-1
args2.train_steps=-1
args2.warmup_steps=0
args2.local_rank=-1
args2.seed=42

args2.no_cuda=False
args2.do_lower_case=True
args2.do_train=True
args2.do_eval=True
args2.do_test=False

args2.num_train_epochs=10
args2.train_batch_size=100
args2.learning_rate=5e-5
args2.max_source_length=80
args2.max_target_length=50

args2.eval_batch_size = 1
args2.do_test=True
args2.do_train=False
args2.beam_size=1
# args2.load_model_path='roberta.bin'

# Setup CUDA, GPU & distributed training
if args2.local_rank == -1 or args2.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args2.no_cuda else "cpu")
    args2.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args2.local_rank)
    device = torch.device("cuda", args2.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args2.n_gpu = 1
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                args2.local_rank, device, args2.n_gpu, bool(args2.local_rank != -1))
args2.device = device


config_class, model_class, tokenizer_class = MODEL_CLASSES[args2.model_type]
config = config_class.from_pretrained(args2.config_name if args2.config_name else args2.model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(args2.tokenizer_name if args2.tokenizer_name else args2.model_name_or_path,do_lower_case=args2.do_lower_case)

decoder_tokenizer = MyTokenizer()

#budild model
encoder = model_class.from_pretrained(args2.model_name_or_path,config=config)    
decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,decoder_tokenizer=decoder_tokenizer,
              beam_size=args2.beam_size,max_length=args2.max_target_length,
              sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

args2.encoder_tokenizer = tokenizer
args2.decoder_tokenizer = decoder_tokenizer

model_dict = {
    'roberta':'gdown https://drive.google.com/uc?id=1S38MsO5zfJW5ijYsS7JxR5Gg9MrXUuvZ',
    'codebert':'gdown https://drive.google.com/uc?id=12__l1gWhjGOUWVAlVNJS_mTN_vTgfmTW'
}

