import re
import os
import json
import subprocess
import warnings
import pickle
import faiss
import argparse
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
import sys
import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import torch.nn as nn
from vllm import LLM, SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.lora.request import LoRARequest

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        return text

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    return input_ids, attention_masks


def load_json_data(data_dir):
    """
    Load multiple JSON files from the folder and merge.
    """

    file = os.listdir(data_dir)
    file_path = os.path.join(data_dir, file[0])
    print(file[0])
    all_data = []
    with open(file_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip('\n')
        js = json.loads(line)
        all_data.append(js)
    
    return all_data

def check_dir(dir_path):
    """
    Determine if the folder exists and if there is content.
    """
    
    if os.path.isdir(dir_path):
        if len(os.listdir(dir_path)) > 0:
            return False
    else:
        os.makedirs(dir_path)
    return True

class Index_Builder:
    def __init__(
            self, 
            data_dir, 
            index_save_dir, 
            use_content_type,
            batch_size,
            peft_model_path,
            language,
            index_name
    ):   
        self.data_dir = data_dir      
        self.index_save_dir = index_save_dir   
        self.use_content_type = use_content_type
        self.batch_size = batch_size
        self.language = language
        self.index_name = index_name
        self.peft_model_path = peft_model_path
        if not os.path.exists(index_save_dir):
            os.makedirs(index_save_dir)
            
    def build_index(self):
        self.build_dense_index()

    def build_model(self, peft_model_name):
        config = LoraConfig.from_pretrained(peft_model_name)
        config.inference_mode = True
        base_model = LLM(
            model=config.base_model_name_or_path, 
            tokenizer=config.base_model_name_or_path,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_lora=True
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,trust_remote_code=True)
        return base_model, tokenizer

    def get_embedding(self, model, tokenizer, batch):
        prompts = []
        for question in batch:
            prompts.append(question)
        outputs = model.encode(
            prompts,
            use_tqdm=False,
            lora_request=LoRARequest('emb_lora',1,self.peft_model_path)
        )
        outputs = torch.Tensor([output.outputs.embedding for output in outputs])
        outputs = nn.functional.normalize(outputs,p=2,dim=-1)
        return outputs
        

    def build_dense_index(self):
        
        dense_index_path = self.index_save_dir + "/dense"
        faiss_save_path = dense_index_path + '/' + self.index_name
        if not check_dir(dense_index_path):
            warnings.warn("Dense index already exists and will be overwritten.", UserWarning)
        
        # load json files
        print("Start loading data...")
        self.all_docs = load_json_data(self.data_dir)
        print(self.all_docs[0])
        print("Finish.")
        # convert doc to vector
        print("Start converting documents to vectors...")
        if self.use_content_type == "title":
            doc_content = ["passage:" + item['title'] + "</s>" for item in self.all_docs]
        elif self.use_content_type == "contents":
            doc_content = ["passage:" + item['contents'] + "</s>" for item in self.all_docs]
        else:
            print("Title + contents")
            doc_content = ["passage:" + item['title'] + item['contents'] + "</s>" for item in self.all_docs]
        model, tokenizer = self.build_model(self.peft_model_path)
        doc_dataset = TextDataset(doc_content, tokenizer)
        doc_loader = torch.utils.data.DataLoader(doc_dataset, batch_size=self.batch_size)
        doc_embedding = []
        for batch in tqdm(doc_loader):
            with torch.no_grad():
                output = self.get_embedding(model, tokenizer, batch)
            doc_embedding.append(output)
        doc_embedding = torch.cat(doc_embedding, dim=0)
        print("Finish converting embeddings.")

        # Build faiss index by using doc embedding
        print("Start building faiss index...")
        hidden_dim = doc_embedding.shape[1]
        dense_index = faiss.IndexFlatL2(hidden_dim)
        dense_index.add(doc_embedding.cpu().numpy())
        faiss.write_index(dense_index,faiss_save_path)
        print("Finish building index.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Creating index based on JSON documents.")

    # Basic parameters
    parser.add_argument('--data_dir', type=str, required=True, 
                        help="The folder path for Json files.")
    parser.add_argument('--index_save_dir', type=str, required=True, 
                        help="Path to the folder where the index is stored.")
    
    # Parameters for building dense index
    parser.add_argument('--use_content_type', type=str, default='title', choices=['title','contents','all'],
                        help="The part of the document used to build an index.")
    parser.add_argument('--index_name',type=str, default='bge_title.index')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help="Batch size used when constructing vector representations of documents")
    parser.add_argument('--peft_model_path', type=str, default=None)
    parser.add_argument('--language', type=str, default="zh", choices=['zh','en'],
                        help="Language of the document.")
                        
    args = parser.parse_args()

    index_builder = Index_Builder(data_dir = args.data_dir,
                                  index_save_dir = args.index_save_dir,
                                  use_content_type = args.use_content_type,
                                  batch_size = args.batch_size,
                                  peft_model_path = args.peft_model_path,
                                  language = args.language,
                                  index_name = args.index_name)
    index_builder.build_index()