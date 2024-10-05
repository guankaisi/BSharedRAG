import sys
import os
import json
import time
import faiss
from tqdm import tqdm
from transformers import AutoModel,AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Dict
from transformers import PretrainedConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import time
from vllm import ModelRegistry
from baichuan_embedding import BaichuanForCausalLM
import gc

# Register the model BaichuanForEmbedding to make it available for vllm
def model_regist(name="BaichuanForCausalLM"):
    ModelRegistry.register_model(name, BaichuanForCausalLM)

def model_emb_mode():
    global original_is_embedding_model 
    original_is_embedding_model = ModelRegistry.is_embedding_model
    # Define a new method that returns True no matter what is passed in
    def always_true_is_embedding_model(model_arch: str) -> bool:
        return True
    # Every model is an embedding model
    ModelRegistry.is_embedding_model = always_true_is_embedding_model
def model_gen_mode():
    ModelRegistry.is_embedding_model = original_is_embedding_model


answer_generation_template_zh = '''下面是参考信息和你需要回答的问题，请你根据参考信息中的内容，用中文回答以下问题。请你利用参考信息详细回答
[参考信息]:
{context} 
[问题]:
{question}
[回答]：'''

class Common_RAG:
    def __init__(self,args):
        self.topk = args.topk
        self.database_path = args.database_path
        self.device = torch.device("cuda")
        self.all_doc = self.get_doc_from_folder()
        self.index = faiss.read_index(args.dense_index)
    def get_doc_from_folder(self):
        """
        get all docs & properties from a folder containing json file
        """
        all_data = []
        with open(self.database_path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip('\n')
            doc = json.loads(line)
            all_data.append(doc)
        return all_data
    def get_emb(self, batch, model, tokenizer):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        attention_mask = inputs['attention_mask']
        pooler_output = model(output_hidden_states=True, return_dict=True, **inputs).last_hidden_state[:, 0]
        masked_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.)
        pooler_output = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return pooler_output.cpu()
    def search(self):
        pass
    def generate(self):
        pass
    def pipeline(self):
        pass
'''
BSharedRAG , building a RAG system with one foudation model and two finetuning lora.
We can switch our model to retrieval mode or generation mode.
'''
class BSharedRAG(Common_RAG):
    def __init__(
        self, 
        args
    ):
        super().__init__(args)
        self.base_model_path = args.base_model_path
        self.retrieval_peft_path = args.emb_peft_path
        self.generation_peft_path = args.gen_peft_path
        self.set_model()
        # self.model, self.tokenizer = self.set_model()
        self.sampling_params = SamplingParams(temperature=0.7, min_tokens=30, max_tokens=4096, repetition_penalty=1.1)
    def set_model(self):
        model_regist()
        # base_model = LLM(
        #     model=self.base_model_path, 
        #     tokenizer=self.base_model_path,
        #     trust_remote_code=True,
        #     dtype="bfloat16",
        #     enable_lora=True
        # )
        # tokenizer = AutoTokenizer.from_pretrained(self.base_model_path,trust_remote_code=True)
        # return base_model, tokenizer    

    def get_emb(self, query, model):
        prompts = []
        for question in query:
            prompts.append(question)
        outputs = model.encode(
            prompts,
            use_tqdm=False,
            lora_request=LoRARequest('emb_lora',1,self.retrieval_peft_path)
        )
        embedding = torch.Tensor([output.outputs.embedding for output in outputs])
        return embedding
    def search(self, query: List[str]) -> List[List[Optional[Dict]]]:
        model_emb_mode()
        model = LLM(
            model=self.base_model_path,
            tokenizer=self.base_model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_lora=True
        )
        query_embed = self.get_emb(query, model)
        query_embed = query_embed.detach().cpu().numpy()
        D, I = self.index.search(query_embed, self.topk)  
        batch_results = []
        for idx, ids in enumerate(I):  
            references = [self.all_doc[id] for id in ids]  
            batch_results.append(references)
        del model  # 显式删除模型
        torch.cuda.empty_cache()
        gc.collect() 
        return batch_results
    def generate(self, query: List[str], context: List[List[Optional[Dict]]]):
        model_gen_mode()
        model = LLM(
            model=self.base_model_path,
            tokenizer=self.base_model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_lora=True
        )
        # self.model.llm_engine._initialize_kv_caches()
        template = answer_generation_template_zh
        prompts = []
        for q, cons in zip(query, context):
            contexts = ""
            for con in cons:
                if con is not None:
                    contexts += (con['title'] + ":" + con['contents'])
            contexts = contexts[:2000]
            question = q.strip('query:').strip('</s>')
            prompt = template.format(context=contexts, question=question)
            prompts.append(prompt)
        outputs = model.generate(
            prompts,
            self.sampling_params,
            use_tqdm=False,
            lora_request=LoRARequest('gen_lora',2,self.generation_peft_path)
        )
        del model  # 显式删除模型
        torch.cuda.empty_cache()
        gc.collect()
        outputs = [output.outputs[0].text for output in outputs]
        return outputs

    def pipeline(self, query:List[str]):
        query = ["query:" + q + "</s>" for q in query]
        start = time.time()
        contexts = self.search(query)
        # contexts = [[{"title":"title","contents":"contents"} for i in range(len(query))]]
        end = time.time()
        print("Time of Retrieval : {}".format(end - start))
        print("检索完成，参考信息为：")
        print(contexts)
        outputs = self.generate(query, contexts)
        return outputs

