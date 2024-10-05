# BSharedRAG: Backbone Shared Retrieval-Augmented Generation for the E-commerce Domain

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](#Python)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-green.svg)](#PyTorch)
[![Transformers](https://img.shields.io/badge/Transformers-4.33.2-orange.svg)](#Transformers)

> This repository includes the source code of the paper [BSharedRAG: Backbone Shared Retrieval-Augmented Generation for the E-commerce Domain](https://arxiv.org/abs/2409.20075v1) (EMNLP 2024 Findings) by Kaisi Guan et al.

## Abstract

 Retrieval Augmented Generation (RAG) system is important in domains such as e-commerce, which has many long-tail entities and frequently updated information. Most existing works adopt separate modules for retrieval and generation, which may be suboptimal since the retrieval task and the generation task cannot benefit from each other to improve performance. We propose a novel Backbone Shared RAG framework (BSharedRAG). It first uses a domain-specific corpus to continually pre-train a base model as a domain-specific backbone model and then trains two plug-and-play Low-Rank Adaptation (LoRA) modules based on the shared backbone to minimize retrieval and generation losses respectively. Experimental results indicate that our proposed BSharedRAG outperforms baseline models by 5% and 13% in Hit@3 upon two datasets in retrieval evaluation and by 23% in terms of BLEU-3 in generation evaluation. 

 ![framework](/assets/framework.jpg)

If you find our work useful, please  cite the paper:

```
@misc{guan2024bsharedragbackbonesharedretrievalaugmented,
      title={BSharedRAG: Backbone Shared Retrieval-Augmented Generation for the E-commerce Domain}, 
      author={Kaisi Guan and Qian Cao and Yuchong Sun and Xiting Wang and Ruihua Song},
      year={2024},
      eprint={2409.20075},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.20075}, 
}
```

## Setup

```
git clone https://github.com/guankaisi/BSharedRAG
cd BSharedRAG/
conda create -n bsharedrag python==3.10
pip install requirments.txt -r
```

## Preparation

### Prepare database

You can download our E-commerce knowledge database from this link.

### Prepare model

You can download our E-commerce domain LLM, retrieval lora and generation lora from this link.

## Workflow

### index database

```bash
CUDA_VISIBLE_DEVICES=0 python indexer.py \
    --data_dir ./database \
    --index_save_dir ./index \
    --peft_model_path emb-peft \
    --batch_size 16 \
    --use_content_type title \
    --language zh \
    --index_name bsharedrag.index
```

### inference

```bash
python pipeline.py
```

### Evaluation

We build an e-commerce Domain Evaluation for Large Language Models. You can evaluate our BSharedRAG or Ecommerce LLMs in [EcomEval](https://github.com/guankaisi/EcomEval).

## Contact

For any question, please feel free to reach me at guankaisi[at]ruc.edu.cn
