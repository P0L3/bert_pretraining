<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/climate-research-domain-berts-pretraining/text-classification-on-climabench)](https://paperswithcode.com/sota/text-classification-on-climabench?p=climate-research-domain-berts-pretraining) -->
[![DOI:10.1007/s42452-025-07740-5](http://img.shields.io/badge/DOI-10.21203/rs.3.rs-6644722/v1.svg)]([https://link.springer.com/article/10.1007/s42452-025-07740-5](https://link.springer.com/article/10.1007/s42452-025-07740-5))
[![Hugging Face - BERTmosphere](https://img.shields.io/badge/HuggingFace-BERTmosphere-blue?logo=huggingface)](https://huggingface.co/collections/P0L3/bertmosphere-681db99388ca86d430f14347)

# BERTmosphere 
Check out the collection of models pretrained based on this code: [BERTmosphere](https://huggingface.co/collections/P0L3/bertmosphere-681db99388ca86d430f14347). 

Thank you [Nishan Chatterjee](https://github.com/nishan-chatterjee) for the creative collection name! 

# Docker initialization
1. Create Docker image from the folder containing [./Dockerfile](Dockerfile)
``` shell
docker build -t bert_pretraining:1.0 . 
```

2. Run docker compose in the folder where [./docker-compose.yml](docker-compose.yml) is and open it in VS code:
``` shell
docker compose up
```

- Command for server is available [here](./docker_command.md).

- Conda environment procedure is available [here](./conda_init.md).

# Workflow
[sentence_tokenizer.py](./PRETRAINING/sentence_tokenizer.py) > [tokenization_remainder.py](./PRETRAINING/tokenization_remainder.py) > [sentences2batches.py](./PRETRAINING/sentences2batches.py) > [csv2dataset.py](./PRETRAINING/csv2dataset.py) 
1. Perform sentence tokenizaton on the data containing Titles and Content from CSV filE using [sentence_tokenizer.py](./PRETRAINING/sentence_tokenizer.py) -> DURATION: 12h for 180,000 papers
``` shell
python3 sentence_tokenizer.py > process_log.txt
```
2. Use created `process_log.txt` to create `left2process.txt` ([left2process.txt](.PRETRAINING/left2process.txt)) that containes failed batches
3. Use [tokenization_remainder.py](./PRETRAINING/tokenization_remainder.py) to create new csv file based on the failed batches
4. Repeat first 3 steps until all data is tokenized into sentences
5. Use [sentences2batches.py](./PRETRAINING/sentences2batches.py) to create csv with text rows each containing ~11 sentences (tokenized ~512 tokens for MLM training) -> DURATION: 1 minute
6. Use [csv2dataset.py](./PRETRAINING/csv2dataset.py) to create final dataset (using dataset library) for training -> DURATION: 3h for 3,600,000 rows -> NOTE: Watch out for RobertaTokenizer/BertTokenizer
7. :pill: :pill: **Step 6 needs to be performed for every model with it's tokenizer!!!** :pill: :pill:
8. Start training BERT with [model_training.py](./PRETRAINING/model_training.py), RoBERTa with [model_training_roberta.py](./PRETRAINING/model_training_roberta.py), BERT from scratch with [model_training_fromscratch.py](PRETRAINING/model_training_fromscratch.py) or [model_training_fromscratch_roberta.py](PRETRAINING/model_training_fromscratch_roberta.py) -> DURATION: Depends on batch size, sequence length and hardware; 13 Days for [this setup](./PRETRAINING/training_notes.md#allenai__scibert_scivocab_uncased_ED4RE_MSL512_ASL50_S3592675_24)
> Note: Fix training paramaters and directories according to your need!
9. Final checkpoint saves in binary format (convinient for older code, but use with caution!). To save other checkpoints in desired format, use [model_chkpt2bin.py](PRETRAINING/model_chkpt2bin.py).

# Model training info
- [training_notes](./PRETRAINING/training_notes.md)

# LinkBERT
1. [linkbert_prep](./PRETRAINING/linkbert_prep.ipynb) is used for initial stats on the data and fetching citations using Semantic Scholar/Crossref

# Cite
``` latex
﻿@Article{Poleksić2025,
author={Poleksi{\'{c}}, Andrija
and Martin{\v{c}}i{\'{c}}-Ip{\v{s}}i{\'{c}}, Sanda},
title={Pretraining and evaluation of BERT models for climate research},
journal={Discover Applied Sciences},
year={2025},
month={Oct},
day={24},
volume={7},
number={11},
pages={1278},
issn={3004-9261},
doi={10.1007/s42452-025-07740-5},
url={https://doi.org/10.1007/s42452-025-07740-5}
}


```

# Data collection
- Used [PDF2TXT repo](https://github.com/P0L3/PDF2TXT)

## TODO 
- [ ] - Add trainig script similar to [evidence_synthesis](https://github.com/dspoka/ccai-nlp-tutorial-2023/blob/main/1_evidence_synthesis.ipynb) from dspoka.
- [ ] - Make a framework for training and evaluation of all available models: BERT, RoBERTa, DeBERTa, ...
- [ ] - https://huggingface.co/docs/transformers/en/main_classes/data_collator#transformers.DataCollatorForPermutationLanguageModeling

# Links
- https://www.youtube.com/watch?v=IC9FaVPKlYc&t=85s
- https://towardsdatascience.com/how-to-train-bert-aaad00533168
- https://github.com/jamescalam/transformers/blob/main/course/training/03_mlm_training.ipynb
- https://aclanthology.org/2023.bionlp-1.19.pdf
- https://github.com/stelladk/PretrainingBERT
- https://github.com/stelladk/PretrainingBERT/blob/main/pretrain.py#L6
- https://huggingface.co/docs/transformers/en/main_classes/data_collator
- https://huggingface.co/docs/transformers/en/notebooks
- https://github.com/huggingface/notebooks/blob/main/examples/language_modeling_from_scratch.ipynb
- https://github.com/huggingface/transformers/tree/main/examples
- https://huggingface.co/docs/datasets/en/loading
- https://www.geeksforgeeks.org/python-random-sample-function/
- https://www.youtube.com/watch?v=IcrN_L2w0_Y
- https://thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python
- https://huggingface.co/docs/transformers/main_classes/trainer#trainingarguments
- https://keras.io/examples/nlp/pretraining_BERT/
- https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891
- https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
