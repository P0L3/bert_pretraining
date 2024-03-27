# Docker initialization
1. Create Docker image from the folder containing [./Dockerfile](Dockerfile)
``` shell
docker build -t bert_pretraining:1.0 . 
```

2. Run docker compose in the folder where [./docker-compose.yml](docker-compose.yml) is and open it in VS code:
``` shell
docker compose up
```
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
- 
