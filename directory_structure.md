## Directory structure

``` shell


└── bert_pretraining
    ├── Dockerfile
    ├── LICENSE
    ├── PRETRAINING
    │   ├── DATASET
    │   │   └── BATCHED
                ├── ED4RE_MSL512_ASL50_S3592675_clirebert_clirevocab_uncased_tes
    │   ├── Images
    │   │   ├── Figure_1.png
    │   │   ├── checkpoint-123000_stats_CliReRoBERTa.png
    │   │   ...
    │   ├── LOCAL_MODELS
    │   │   ├── CliReBERT
    │   │   │   ├── config.json
    │   │   │   └── tokenizer.json
    │   │   └── CliReRoBERTa
    │   │       ├── config.json
    │   │       ├── merges.txt
    │   │       └── vocab.json
    │   ├── MODELS
            ├──       allenai__scibert_scivocab_uncased_ED4RE_MSL512_ASL50_S3592675_24
    │                   ├── checkpoint-1000
    │                   ├── checkpoint-10000

    │   ├── csv2dataset.py
    │   ├── eval.txt
    │   ├── jsonvocab2text.py
    │   ├── left2process.txt
    │   ├── left2process_remaining.txt
    │   ├── model_chkpt2bin.py
    │   ├── model_mlm_testing.py
    │   ├── model_training.py
    │   ├── model_training_fromscratch.py
    │   ├── model_training_fromscratch_roberta.py
    │   ├── model_training_roberta.py
    │   ├── plot_metrics.py
    │   ├── process_log.txt
    │   ├── process_log_remaining.txt
    │   ├── process_log_remaining_50.txt
    │   ├── savetokenizer_vocab.py
    │   ├── sentence_tokenizer.py
    │   ├── sentences2batches.py
    │   ├── tokenization_remainder.py
    │   ├── training_data.py
    │   ├── training_loss.py
    │   ├── training_notes.md

    ├── README.md
    ├── conda_init.md
    ├── docker-compose.yml
    ├── docker_command.md
    └── requirements.txt


```