#### BERT
``` shell
BATCH:                       256
SEQUENCE_LENGTH:             512
STEPS:                 1,000,000
EPOCHS:                       40
DATA:             3.30B(~16.0GB)    
HARDWARE:           16 TPU chips
TIME:                     4 Days
```

#### SciBERT
``` shell
BATCH:                         ?
SEQUENCE_LENGTH:         512/128
STEPS:                1,000,000?
EPOCHS:                        ?
DATA:             3.17B(~16.0GB)    
HARDWARE:        TPU v3 (8 core)
TIME:                 5 + 2 Days
```

#### JuriBERT
``` shell
BATCH:                         4
SEQUENCE_LENGTH:             512
STEPS:                 1,000,000 
EPOCHS:                        ?
DATA:              1.40B(~6.3GB)    
HARDWARE:  Nvidia GTX 1080Ti GPU
TIME:                          ?
```

#### allenai__scibert_scivocab_uncased_ED4RE_MSL512_ASL50_S3592675_24
``` shell
BATCH:                        24
SEQUENCE_LENGTH:             512
STEPS:                  ~142,208 
EPOCHS:                        8
DATA:              1.25B(~5.8GB)    
HARDWARE: Nvidia Quadro RTX 6000
TIME:                   ~12 Days
```