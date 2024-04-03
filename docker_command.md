## Run docker container
``` shell
docker run -d \
  --name bert_pretraining \
  --restart always \
  -v /home/andrijapoleksic/BERT_PRETRAINING/bert_pretraining/PRETRAINING:/bert_pretraining/PRETRAINING \
  -v /srv/andrijapoleksic/DATASET/BATCHED:/bert_pretraining/PRETRAINING/DATASET/BATCHED \
  -v /srv/andrijapoleksic/MODEL:/bert_pretraining/PRETRAINING/MODELS \
  --memory=64g \
  --cpus="16" \
  --gpus "count=1,capabilities=compute" \
  bert_pretraining:1.1 \
  tail -f /dev/null
```
## Attach a terminal to it
``` shell
docker exec -it bert_pretraining /bin/bash
```