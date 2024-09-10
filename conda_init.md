## Step-by-step conda env initialization

1. Create: 
``` shell
conda create --name bp python=3.8.0
```

2. Activate
``` shell
conda activate bp
```

3. Install pytorch 2.2.1 (should be searched for a specific CUDA and Python version)
``` shell
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. Install requirements (should comment-out the pytorch requirement in requirements.txt to prevent clashes)
``` shell
pip install -r requirements.txt 
```

5. Use