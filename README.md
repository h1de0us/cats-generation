## Cats generation

[Wandb report link](https://wandb.ai/h1de0us/cats-generation/reports/Cats-generation--Vmlldzo2Mjk0MTcz)

### Installation guide
To install all the dependencies, run
```shell
pip install -r ./requirements.txt
```

To train a model locally (cuda is required), download [this dataset](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) and unarchive it into "data/cats" folder. Then you may run
```shell
python3 train.py
```

Also you may use main.ipynb and upload it to kaggle. In this case, you may use it without any extra steps (but you should add the provided dataset as a kaggle input). In both cases don't forget to replace <YOUR_API_KEY> with your real API key.

To download model weights, follow [this link](https://drive.google.com/file/d/1stGE9soueOgwmCvdBkAd7Qvni3PP2tUs/view?usp=drive_link). Then put the checkpoint into the "saved" folder. 
You can use the following script to use pretrained model and generate some cats!
```shell
python3 generate.py
```

### Credits
This repository is based on an [official pytorch tutorial for DCGAN training](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
