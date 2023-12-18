## Cats generation

### Installation guide
To install all the dependencies, run
```shell
pip install -r ./requirements.txt
```

To train a model locally (cuda is required), download [this dataset](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) and unarchive it into "data/cats" folder. Then you may run
```shell
python3 train.py
```

Also you may use main.ipynb and upload it to kaggle. In this case, you may use it without any extra steps (but you should add the provided dataset as a kaggle input)

### Credits
This repository is based on an [official pytorch tutorial for DCGAN training](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
