<h1 align="center">Visual-PCQA</h1>
<p align="center">
Hai M
</p>
## ⚙️ Installation
All experiments are conducted on Ubuntu 22.04 and CUDA 12.9.

```
conda create --name visual-pcqa python=3.9
conda activate visual-pcqa
pip install -r requirements.txt
```


## 📦 Data Preparation

We provide the download link for the Waterloo Point Cloud Database (WPC) dataset: [WPC](https://github.com/qdushl/Waterloo-Point-Cloud-Database).

You can run the ```preprocess``` to generate the pseudo-reference and distorted point clouds. (This part of the code will be released upon acceptance of the paper.)

## 📈 Results

Experimental results of Visual-PCQA on four public datasets.

| Metric | SJTU-PCQA | WPC | LS-PCQA | WPC2.0 |
|:------:|:---------:|:---:|:--------:|:------:|
| PLCC ↑ | 0.9578    | 0.9063 | 0.6587 | 0.9060 |
| SRCC ↑ | 0.9339    | 0.9034 | 0.6469 | 0.9073 |
| RMSE ↓ | 0.6847    | 9.5912 | 0.5796 | 9.2092 |

## ⚠️ Note

Currently, Visual-PCQA only provides the testing stage code. The complete training and testing code will be released upon acceptance of the paper. In addition, we provide the ```model_pth``` on the first train-test split of the WPC dataset to demonstrate the effectiveness of the proposed method.




