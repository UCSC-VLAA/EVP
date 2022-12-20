# Unleashing the Power of Visual Prompting At the Pixel Level

This is the official implementation of the paper 'Unleashing the Power of Visual Prompting At the Pixel Level'.

![](https://github.com/jywu511/Unleashing-the-Power-of-Visual-Prompting-At-the-Pixel-Level/blob/main/methods.png)

## Installation

Clone this repo:

```bash
git clone https://github.com/UCSC-VLAA/EVP
cd EVP
```

Our code is built on:

torch>=1.10.1
torchvision>=0.11.2


Then install dependencies by:

```bash
pip install -r requirments.txt

pip install git+https://github.com/openai/CLIP.git
```

## Data Preparation

See [DATASET.md](https://github.com/UCSC-VLAA/EVP/blob/main/DATASET.md)
for detailed instructions and tips.

## Train/Test for CLIP Model

* Train the Enhanced Visual Prompting on CIFAR100:

```bash
python main.py 
```

* Test the Enhanced Visual Prompting:

```bash
python main.py --evaluate
```

## Train/Test for non-CLIP Model

We propose a simple pre-processing step to match the pre-trained classes and the downstream classes for non-CLIP model. 

* Train the Enhanced Visual Prompting for the non-CLIP Model:

```bash
python main.py --non_CLIP
```

* Test the Enhanced Visual Prompting for the non-CLIP Model:

```bash
python main.py --non_CLIP --evaluate 
```


## Contact

Junyang Wu
- email: SJTUwjy@sjtu.edu.cn


Xianhang Li
- email: xli421@ucsc.edu


If you have any question about the code and data, please contact us directly.



