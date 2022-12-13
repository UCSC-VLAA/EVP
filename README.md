# Unleashing the Power of Visual Prompting At the Pixel Level

This is the official implementation of the paper 'Unleashing the Power of Visual Prompting At the Pixel Level'.

![](https://github.com/jywu511/Unleashing-the-Power-of-Visual-Prompting-At-the-Pixel-Level/blob/main/methods.png)

## Installation

Clone this repo:

```bash
git clone https://github.com/jywu511/Unleashing-the-Power-of-Visual-Prompting-At-the-Pixel-Level.git
cd visual_prompting
```

Then Install dependencies by:

torch>=1.10.1
torchvision>=0.11.2

```bash
pip install -r requirments.txt

pip install git+https://github.com/openai/CLIP.git
```

## Data Preparation

See [Dataset Preparation.md](https://github.com/jywu511/Unleashing-the-Power-of-Visual-Prompting-At-the-Pixel-Level/blob/main/datasets/Dataset%20Preparation.md)
for detailed instructions and tips.

## Train/Test for CLIP

* Train the Enhanced Visual Prompting:

```bash
python main.py 
```

* Test the Enhanced Visual Prompting:

```bash
python main.py --evaluate
```

## Train/Test for non-CLIP Model

We propose a simple pre-processing step to match the pre-trained classes and the downstream classes. You can get the
corresponding index:

```bash
python get_index.py
```

* Train the Enhance Visual Prompting for the non-CLIP Model:

```bash
python main_non_CLIP.py --evaluate False
```

* Test the Enhance Visual Prompting for the non-CLIP Model:

```bash
python main_non_CLIP.py --evaluate True
```



