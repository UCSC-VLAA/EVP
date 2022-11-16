# Unleashing-the-Power-of-Visual-Prompting-At-the-Pixel-Level

This is the official implementation of the paper 'Unleashing the Power of Visual Prompting At the Pixel Level'.


## Installation

Clone this repo:
```bash
git clone https://github.com/jywu511/Unleashing-the-Power-of-Visual-Prompting-At-the-Pixel-Level.git
cd visual_prompting
```

Then Install dependencies by:
```bash
pip install -r requirements.txt

```


## Data Preparation

See [data_preparation.md](https://github.com/jywu511/Unleashing-the-Power-of-Visual-Prompting-At-the-Pixel-Level/blob/main/Dataset%20Preparation.md) for detailed instructions and tips.


## Train/Test for CLIP

* Train the Enhanced Visual Prompting:
```bash
python main.py --evaluate False
```

* Test the Enhanced Visual Prompting:
```bash
python main.py --evaluate True
```


