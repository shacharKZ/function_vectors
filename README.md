# Reversed Attention + Function Vectors

This repo is a fork of Function Vectors [(Offical Repo)](https://github.com/ericwtodd/function_vectors) that was used for parts of the results in Reversed Attention [(Offical Repo)](https://github.com/shacharKZ/Reversed-Attention)

NOTE: the version of code that was used for Reversed Attention might be different from the most updated version of Function Vectors' repo

Follow Function Vectors instructions to run the experiments

# Function Vectors in Large Language Models
### [Project Website](https://functions.baulab.info) | [Arxiv Preprint](https://arxiv.org/abs/2310.15213) | [OpenReview](https://openreview.net/forum?id=AwyxtyMwaG)

This repository contains data and code for the paper: [Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213).

<p align="left">
<img src="https://functions.baulab.info/images/Paper/fv-demonstrations.png" style="width:100%;"/>
</p> 

## Setup

We recommend using conda as a package manager. 
The environment used for this project can be found in the `fv_environment.yml` file.
To install, you can run: 
```
conda env create -f fv_environment.yml
conda activate fv
```

## Demo Notebook
Checkout `notebooks/fv_demo.ipynb` for a jupyter notebook with a demo of how to create a function vector and use it in different contexts.

## Data
The datasets used in our project can be found in the `dataset_files` folder.

## Code
Our main evaluation scripts are contained in the `src` directory with sample script wrappers in `src/eval_scripts`.

Other main code is split into various util files:
- `eval_utils.py` contains code for evaluating function vectors in a variety of contexts
- `extract_utils.py`  contains functions for extracting function vectors and other relevant model activations.
- `intervention_utils.py` contains main functionality for intervening with function vectors during inference
- `model_utils.py` contains helpful functions for loading models & tokenizers from huggingface
- `prompt_utils.py` contains data loading and prompt creation functionality

## Citing
We thank the authors of the Function Vectors paper. Please cite them:


This work appeared at ICLR 2024. The paper can be cited as follows:

```bibtex
@inproceedings{todd2024function,
    title={Function Vectors in Large Language Models}, 
    author={Eric Todd and Millicent L. Li and Arnab Sen Sharma and Aaron Mueller and Byron C. Wallace and David Bau},
    booktitle={Proceedings of the 2024 International Conference on Learning Representations},
    year={2024},
}
