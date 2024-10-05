# NSE
- Code for [``Neuron-Level Sequential Editing for Large Language Models``]

![alt text](resource/intro_fig.png)

- Neuron-level Sequential Editing (NSE), a new model editing method designed for sequential model editing in large language models. NSE prevents model failure by optimizing the hidden states of the target layer using the model's original weights. To mitigate model forgetting, it iteratively selects neurons in multiple layers based on their activation values. Empirical experiments show that NSE significantly outperforms existing parameter-modifying model editing methods in the context of sequential editing.
- Our work focuses on optimizing sequential model editing from two perspectives: the computation of values and the updating of weights \(W\). Additionally, we recommend that readers interested in sequential model editing consult our complementary study, [AlphaEdit](https://arxiv.org/pdf/2410.02355), which enhances sequential editing from an objective standpoint. Together, these approaches offer synergistic improvements to the field.

![alt text](resource/model_fig.png)
*Figure: This is the overall architecture of our NSE method.*

## Requirements
**At least one A40 48G GPU.**

- pytorch==1.12.1
- einops==0.4.0
- higher==0.2.1
- hydra-core==1.2.0
- transformers==4.23.1
- datasets==1.18.3
- matplotlib==3.6.1
- spacy==3.4.1
- scipy==1.9.2
- scikit-learn==1.0.2
- nltk==3.7

## Quick Start
### An example for editing Llama3 (8B) on counterfact dataset using NSE
#### 1. Edit Llama3 (8B) model 
 
    python3 -m experiments.evaluate     --alg_name=NSE     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B.json --ds_name=mcf --dataset_size_limit=2000    --num_edits=100 --downstream_eval_steps=5

This command runs an evaluation script for the NSE algorithm using the Llama3-8b-instruct. Below are the explanations for each argument:

- `--alg_name=NSE`: Specifies the name of the algorithm being used, which is NSE in this case.
- `--model_name=meta-llama/Meta-Llama-3-8B-Instruct`: Indicates the name of the model being evaluated, here it is Llama-3-8B-Instruct.
- `--hparams_fname=Llama3-8B.json`: Points to the JSON file containing hyperparameters specific to the Llama-3-8B-Instruct model.
- `--ds_name=mcf`: Specifies the dataset name, in this case, "mcf".
- `--dataset_size_limit=2000`: Sets the total number of editing samples to 2000.
- `--num_edits=100`: Defines the batch size for each round of editing, meaning 100 edits will be performed in each batch. 
- `--downstream_eval_steps=5`: indicates that a test of general capabilities is conducted after every 5 rounds of editing.
#### 2. Summarize the results

    python summarize.py --dir_name=NSE --runs=run_<run1>,run_<run2>

## Acknowledgment
Our code is based on  [``MEMIT``](https://github.com/kmeng01/memit.git).
