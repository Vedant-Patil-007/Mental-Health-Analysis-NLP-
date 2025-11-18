---
library_name: transformers
license: apache-2.0
base_model: distilbert-base-uncased
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: Mental-Health-Analysis
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Mental-Health-Analysis

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the None dataset.
It achieves the following results on the evaluation set:
- Accuracy: 0.9950
- F1: 0.9950
- Loss: 0.0299
- Precision: 0.9950
- Recall: 0.9950

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 3

### Training results

| Training Loss | Epoch  | Step | Accuracy | F1     | Validation Loss | Precision | Recall |
|:-------------:|:------:|:----:|:--------:|:------:|:---------------:|:---------:|:------:|
| 0.2938        | 0.2006 | 500  | 0.9574   | 0.9581 | 0.1866          | 0.9612    | 0.9574 |
| 0.119         | 0.4013 | 1000 | 0.9768   | 0.9769 | 0.1090          | 0.9773    | 0.9768 |
| 0.0992        | 0.6019 | 1500 | 0.9819   | 0.9820 | 0.0926          | 0.9822    | 0.9819 |
| 0.0039        | 0.8026 | 2000 | 0.9876   | 0.9876 | 0.0731          | 0.9878    | 0.9876 |
| 0.1583        | 1.0032 | 2500 | 0.9911   | 0.9911 | 0.0475          | 0.9911    | 0.9911 |
| 0.0021        | 1.2039 | 3000 | 0.9915   | 0.9915 | 0.0448          | 0.9915    | 0.9915 |
| 0.033         | 1.4045 | 3500 | 0.9916   | 0.9916 | 0.0535          | 0.9916    | 0.9916 |
| 0.0332        | 1.6051 | 4000 | 0.9915   | 0.9915 | 0.0428          | 0.9915    | 0.9915 |
| 0.0008        | 1.8058 | 4500 | 0.9936   | 0.9936 | 0.0360          | 0.9936    | 0.9936 |
| 0.0263        | 2.0064 | 5000 | 0.9941   | 0.9941 | 0.0301          | 0.9941    | 0.9941 |
| 0.0214        | 2.2071 | 5500 | 0.9939   | 0.9939 | 0.0345          | 0.9939    | 0.9939 |
| 0.0006        | 2.4077 | 6000 | 0.9945   | 0.9945 | 0.0302          | 0.9945    | 0.9945 |
| 0.0114        | 2.6083 | 6500 | 0.9952   | 0.9952 | 0.0281          | 0.9952    | 0.9952 |
| 0.0418        | 2.8090 | 7000 | 0.9950   | 0.9950 | 0.0299          | 0.9950    | 0.9950 |


### Framework versions

- Transformers 4.56.2
- Pytorch 2.8.0+cu126
- Datasets 4.0.0
- Tokenizers 0.22.1
