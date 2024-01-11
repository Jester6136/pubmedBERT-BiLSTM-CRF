
# PubmedBERT - BiLSTM - CRF
Code for our ICTA 2023 paper "An Architecture for More Fine-grained Hidden Representation in Named Entity Recognition for Biomedical Texts". Please cite our paper if you find this repository helpful in your research:

```
@InProceedings{10.1007/978-3-031-49529-8_13,
author="Tho, Bui Duc
and Giang, Son-Ba
and Nguyen, Minh-Tien
and Nguyen, Tri-Thanh",
editor="Nghia, Phung Trung
and Thai, Vu Duc
and Thuy, Nguyen Thanh
and Son, Le Hoang
and Huynh, Van-Nam",
title="An Architecture for More Fine-Grained Hidden Representation in Named Entity Recognition for Biomedical Texts",
booktitle="Advances in Information and Communication Technology",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="114--125",
abstract="This paper introduces a model for Biomedical Named Entity Recognition (BioNER). Different from existing models that mainly rely on pre-trained models, i.e., PubMedBERT, the proposed model is empowered by using PubMedBERT as the main backbone for mapping input sequences to contextual vectors. To learn more fine-grained hidden representation and effectively adapt to the recognition downstream task, the model stacks BiLSTM and CRFs on top of PubMedBERT. Given an input sentence, the model first maps the sentence into contextual vectors by PubMedBERT. The vectors are next fed into a BiLSTM layer for learning a more fine-grained hidden representation that serves as the input for sequence labeling by using CRFs. We confirm the efficiency of the model on benchmark corpora. Experimental results on 29 diverse datasets indicate that the proposed model obtains promising results compared to good as well as state-of-the-art baselines. The ablation study also shows the behavior of the model in several aspects.",
isbn="978-3-031-49529-8"
}
```

This project implements our PubmedBERT - BiLSTM - CRF. The implementation is build upon [fairseq](https://github.com/pytorch/fairseq), and heavily inspired by [CLNER](https://github.com/Alibaba-NLP/CLNER), many thanks to the authors for making their code avaliable.

## Guide

- [Requirements](#requirements)
- [Datasets](#datasets)
- [Training](#training)
- [Config File](#Config-File)

## Requirements
The project is based on PyTorch 1.1+ and Python 3.6+. To run our code, install:

```
pip install -r requirements.txt
```

The following requirements should be satisfied:
* [transformers](https://github.com/huggingface/transformers): **3.0.0** 

## Datasets
The datasets used in our paper are available [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data).

## Training

### Training NER Models

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/pubmed_bilstm_crf.yaml
```

### Dataset config

To set the dataset manully, you can set the dataset in the `$config_file` by:

```yaml
targets: ner
ner:
  Corpus: ColumnCorpus-1
  ColumnCorpus-1: 
    data_folder: datasets/MTL-Bioinformatics-2016/AnatEM-IOB
    column_format:
      0: text
      1: ner
    tag_to_bioes: ner
  tag_dictionary: resources/taggers/your_ner_tags.pkl
```


The `tag_dictionary` is a path to the tag dictionary for the task. If the path does not exist, the code will generate a tag dictionary at the path automatically. The dataset format is: `Corpus: $CorpusClassName-$id`, where `$id` is the name of datasets (anything you like). You can train multiple datasets jointly. For example:

Please refer to [Config File](#Config-File) for more details.

## Config File

The config files are based on yaml format.

* `targets`: The target task
  * `ner`: named entity recognition
  * `upos`: part-of-speech tagging
  * `chunk`: chunking
  * `ast`: abstract extraction
  * `dependency`: dependency parsing
  * `enhancedud`: semantic dependency parsing/enhanced universal dependency parsing
* `ner`: An example for the `targets`. If `targets: ner`, then the code will read the values with the key of `ner`.
  * `Corpus`: The training corpora for the model, use `:` to split different corpora.
  * `tag_dictionary`: A path to the tag dictionary for the task. If the path does not exist, the code will generate a tag dictionary at the path automatically.
* `target_dir`: Save directory.
* `model_name`: The trained models will be save in `$target_dir/$model_name`.
* `model`: The model to train, depending on the task.
  * `FastSequenceTagger`: Sequence labeling model. The values are the parameters.
  * `SemanticDependencyParser`: Syntactic/semantic dependency parsing model. The values are the parameters.
* `embeddings`: The embeddings for the model, each key is the class name of the embedding and the values of the key are the parameters, see `flair/embeddings.py` for more details. For each embedding, use `$classname-$id` to represent the class. For example, if you want to use BERT and M-BERT for a single model, you can name: `TransformerWordEmbeddings-0`, `TransformerWordEmbeddings-1`.
* `trainer`: The trainer class.
  * `ModelFinetuner`: The trainer for fine-tuning embeddings or simply train a task model without ACE.
  * `ReinforcementTrainer`: The trainer for training ACE.
* `train`: the parameters for the `train` function in `trainer` (for example, `ReinforcementTrainer.train()`).
