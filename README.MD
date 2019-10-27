# Neural Token Representations and Negation and Speculation Scope Detection in Biomedical and General Domain Text

Codes for paper 'Neural Token Representations and Negation and Speculation Scope Detection in Biomedical and General Domain Text' by Elena Sergeeva, Henghui Zhu, Amir Tahmasebi and Peter Szolovits in LOUHI 2019

# Environment

The script is running in Python 3.6. The requirements for the package is listed in the `requirements.txt` file.

# Data

This repo contains the scripts for [BioScope dataset](http://rgai.inf.u-szeged.hu/index.php?lang=en&page=bioscope). One can download the Biology Abstracts in the link and request the clinical report. By default, I put the xml file in folder `data/raw`.

I find there is some problems when I try to parse the document, which are related to the white space. Therefore, I provide the patch files for the xml. So one can run the following command to avoid those issue
```bash
patch data/raw/abstracts_pmid.xml -i data/raw/abstracts_pmid_modified.patch -o data/raw/abstracts_pmid_modified.xml
patch data/raw/clinical.xml -i data/raw/clinical_modified.patch -o data/raw/clinical_modified.xml
```

# Preprocessing

In the preprocessing step, we need to first convert the xml files to json to be parsed easily in the next step.
```bash
python preprocessing/bioscope2json.py
```
Then we convert the json file into a standard conll format using
```bash
python preprocessing/json2conll_gold_cue.py
```
Finally, for bidirectional LSTM with Glove, ELMo and BERT models and BERT finetuning models, one can run `conll2tfrecord_glove_gold_cue.py`, `conll2tfrecord_elmo_gold_cue.py`, `conll2tfrecord_bert_gold_cue.py`, and `conll2tfrecord_bert_ft_gold_cue.py` in the `preprocessing` folder for generating the tfrecords files. 

# Model Training and Evaluation

For bidirectional LSTM with Glove, ELMo and BERT models and BERT finetuning model, one can run `bilstm_glove_gold_cue.py`, `bilstm_elmo_gold_cue.py`, `bilstm_bert_gold_cue.py`, and `bert_finetune_gold_cue.py` with flag `--training=True` for training. For evaluation, run the same script with flag `--training=False` .