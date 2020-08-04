# Cross-Modal-BERT
Implementation of the paper: Cross-Modal BERT for Text-Audio Sentiment Analysis (MM 2020)

In this paper, we propose a Cross-Modal BERT (CM-BERT) that introduces the information of audio modality to help text modality fine-tune the pre-trained BERT model. As the core unit of the CM-BERT, the masked multimodal attention is designed to dynamically adjust the weight of words through the cross-modal interaction.

The architecture of the proposed method:

![Alt text](https://github.com/thuiar/Cross-Modal-BERT/blob/master/img/architecture%20.png)

# Usage
1、Install all required library

```
pip install -r requirements.txt
```

2、Get the pre-trained BERT model and modify the --bert_model in run_classifier.py

```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```
3、Run the experiments by:

```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```
