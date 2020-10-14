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

You can download the pre-trained BERT model from [pre-trained BERT model](https://drive.google.com/file/d/1dKSzsgXORN7WVaJJYvNzqFPCQbn-aJcb/view?usp=sharing), or you can use the following code to get it.
```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```
3、Run the experiments by:

```
python run_classifier.py
```
# Results
Experimental results on CMU-MOSI dataset.

Model  | Modality  | Acc7 | Acc2 | F1 | MAE | Corr
---- | ----- | ------  | ----- | ------ | ----- | ------ 
EF-LSTM  | T+A+V | 33.7 | 75.3 | 75.2 | 1.023 | 0.608
LMF  | T+A+V | 32.8 | 76.4 | 75.7 | 0.912 | 0.668
MFN  | T+A+V | 34.1 | 77.4 | 77.3 | 0.965 | 0.632
MARN  | T+A+V | 34.7 | 77.1 | 77.0 | 0.968 | 0.625
RMFN  | T+A+V | 38.3 | 78.4 | 78.0 | 0.922 | 0.681
MFM  | T+A+V | 36.2 | 78.1 | 78.1 | 0.951 | 0.662
MCTN  | T+A+V | 35.6 | 79.3 | 79.1 | 0.909 | 0.676
MulT  | T+A+V | 40.0 | 83.0 | 82.8 | 0.871 | 0.698
T-BERT  | T+A+V | 41.5 | 83.2 | 82.3 | 0.784 | 0.774
CM-BERT(ours)  | T+A | 44.9 | 84.5 | 84.5 | 0.729 | 0.791

# Citation
If you mentioned the method in your research, please cite this article:
```
@inproceedings{10.1145/3394171.3413690,
author = {Yang, Kaicheng and Xu, Hua and Gao, Kai},
title = {CM-BERT: Cross-Modal BERT for Text-Audio Sentiment Analysis},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3413690},
doi = {10.1145/3394171.3413690},
abstract = {Multimodal sentiment analysis is an emerging research field that aims to enable machines to recognize, interpret, and express emotion. Through the cross-modal interaction, we can get more comprehensive emotional characteristics of the speaker. Bidirectional Encoder Representations from Transformers (BERT) is an efficient pre-trained language representation model. Fine-tuning it has obtained new state-of-the-art results on eleven natural language processing tasks like question answering and natural language inference. However, most previous works fine-tune BERT only base on text data, how to learn a better representation by introducing the multimodal information is still worth exploring. In this paper, we propose the Cross-Modal BERT (CM-BERT), which relies on the interaction of text and audio modality to fine-tune the pre-trained BERT model. As the core unit of the CM-BERT, masked multimodal attention is designed to dynamically adjust the weight of words by combining the information of text and audio modality. We evaluate our method on the public multimodal sentiment analysis datasets CMU-MOSI and CMU-MOSEI. The experiment results show that it has significantly improved the performance on all the metrics over previous baselines and text-only finetuning of BERT. Besides, we visualize the masked multimodal attention and proves that it can reasonably adjust the weight of words by introducing audio modality information.},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {521–528},
numpages = {8},
keywords = {pretrained model, multimodal sentiment analysis, attention network},
location = {Seattle, WA, USA},
series = {MM '20}
}
```
