# Transformers for text analysis

The repository contains transformer models for text analysis.
The work can be organized in the following sequence (pipeline):
* Spam filter
* Toxic filter
* Sentimental analysis and topic modeling


## Spam filter

### Files:

* `spam_classifier.py` - contains class `SpamClassifier`, which represent classifier for spam,

    Weights should be downloaded manually from
[google disk](https://drive.google.com/file/d/1rpwl69WpuXcPkqaEuNNA13qI7AwbbDYU/view?usp=sharing).

    Put the downloaded files next to the file `spam_classifier.py`

* `spam_bert.ipynb` and `spam_baseline.ipynb` - contains training and validation BERT model and baseline model.

## Toxic filter

### Files:

* `toxic_classifier.py` - contains function `get_toxic_classifier()`, which loads weights and returns a classifier.
It can be used for inference.

    Weights should be downloaded manually from
[google disk](https://drive.google.com/file/d/15oZMdAwQ5U3xUnSmyNIhr5wE1bY0Nrys/view?usp=sharing).

* `training.ipynb` - contains training and validation. In the end of the notebook, the performance of the model
was compared with baseline model.

## Sentimental analysis

* `sentiment_analysis_model.py` - python script for loading model's weights and inference

* `bert-sentiment-emotions-analysis.ipynb` - notebook with BERT model's performance on solving sentiment (emotions) analysis task

    Weights should be downloaded manually from
[google disk](https://drive.google.com/file/d/1-24yxp4e5ViSONAbt90te8-JgeGykmhw/view?usp=sharing).

## Topic modeling
