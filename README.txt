This repository contains the IMS System submission for the WASSA-2017 Shared Task on Emotion Intensity (EmoInt)
Task: Given a tweet and an emotion x {anger,fear,joy,sadness}, determine the intensity or degree of emotion X felt by the speaker -- a real-valued score between 0 and 1.

Original Paper link: http://www.romanklinger.de/publications/emo_ims_kkk.pdf

Pre-requisite:
Paste the twitter_sgns_subset.txt.gz file in "<environmen_path>\Lib\site-packages\gensim\test\test_data"


To train the CNN-LSTM model: python run_CNN_LSTM.py

The scripts trains one model per emotion for the given test file
By default we rely on the official training data for training
Note that we provide here only a subset of our vectors
twitter_sgns_subset.txt.gz covers the shared task vocabulary
vectors are in word2vec format (can be gz, txt or binary)
The output of the regression is a single file per training emotion

