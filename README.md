# NameClassification

An LSTM model that reads surnames as a series of characters and outputs a prediction as to which language the name originates from. There are 18 languages of origin total. This is heavily based on [this](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) Pytorch tutorial on RNN's. I've changed the vanilla RNN model they use with an LSTM model. 

The files **lstm.py** and **plot.py** contain the lstm model and the function that plots the confusion matrix. Just run the file **main.py** to train the model and output the plots of loss over time and the confusion matrix.
