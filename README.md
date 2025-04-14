# Transformer-based-Tagging-and-Training-Framework-

This project implements a one-layer, single-headed Transformer model using Python for some simple natural-language tasks. The model is tested and trained on synthetic tasks such as generating text for identifying part-of-speech tag. This project is adapted from CS440 Artificial Intelligence at University of Illinois Urbana-Champaign

## data
This folder contains the training and testing datasets for the model. 
- Task 1 files includes data, vocabulary, and a pre-trained model for testing the text generation function. These files can be loaded and updated with functions from **reader.py**. 
- Task 2 files contain the training data, vocabulary, and test data. These files can be used for training a new model.

## src
This folder contains the Python code of the Transformer. 
- **reader.py** has the functions to save and load from the **data** folder. It reads the vocabulary list and sentences array. Vocabulary list is a list of all possible words in the dictionary. Sentences are the text for training. If a sentence contains *T* words, the first *T-2* are the prompt text, and the last 2 words are the target output that the transformer will generate.
- **transformer.py** contains the functions to generate and train the Transformer. In order to generate the new words in the sentences, it uses a query based on the average of the embeddings of the prompt words. Each embedding is a one-hot vector, labeling the identity of the corresponding word. It then create the keys, outputs, queries, and value vectors. To generate output, the softmax function is applied to the outputs vector to select the new word. To train the model, the partial derivatives of the loss with respect to all of the model parameters and used to updates these vectors. The train function will initiate an instance of the model and return the improvement in loss over iterations.
