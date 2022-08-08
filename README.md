# ML-Algorithms - Fine-Tuning
## Twitter sentiment analysis is performed by using dimensionality reduction techniques in ML algorithms
- Application of ML algorithms is performed by using `pipeline` and `hyperparameter search` for fine tuning as an example.
- Detailed information can be found via course link: https://bit.ly/intro_nlp
## IMDB Movie Reviews Sentiment Analysis with `TF-IDF`
- Using pipeline with the `GridSearchCV` to fine-tune hyperparameters, `Logistic regression` and `SVM` model are used to predict the sentiment analysis of a movie review. After that, model is saved and loaded again to test it with text.
- `Pickle` is used to save and load the model.
## Sentiment Analysis with Deep Learning using ANN and CNN in Tensorflow
- Simple `Artificial Neural Network (ANN)` and `1D Convolutional Neural Network (CNN)` models are used to predict sentiment analysis for IMDB movei reviews, which is a binary classification problem.
- Spacy `en_core_web_lg` pretrained model (https://spacy.io/models/en) is used to convert texts to vectors in dataframes.
- Feature scaling is applied to X_train and X_test using MinMaxScaler before being fed inputs into the neural network.


