from models.sentiment_clf import SentimentClassifier, SentimentClassifier1
from models.language_model import LanguageModel


# Additional ideas
# LM
# awd lstm
# trigger word detection
# NMT
# similarity (siamese)

models = {
    "rnnsentimentclf": SentimentClassifier(3000, 100, 'RNN'),
    "grusentimentclf": SentimentClassifier(3000, 100, 'GRU'),
    "lstmsentimentclf": SentimentClassifier(3000, 100, 'LSTM'),
    "rnnlanguagemodel": LanguageModel(30, 100, "RNN"),
    "grulanguagemodel": LanguageModel(30, 100, "GRU"),
    "lstmlanguagemodel" : LanguageModel(30,100,"LSTM"),
    'test': SentimentClassifier1(3000, 100),
}

