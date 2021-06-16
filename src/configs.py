from enum import Enum, auto

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from models.cnn.lit_module import CNNLitModule
from models.protoconv.lit_module import ProtoConvLitModule
from models.transformer.lit_module import TransformerLitModule

transformer_data = {
    'distilbert': [DistilBertForSequenceClassification, DistilBertTokenizerFast, 'distilbert-base-uncased'],
}

dataset_tokens_length = {
    'amazon': 256,
    'hotel': 256,
    'imdb': 512,
    'yelp': 512,
    'rottentomatoes': 128,
}

dataset_to_number_of_prototypes = {
    'imdb': 32,
    'amazon': 32,
    'yelp': 32,
    'rottentomatoes': 32,
    'hotel': 32
}

dataset_to_separation_loss = {
    'imdb': 0.005,
    'amazon': 0.005,
    'yelp': 0.005,
    'rottentomatoes': 0.005,
    'hotel': 0.005
}


model_to_litmodule = {
    'distilbert': TransformerLitModule,
    'cnn': CNNLitModule,
    'protoconv': ProtoConvLitModule
}


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class Models(AutoName):
    distilbert = auto()
    cnn = auto()
    protoconv = auto()


class Datasets(AutoName):
    amazon = auto()
    hotel = auto()
    imdb = auto()
    yelp = auto()
    rottentomatoes = auto()
