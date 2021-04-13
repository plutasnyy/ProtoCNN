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

model_to_litmodule = {
    'distilbert': TransformerLitModule,
    'cnn': CNNLitModule,
    'protoconv': ProtoConvLitModule
}
