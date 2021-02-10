from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model_data = {
    'distilbert': [DistilBertForSequenceClassification, DistilBertTokenizerFast, 'distilbert-base-uncased'],
}

dataset_tokens_length = {
    'amazon': 256,
    'hotel': 256,
    'imdb': 512,
    'yelp': 512,
    'rottentomatoes': 128,
}
