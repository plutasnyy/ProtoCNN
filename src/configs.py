from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model_data = {
    'distilbert': [DistilBertForSequenceClassification, DistilBertTokenizerFast, 'distilbert-base-uncased'],
}