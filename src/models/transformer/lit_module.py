import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models.transformer.datasets import SentimentDataset


class TransformerLitModule(pl.LightningModule):

    def __init__(self, model, tokenizer, lr, fold_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = lr
        self.fold_id = fold_id

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_nb):
        outputs = self(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
        self.log(f'train_loss_{self.fold_id}', outputs.loss.item(), prog_bar=True)
        self.log(f'train_acc_{self.fold_id}', self.train_acc(outputs.logits, batch['label']), prog_bar=True,
                 on_step=False, on_epoch=True)
        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_nb):
        outputs = self(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
        self.log(f'val_loss_{self.fold_id}', outputs.loss.item(), prog_bar=True)
        self.log(f'val_acc_{self.fold_id}', self.valid_acc(outputs.logits, batch['label']), prog_bar=True,
                 on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)
        return {
            'optimizer': optimizer,
            'lr_scheduler': StepLR(optimizer, step_size=1, gamma=0.1),
            'monitor': f'val_loss_{self.fold_id}'
        }

    @classmethod
    def from_params_and_dataset(cls, train_df, valid_df, params, fold_id, embeddings=None):
        from configs import transformer_data
        model_class, tokenizer_class, model_name = transformer_data[params.model]
        tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=True)
        model_backbone = model_class.from_pretrained(model_name, num_labels=2, output_attentions=False,
                                                     output_hidden_states=False)

        train_loader = DataLoader(SentimentDataset(train_df, tokenizer=tokenizer, length=params.tokenizer_length),
                                  num_workers=8, batch_size=params.batch_size, shuffle=True)
        val_loader = DataLoader(SentimentDataset(valid_df, tokenizer=tokenizer, length=params.tokenizer_length),
                                num_workers=8, batch_size=params.batch_size, shuffle=False)

        model = cls(model=model_backbone, tokenizer=tokenizer, lr=params.lr, fold_id=fold_id)
        return model, train_loader, val_loader
