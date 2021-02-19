import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


class TransformerLitModule(pl.LightningModule):

    def __init__(self, model, tokenizer, lr, fold_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
