import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


class TransformerLitModule(pl.LightningModule):

    def __init__(self, model, tokenizer, lr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = lr

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_nb):
        outputs = self(batch['input_ids'], token_type_ids=None, attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        self.log('train_loss', outputs.loss.item())
        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_nb):
        outputs = self(
            batch['input_ids'],
            token_type_ids=None,
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('val_loss', loss.item())

        logits = outputs.logits.detach().cpu().numpy()
        y_pred = np.argmax(logits, axis=-1).astype(int)

        y_true = batch['labels'].to('cpu').numpy().astype(int)
        no_pad_id = batch['attention_mask'].to('cpu').numpy().astype('bool')

        f1_avg = list()
        for i in range(len(y_true)):
            y_pred_no_pad = y_pred[i][no_pad_id[i]]
            y_true_no_pad = y_true[i][no_pad_id[i]]
            f1 = f1_score(y_true_no_pad, y_pred_no_pad)
            f1_avg.append(f1)

        self.log('f1', np.mean(np.array(f1_avg)))

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': StepLR(optimizer, step_size=1, gamma=0.1),
            # 'monitor': 'val_loss'
        }
