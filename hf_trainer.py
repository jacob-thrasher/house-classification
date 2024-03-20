import os
import torch
import transformers
import numpy as np
from torch import nn
from datasets import load_metric
from transformers import AudoModelForImageClassification, TrainingArguments, Trainer
from data import ErieParcels


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    metric = load_metric('accuracy')
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

root = '/home/jacob/Documents/data/erie_data'
model_path = "microsoft/swinv2-tiny-patch4-window8-256"
model_name = 'swinv2'

train_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erietrain.csv'), year_regression=False, model_path=model_path)
val_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), year_regression=False, model_path=model_path)



model = transformers.Swinv2ForImageClassification.from_pretrained(model_path)
model.classifier = nn.Linear(768, 2)

image_processor = transformers.AutoImageProcessor.from_pretrained(model_path)
  
args = TrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)


train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


