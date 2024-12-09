import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import os
from transformers import TrainerCallback, ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
import evaluate
import numpy as np


class ViT_model():
    def __init__(self, train_dataset, valid_dataset,test_dataset, classes,epoch, batch_size, learning_rate, device=None):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.num_classes = len(classes)
        self.classes = classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_dataset = test_dataset
        self.epochs = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.id2label = {id: label for id, label in enumerate(self.classes)}
        self.label2id = {label: id for id, label in self.id2label.items()}

        # Load model
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id
        ).to(self.device)

        # Feature extractor
        self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        # Metric
        self.metric = evaluate.load("accuracy")

        # Training arguments
        self.training_args = TrainingArguments(
            output_dir="vit_flowers",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir='logs',
            remove_unused_columns=False,
        )

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([example[0].float() for example in examples])
        labels = torch.tensor([example[1] for example in examples], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)

    def train(self):
        class DetailedLoggingCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                print("Logs:", logs)
                if logs is not None:
                    if state.global_step % 100 == 0:
                        train_loss = logs.get('loss', 'N/A')
                        train_accuracy = logs.get('accuracy', 'N/A')
                        print(f"Step {state.global_step}:")
                        print(f"  Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

                    if 'eval_loss' in logs:
                        valid_loss = logs.get('eval_loss', 'N/A')
                        valid_accuracy = logs.get('eval_accuracy', 'N/A')
                        print(f"Epoch {state.epoch}:")
                        print(f"  Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}")
                        print("-" * 50)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            tokenizer=self.feature_extractor,
            callbacks=[DetailedLoggingCallback()]
        )

        trainer.train()
        trainer.save_model('best_emotion_model-{}-{}-{}'.format(self.epochs,self.batch_size, self.learning_rate))
        outputs = trainer.predict(self.test_dataset)
        print(outputs.metrics)
