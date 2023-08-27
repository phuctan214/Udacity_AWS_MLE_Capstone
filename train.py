from transformers import AutoImageProcessor, BeitForImageClassification, BeitModel
from transformers import Trainer,TrainingArguments,AutoTokenizer,default_data_collator
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import logging
import sys
import argparse
import os
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--model_id", type=str, default="microsoft/beit-base-patch16-224")
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    dataset = load_dataset(args.training_dir)
    
    # load processor
    image_processor = AutoImageProcessor.from_pretrained(args.model_id)

    # preprocess function
    def preprocess_function(examples):
        return image_processor(examples["image"],return_tensors="pt")

    # preprocess dataset
    train_dataset = dataset['train'].map(preprocess_function,batched=True)
    val_dataset = dataset['validation'].map(preprocess_function,batched=True)
    test_dataset = dataset['test'].map(preprocess_function,batched=True)
    
    train_dataset = train_dataset.remove_columns(["image"])
    val_dataset = val_dataset.remove_columns(["image"])
    test_dataset = test_dataset.remove_columns(["image"])
    
    
    logger.info("Preprocessing train, validation, test done ")

    # define labels
    num_labels = len(train_dataset.unique("label"))

    # print size
    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded val_dataset length is: {len(val_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # download model from model hub
    num_labels = len(train_dataset.unique("label"))
    model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224", ignore_mismatched_sizes=True, num_labels=num_labels)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # update the config for prediction
    label2id = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
    }
    id2label = {
        0: "1",
        1: "2",
        2: "3",
        3: "4",
        4: "5",
    }
    trainer.model.config.label2id = label2id
    trainer.model.config.id2label = id2label

    # Saves the model to s3
    trainer.save_model(args.model_dir)