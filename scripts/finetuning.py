# -*- coding: utf-8 -*-
import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import  get_linear_schedule_with_warmup
from transformers import CamembertTokenizer, CamembertForSequenceClassification


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help='Path to input data')
parser.add_argument("--out_dir", type=str, default='./models', help='Output directory')
parser.add_argument("--log_dir", type=str, default='./runs', help='Tensorboard log directory')
parser.add_argument("--model_name", type=str, default='camembert_mad_v0', help='Model Name')
parser.add_argument("--train_ratio", type=float, default=0.8, help='Size of train split')
parser.add_argument("--batch_size", type=int, default=32, help='Batch size')
parser.add_argument("--num_labels", type=int, default=2, help='Number of labels')
parser.add_argument("--max_len", type=float, default=128, help='Maximum length of token sequence')
parser.add_argument("--epochs", type=int, default=4, help='Number of epochs')
parser.add_argument("--lr", type=float, default=2e-5, help='Learning Rate')
parser.add_argument("--eps", type=float, default=1e-8, help='Epsilon')
parser.add_argument("--device_id", type=int, default=0, help='Device ID')


def train(model, train_dataloader, optimizer, scheduler, device, desc) -> float:
    model.train()
    # Tracking variables
    running_loss = 0.0
    progress_bar = tqdm(train_dataloader,desc=desc)
    for step, batch in enumerate(progress_bar):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        model.zero_grad()

        # Forward Pass
        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels
        )
        # Extract the loss from the output
        loss = outputs[0]
        # Accumulate the training loss over all of the batches
        running_loss += loss.item()
        # Backward Pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to prevent the "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    mean_train_loss = running_loss/step
    return mean_train_loss


def eval(model, val_dataloader, device, desc) -> float:
    model.eval()
    # Tracking variables
    running_accuracy = 0.0
    progress_bar = tqdm(val_dataloader,desc=desc)
    # Evaluate data for one epoch
    for step, batch in enumerate(progress_bar):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask
            )

        # Get the "logits" output by the model.
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Calculate the accuracy for this batch of test sentences.
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        val_accuracy = np.sum(pred_flat == labels_flat)/len(labels_flat)
        # Accumulate the total accuracy.
        running_accuracy += val_accuracy

    # Report the final accuracy for this validation run.
    mean_val_accuracy = running_accuracy/step
    return mean_val_accuracy


if __name__ == '__main__':
    # Parse arguments
    args = parser.parse_args()

    data = pd.read_csv(
        args.data_path,
        sep='\t',
        on_bad_lines='skip'
    )
    print(f'Size of dataset: {len(data)}')
    text_list = data['text'].values[:]
    label_list = data['label'].values[:]
    print(f'Maximum length of text: {max([len(str(t)) for t in text_list])}')

    # Check for GPU
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('We will use the GPU:', torch.cuda.get_device_name(args.device_id))
        device = torch.device(args.device_id)
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    print(f'Device ID -> {device}')

    # Initialize Tensorboard Writer
    tensorboard_dir = os.path.join(args.log_dir, args.model_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer= SummaryWriter(log_dir=tensorboard_dir)

    # Initializing tokenizer
    MODEL_NAME = 'camembert/camembert-large'
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
    # Load Model
    model = CamembertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, # Use the 12-layer CamemBERT
        num_labels=args.num_labels,               # Binary classification.
        output_attentions=False,                  # Whether the model returns attentions weights.
        output_hidden_states=False,               # Whether the model returns all hidden-states.
    )
    model.to(device)

    # Tokenize all of the texts and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    for text in text_list:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            str(text),                  # Sentence to encode.
            add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
            max_length=args.max_len,    # Set maximum length of sequence
            padding='max_length',       # Pad to max length
            truncation=True,            # Truncate to max length
            return_attention_mask=True, # Construct attn. masks.
            return_tensors='pt',        # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])  

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(label_list)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Calculate the number of samples to include in each set.
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset), # Select batches randomly
        batch_size=args.batch_size # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    val_dataloader = DataLoader(
        val_dataset, # The validation samples.
        sampler=SequentialSampler(val_dataset), # Pull out batches sequentially.
        batch_size=args.batch_size # Evaluate with this batch size.
    )

    # For the purposes of fine-tuning, the authors recommend choosing from the following values:
    # - Batch size: 16, 32  (We chose 32 when creating our DataLoaders).
    # - Learning rate (Adam): 5e-5, 3e-5, 2e-5  (We'll use 2e-5).
    # - Number of epochs: 2, 3, 4  (We'll use 4).
    # - epsilon parameter `eps = 1e-8` is "a very small number to prevent any division by zero in the implementation

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.lr)

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0, # Default value in run_glue.py
        num_training_steps = len(train_dataloader) * args.epochs
    )

    # Run Finetuning
    for epoch in range(0, args.epochs):
        # Run Training Loop
        tic = time.time()
        mean_train_loss = train(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            desc=f'[{str(device).upper()}] Train Epoch {epoch+1}/{args.epochs}'
        )

        # Run Validation Loop
        mean_val_accuracy = eval(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            desc=f'[{str(device).upper()}] Validate Epoch {epoch+1}/{args.epochs}'
        )
        tt = (time.time() - tic)/60
        # Print evolution of accuracy and loss
        tqdm.write(
            f'[Epoch {epoch+1}/{args.epochs}] ' +
            f'Train Loss: {mean_train_loss:.3E} ' +
            f'| Accuracy: {mean_val_accuracy:.3E}'+
            f'| Time taken: {tt:.2f} mins'
        )
        writer.add_scalar('Loss/train', mean_train_loss, epoch)
        writer.add_scalar('Accuracy', mean_val_accuracy, epoch)

    # Saving model
    output_dir = os.path.join(args.out_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
