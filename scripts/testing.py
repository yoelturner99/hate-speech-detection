# -*- coding: utf-8 -*-
import time
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import CamembertTokenizer, CamembertForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help='Path to input data')
parser.add_argument("--model_dir", type=str, help='Model directory')
parser.add_argument("--batch_size", type=int, default=32, help='Batch size')
parser.add_argument("--num_labels", type=int, default=2, help='Number of labels')
parser.add_argument("--max_len", type=float, default=128, help='Maximum length of token sequence')
parser.add_argument("--device_id", type=int, default=0, help='Device ID')


def test(model, test_dataloader, device, desc) -> tuple:
    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions , true_labels = [], []

    # Predict
    for batch in tqdm(test_dataloader,desc=desc):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
      
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
      
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(
                b_input_ids, 
                token_type_ids=None, 
                attention_mask=b_input_mask
            )

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
   
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
   
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate the MCC
    return flat_predictions, flat_true_labels


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

    # Initializing tokenizer
    tokenizer = CamembertTokenizer.from_pretrained(args.model_dir)
    # Load Model
    model = CamembertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.model_dir, # Use the 12-layer CamemBERT
        num_labels=args.num_labels,                   # Binary classification.
        output_attentions=False,                      # Whether the model returns attentions weights.
        output_hidden_states=False,                   # Whether the model returns all hidden-states.
    )
    model.to(device)

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
    
    # Create the DataLoader for our test set.
    test_data = TensorDataset(input_ids, attention_masks, labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

     # Testing Model
    tic = time.time()
    pred, lab = test(
        model=model,
        test_dataloader=test_dataloader,
        device=device,
        desc=f'[{str(device).upper()}] Testing Model'
    )
    tt = (time.time() - tic)/60 
    print(f'Inference time for {len(lab)} sentences: {tt:.2f} mins')
    
    cm = confusion_matrix(lab, pred)
    f1 = f1_score(lab, pred, average='weighted')
    recall = recall_score(lab, pred, average='weighted')
    precision = precision_score(lab, pred, average='weighted')
    accuracy = sum([x == y for x, y in zip(list(pred), lab)])/len(lab)

    # Print results
    print(f"F1 Score  : {100*f1:.2f} %")
    print(f"Recall    : {100*recall:.2f} %")
    print(f"Precision : {100*precision:.2f} %")
    print(f'Accuracy  : {100*accuracy:.2f} %')
    print(f"\nConfusion Matrix :")

    cm_df = pd.DataFrame(cm, index=['Non-hateful', 'Hateful'], columns=['Non-hateful', 'Hateful'])
    cm_df
