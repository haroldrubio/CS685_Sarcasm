import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from classifier import *
import torch
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import sys

from  utils import *
from sentence_transformers import SentenceTransformer, util

from transformers import BertTokenizer, BertModel
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
import torch.nn as nn
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
main_path='/home/nazaninjafar/UMass/github_repo/CS685_Sarcasm/'
path='/home/nazaninjafar/UMass/github_repo/CS685_Sarcasm/user_tok/'
timeline_posts=pd.read_csv(path+"/timeline_posts.tsv")

timeline_posts["user"] = timeline_posts["user"].astype('category')
timeline_posts["label_coded"] = timeline_posts["user"].cat.codes
X = timeline_posts.post.values
y = timeline_posts.label_coded.values
usernums=len(timeline_posts.label_coded.unique())
train_inputs=torch.load(path+'/user_embedding_train_inputs.pt')
train_masks=torch.load(path+'/user_embedding_train_masks.pt')

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y)

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
batch_size = 16

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

learningrate=5e-5
from transformers import AdamW, get_linear_schedule_with_warmup
from classifier import *
def initialize_model(epochs=4,lr=learningrate):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
#     bert_model = BertM(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    h_dims=512 #hidden dimesion
    out_dims=usernums #number of classes
    classifier=BertClassifier(h_dims,out_dims)
    
    classifier.to(device)
    # Create the optimizer
    optimizer = AdamW(classifier.parameters(),
                      lr=learningrate,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return classifier, optimizer, scheduler

# Specify loss function
loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} |{'Train Acc':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()
            # print('b_input_ids',b_input_ids.size())
            # print('b_attn_mask',b_attn_mask.size())
            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels.long())
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy,_ = evaluate(model, val_dataloader)
            _,train_accuracy,_=evaluate(model,train_dataloader)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {train_accuracy:^9.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


epochs=3
learningrate=1e-5
bert_classifier, optimizer, scheduler = initialize_model(epochs,learningrate)
set_seed(50)    # Set seed for reproducibility

train(bert_classifier, train_dataloader, epochs, evaluation=False)
torch.save(bert_classifier,main_path+'/models/saved_models/userembedding_model_single_auth.pt')








