import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from classifier import *
import torch
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import sys
import argparse
from sklearn.metrics import confusion_matrix
from  utils import *
from sentence_transformers import SentenceTransformer, util

from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, classification_report
batch_size = 32
MAX_LENGTH=50
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


parser = argparse.ArgumentParser()
## learning
parser.add_argument('--epoch', type=int, default=2, help='number of epochs for train [default: 2]')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 32]')
# parser.add_argument('--emotion_aggregated',action="store_true", help='Whether to use emotion representation')
# parser.add_argument('--learningrate', type=float, default=1e-3, help='learning rate [default: 1e-3]')
parser.add_argument(
        '--modeltype',
        type=str,
        default='user',
        choices=['user', 'uemohybrid','roberta','bert','uhybrid'],
        help='Type of models [default: user]'
    )
args = parser.parse_args()



def read_data_from_file(file_path,test_size):
    train_df=pd.read_csv(file_path+"/train_main_users.txt",delimiter="\t",header=None)
    test_df=pd.read_csv(file_path+"/test_main_users.txt",delimiter="\t",header=None)
    # train_df = train_df[:20000]
    # test_df=test_df[:4000]
    X_train=train_df[2].values
    y_train=train_df[0].values
    X_test=test_df[2].values
    y_test=test_df[0].values
    print(len(y_test))
    len_train=len(X_train)
    # user_names=train_df[1].unique()
    # user_names=np.unique((np.append(user_names,test_df[1].unique())))
    # spliting test data into validation and test
    random_state = 50
    indices = np.arange(len(X_test))
    train_idx, val_idx, y_val, y_test= train_test_split(indices, y_test,stratify = y_test, test_size=test_size, random_state=random_state)
    X_val = X_test[train_idx]
    X_tst = X_test[val_idx]


    path='/home/nazaninjafar/UMass/github_repo/CS685_Sarcasm/user_tok/'
    timeline_posts=pd.read_csv(path+"/timeline_posts.tsv")

    timeline_posts["user"] = timeline_posts["user"].astype('category')
    timeline_posts["label_coded"] = timeline_posts["user"].cat.codes
    X = timeline_posts.post.values
    y = timeline_posts.label_coded.values
    usernums=len(timeline_posts.label_coded.unique())
    print(usernums)
    emo_vecs = np.load(main_path+'models/saved_models/balanced_main/emo_vecs.npy')
    emo_train=emo_vecs[:len_train]
    emo_val=emo_vecs[len_train:]

    X_emo_train=torch.tensor(emo_train).type(torch.FloatTensor)
    X_emo_val = emo_val[train_idx]
    X_emo_test = emo_val[val_idx]
    X_emo_val=torch.tensor(X_emo_val).type(torch.FloatTensor)
    X_emo_test=torch.tensor(X_emo_test).type(torch.FloatTensor)
    print(X_emo_train.size())
    print(X_emo_val.size())
    print(X_emo_test.size())

    return X_train,y_train,X_val,y_val,X_tst,y_test,usernums,X_emo_train,X_emo_val,X_emo_test


def tokenize_input(X_train,X_val,X_test,load_dir):
    # tokenizer = RobertaTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # else:
    #     print("should be here")
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # train_inputs, train_masks = preprocessing_for_bert(X_train,MAX_LENGTH,tokenizer)
    # val_inputs, val_masks = preprocessing_for_bert(X_val,MAX_LENGTH,tokenizer)
    # test_inputs, test_masks = preprocessing_for_bert(X_test,MAX_LENGTH,tokenizer)
    train_inputs = torch.load(f'{load_dir}/train_inputs.pt')
    train_masks=torch.load(f'{load_dir}/train_masks.pt')
    val_inputs = torch.load(f'{load_dir}/val_inputs.pt')
    val_masks = torch.load(f'{load_dir}/val_masks.pt')
    test_inputs = torch.load(f'{load_dir}/test_inputs.pt')
    test_masks = torch.load(f'{load_dir}/test_masks.pt')
    # train_inputs = train_inputs[:20000][:]
    # train_masks = train_masks[:20000][:]
    # val_inputs = val_inputs[:2000][:]
    # val_masks = val_masks[:2000][:]
    # test_inputs = test_inputs[:2000][:]
    # test_masks = test_masks[:2000][:]
    print(train_inputs.size())
 
    return train_inputs, train_masks,val_inputs,val_masks,test_inputs, test_masks
def get_dataloaders(y_train,y_val,y_test,train_inputs, train_masks,val_inputs, val_masks,test_inputs, test_masks,batch_size):
    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)
    test_labels = torch.tensor(y_test)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    # Create the DataLoader for our test set
    test_data = TensorDataset(test_inputs, test_masks, val_labels)
    test_sampler = SequentialSampler(val_data)
    test_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    return train_dataloader,val_dataloader,test_dataloader


def load_user_model(model_path,num_user):
    #initialize model:
    h_dims = 512
    out_dims=num_user
    user_model = BertClassifier(h_dims,out_dims)
    user_model=torch.load(model_path+'userembedding_model_single_auth.pt')
    return user_model


learningrate = 2e-5
def initialize_model(train_dataloader,user_model,epochs=4,lr=learningrate,emotion_aggregated=False,model='uhybrid'):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    h_dims=50 #hidden dimesion
    out_dims=2 #number of classes
    emo_dims=6
    if model == 'uhybrid':
        h_dims=512
        print("initializing model..")
        classifier = UserAugmentedClassifier(user_model,h_dims,out_dims)
    else:
        classifier = FinetunedClassifier(h_dims,out_dims,emo_dims)
    
    # h_dims=50 #hidden dimesion
    # out_dims=2 #number of classes
    # print(model)
    # if emotion_aggregated:
    #     emo_dims=6
    #     classifier=MergedClassifier(h_dims,out_dims,model)
    # else:
    #     # Instantiate Bert Classifier
        
    #     classifier = BertClassifier(h_dims,out_dims,model,freeze_bert=False)
    # Tell PyTorch to run the model on GPU
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

def train(model, train_dataloader, val_dataloader=None, epochs=4,evaluation=False):
    """Train the BertClassifier model.
    """

    

    #early stopping
    patience = 5
    epochs_no_improve = 0
    early_stop = False
    min_val_loss=np.Inf
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
            # Zero out any previously calculated gradients
            model.zero_grad()
            b_input_ids, b_attn_mask,b_input_ids2, b_attn_mask2,b_emo, b_labels = tuple(t.to(device) for t in batch)
            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask,b_input_ids2, b_attn_mask2,b_emo)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
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
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {train_accuracy:^9.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            if epoch_i > patience and epochs_no_improve >= patience:
                print('Early stopping!' )
                early_stop = True
                break
            else:
                continue
            print("-"*70)
        print("\n")

    
    print("Training complete!")
    return val_accuracy




def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask,b_input_ids2, b_attn_mask2,b_emo,_ = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask,b_input_ids2, b_attn_mask2,b_emo)
        all_logits.append(logits)
    
    
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    preds=[]
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask,b_input_ids2, b_attn_mask2,b_emo, b_labels = tuple(t.to(device) for t in batch)
        # # Perform a forward pass. This will return logits.
        # logits = model(b_input_ids, b_attn_mask,b_input_ids2, b_attn_mask2)
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask,b_input_ids2, b_attn_mask2,b_emo)
            # Compute loss
            loss = loss_fn(logits, b_labels)
            print("val loss:",loss)
            val_loss.append(loss.item())

            # Get the predictions
        pred= torch.argmax(logits, dim=1).flatten()
        preds.append(pred)
        # Calculate the accuracy rate
        accuracy = (pred == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy,preds

if __name__ == "__main__":
    main_path='/home/nazaninjafar/UMass/github_repo/CS685_Sarcasm/'
    saved_model_path='/home/nazaninjafar/UMass/github_repo/CS685_Sarcasm/models/saved_models/balanced_main/'
    model_type=args.modeltype
    epochs=args.epoch

    #hyperparameters
    batch_size=32
    test_split=0.5
    lr=1e-3
    save_dir_bert = '/home/nazaninjafar/UMass/github_repo/CS685_Sarcasm/models/tokenized_data/bert'
    save_dir_roberta = '/home/nazaninjafar/UMass/github_repo/CS685_Sarcasm/models/tokenized_data/roberta'
    X_train,y_train,X_val,y_val,X_test,y_test,user_nums,X_emo_train,X_emo_val,X_emo_test = read_data_from_file(main_path,test_split)
    user_model=load_user_model(saved_model_path,user_nums)
    print(len(y_val))
    train_inputs, train_masks,val_inputs,val_masks,test_inputs, test_masks=tokenize_input(X_train,X_val,X_test,save_dir_bert)
    train_inputs_rob, train_masks_rob,val_inputs_rob,val_masks_rob,test_inputs_rob, test_masks_rob=tokenize_input(X_train,X_val,X_test,save_dir_roberta)
    len_train = len(X_train)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)
    test_labels = torch.tensor(y_test)
    

    batch_size = 32

    print(train_inputs.size())
    print(train_inputs_rob.size())
    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)
    test_labels = torch.tensor(y_test)
    train_data = TensorDataset(train_inputs, train_masks,train_inputs_rob, train_masks_rob,X_emo_train, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks,val_inputs_rob,val_masks_rob,X_emo_val, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    # Create the DataLoader for our test set
    test_data = TensorDataset(test_inputs, test_masks,test_inputs_rob, test_masks_rob,X_emo_test, val_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=val_sampler, batch_size=batch_size)
    print("data loading successfully done by pytorch")
    # return train_dataloader,val_dataloader,test_dataloader
    # train_dataloader, val_dataloader, test_dataloader = get_dataloaders(y_train,y_val,y_test,train_inputs, train_masks,val_inputs, val_masks,test_inputs, test_masks, batch_size)
    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()


    #start training..
    print("start training .. ")
    # model_type='user'
    
    bert_classifier, optimizer, scheduler = initialize_model(train_dataloader,user_model,epochs,lr,model=model_type)
    val_acc=train(bert_classifier, train_dataloader, val_dataloader, epochs,evaluation=True)
    #     if val_acc>best_configs['max_acc']:
    #         best_configs['max_acc'] = val_acc
    #         best_configs['lr']=lr
    # print(best_configs)
    test_loss, test_accuracy,y_pred= evaluate(bert_classifier, test_dataloader)
    np_preds=[]
    for i in y_pred:
        b=i.cpu().detach().numpy()
        np_preds=np.append(np_preds,b,axis=0)
    np_preds=np_preds.astype(int) 
    print(np_preds.shape)
    print(y_test.shape)
    class_names = ['sarcastic', 'not sarcastic']
    print(classification_report(y_test, np_preds, target_names=class_names))
    # confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, np_preds).ravel()
    print("confusion matrix: ",tn,fp,fn,tp)
    with open('misclassified_test.txt', 'a') as the_file:
        
        for i in range(len(y_test)):
            if y_test[i] !=np_preds[i]:
                the_file.write(str(i)+'\t'+X_test[i]+'\n')
    # the_file.close()
    # #     if val_acc>best_configs['max_acc']:
    # #         best_configs['max_acc'] = val_acc
    # #         best_configs['lr']=lr
    # # print(best_configs)
    # test_loss, test_accuracy,y_pred= evaluate(bert_classifier, test_dataloader)
    # np_preds=[]
    # for i in y_pred:
    #     b=i.cpu().detach().numpy()
    #     np_preds=np.append(np_preds,b,axis=0)
    # np_preds=np_preds.astype(int) 
    # class_names = ['sarcastic', 'not sarcastic']
    # print(classification_report(y_test, np_preds, target_names=class_names))
    # torch.save(bert_classifier,f'{main_path}/models/saved_models/{model_type}ablation.pt')