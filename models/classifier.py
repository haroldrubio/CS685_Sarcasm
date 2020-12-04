# %%time
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
# model = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
from transformers import RobertaTokenizer, RobertaModel
from transformers import DistilBertModel , DistilBertTokenizerFast
# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self,h_dims,out_dims,model_type='bert',freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        self.out_dims=out_dims
        self.h_dims=h_dims
        self.model_type=model_type
        D_in, H, D_out = 768, self.h_dims, self.out_dims
        if model_type=='bert':
            # Instantiate BERT model
            print("here")
            self.model = BertModel.from_pretrained('bert-base-uncased')
        else:
            # Instantiate BERT model
            self.model = RobertaModel.from_pretrained('roberta-base')
        

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
#         print("input_ids",input_ids.size())
#         print("attention_mask",attention_mask.size())
        # Feed input to BERT
        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)
        return logits

    def get_embedding(self,input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        return self.classifier[0](last_hidden_state_cls)



class Net(nn.Module):
  def __init__(self, n_classes):
    super(Net, self).__init__()
    self.model = RobertaModel.from_pretrained('roberta-base')
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.model.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    # print(pooled_output.size())
    output = self.drop(pooled_output)
    return self.out(output)



class EmotionClassifier(nn.Module):
    """classification model with external features.
    """
    def __init__(self,h_dims,out_dims,emo_dims):
        super(EmotionClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        self.out_dims=out_dims
        self.h_dims=h_dims
        self.emo_dims=emo_dims
        H, D_out =  self.h_dims, self.out_dims

       
        

        # Instantiate an one-layer feed-forward classifier
        self.fc1 = nn.Linear(self.emo_dims, H)
        self.fc2 = nn.Linear(H, D_out)
       
        
    def forward(self, x):
        """
        independently classify the emotion vectors .
        @param    emotion vector
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        out1 = self.fc1(x)
        # print(out1.size())
        logits = self.fc2(F.relu(out1))
        return logits


class UserAugmentedClassifier(nn.Module):
    def __init__(self, usermodel,robmodel,h_dims,out_dims):
        super(UserAugmentedClassifier, self).__init__()
        self.out_dims=out_dims
        self.h_dims=h_dims
        D_in, H, D_out = 768+self.h_dims+6, 50, self.out_dims
        self.usermodel = usermodel
        self.robmodel=robmodel
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
        for param in usermodel.parameters():
            param.requires_grad = False
        for param in robmodel.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask,rob_input_ids, rob_attention_mask,x3):
        # outputs = self.model(input_ids, attention_mask)
        # last_hidden_state_cls = outputs[0][:, 0, :]
        # # print(last_hidden_state_cls.size())
        x1 = self.usermodel.get_embedding(input_ids, attention_mask)
        # print("x1 size",x1.size())
        _,x2 = self.robmodel.model(rob_input_ids, rob_attention_mask)

        # print("x2 size",x2.size())
        x = torch.cat((x1, x2,x3), dim=1)
        
        x = self.classifier(x)
        return x

class SubRedditClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SubRedditClassifier, self).__init__()
        # config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    # for p in self.bert.parameters():
    #   p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # print(output)
        hidden = output.last_hidden_state
        # pooled_output = output.pooler_output
        output = hidden[:,0,:] 
        return self.out(output)

    def get_embedding(self,input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        return last_hidden_state_cls


class SubAugmentedClassifier(nn.Module):
    def __init__(self, usermodel):
        super(SubAugmentedClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2
        self.usermodel = usermodel
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.model2 = RobertaModel.from_pretrained('roberta-base')
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
        for param in self.usermodel.parameters():
            param.requires_grad=False

        
    def forward(self, input_ids, attention_mask):
        outputs = self.usermodel.bert(input_ids, attention_mask)
        x = outputs[0][:, 0, :]
        # x1 = self.usermodel.classifier[0](last_hidden_state_cls)
        # print("x1 size",x1.size())
        # out = self.model2(rob_input_ids, rob_attention_mask)
        # x2 = out[0][:, 0, :]
        # # print("x2 size",x2.size())
        # x = torch.cat((x1, x2,x3), dim=1)
        # print(x)
        x = self.classifier(x)
        return x