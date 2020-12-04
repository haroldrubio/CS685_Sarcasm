# CS685_Sarcasm
User-Augmented Transformer-based Sarcasm Detector on the SARC Dataset

## Additional JSON Data
Link to the [subreddit](https://drive.google.com/file/d/11F3ebmz1If_330KhwV9jMnpmZCGkICOl/view?usp=sharing) dataset\
This is a list of objects that maps 'text' to the raw post, and 'subreddit' to the originating subreddit. This contains all posts, ancestors and responses, that appear only in the unbalanced dataset.

## Accessing the Data
Each text file contains a single line per example:\
`example_number sarcasm_label list_of_space_separated_tokens`\
The `user_tok` directory contains data of the format:\
`example_number sarcasm_label user_id list_of_space_separated_tokens`

## Data
**Note: There are 4 RoBERTa-tokenized posts in main_tok that exceed the 512 sequence length constraint.**\
The data included is the processed data from the [SARC](https://nlp.cs.princeton.edu/SARC/2.0/) dataset. `main_tok` and `pol_tok` contain tokenized versions of the data in `main` and `pol` from the SARC dataset by the NLTK word tokenizer and hugging-face RoBERTa tokenizer. The data included are single post responses.\
Reddit posts appear in pairs: odd numbered posts are children of the preceeding even-numbered post.\
Below is the structure of the data-holding variables found in the `load_data` notebook, and the structure of the data in `main_tok` and `pol_tok`:
![cs685data](https://user-images.githubusercontent.com/43583679/99214836-3658ba80-279f-11eb-9cef-979076559a60.png)


## Dependencies
External Dependencies: nltk, transformers, np, matplotlib \
\
Download main and pol folders [here](https://nlp.cs.princeton.edu/SARC/2.0/) to look further into the data and to reproduce the included data.\
\
(Optional) Download a GLoVE embedding from: https://nlp.stanford.edu/projects/glove/ and modify the variable at the top of the notebook depending on the embedding you choose
# Models
For our hybrid model we first pretrain a user classification model. The code for this part is accessible in `models\ubert.py`. 
In this project we defined several classifiers which each of them is also accessble in `models\classifier.py`. 
Our main training code of hybrid model is in `models/main.py`. 
To run this code for `user  embedding` use this command below: 
`main.py --modeltype uhybrid --epoch 10` 
And to run this code for `subreddit embedding` use `--modeltype subybrid`. 
