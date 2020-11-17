# CS685_Sarcasm
User-Augmented Transformer-based Sarcasm Detector on the SARC Dataset

## Accessing the Data
Each text file contains a single line per example:\
`example_number sarcasm_label list_of_space_tokens`\
The `user_tok` directory contains data of the format:\
`example_number sarcasm_label user_id list_of_space_tokens`

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
