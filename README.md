# CS685_Sarcasm
User-Augmented Transformer-based Sarcasm Detector on the SARC Dataset

## Data
**Note: There are 3 RoBERTa-tokenized posts in main_tok that exceed the 512 sequence length constraint.**\
The data included is the processed data from the [SARC](https://nlp.cs.princeton.edu/SARC/2.0/) dataset. `main_tok` and `pol_tok` contain tokenized versions of the data in `main` and `pol` from the SARC dataset by the NLTK word tokenizer and hugging-face RoBERTa tokenizer. The data included are single posts with no ancestors, responses or information about the author of the post; it only contains the raw tokens from the first entry in the response list for a given example. Below is the structure of the data-holding variables found in the `load_data` notebook, and the structure of the data in `main_tok` and `pol_tok`:
![cs685_data_2](https://user-images.githubusercontent.com/43583679/95688750-6fb76c80-0bda-11eb-9506-da9a3d38d614.png)


## Dependencies
External Dependencies: nltk, transformers, np, matplotlib \
\
Download main and pol folders [here](https://nlp.cs.princeton.edu/SARC/2.0/) to look further into the data and to reproduce the included data.\
\
(Optional) Download a GLoVE embedding from: https://nlp.stanford.edu/projects/glove/ and modify the variable at the top of the notebook depending on the embedding you choose
