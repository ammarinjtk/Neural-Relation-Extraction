## Training Word Embeddings for Deep Learning in Biomedical Text Mining Tasks

The section contains the code used for `Training Word Embeddings for Deep Learning in Biomedical Text Mining Tasks` paper.

To reimplement this model, you are supposed to download the required material files as follows:

- [Token (.pickle) files](https://drive.google.com/drive/folders/1QfyTYGVpjPEGEfpFSwZFf9AfJ0l_wTvO?usp=sharing) which were extracted from PubMed database using `bacteria` as a keyword.

And update the path setting variables: `PICKLE_PATH` and `LI_W2V_OUTPUT_DIR_PATH`, in `constants.py` to your local environments.

In this project, Gensim is mainly required. You should first prepare inputs for the model using the following commands:

```
python create_vocab.py
```

and then

```
python create_ent_pairs.py
```

Afer that, to pre-train the word embedding model, the following command could be used:

```
python train_gensim.py
```
