## Biomedical event extraction based on GRU integrating attention mechanism

The section contains the code used for [Biomedical event extraction based on GRU integrating attention mechanism](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6101075/pdf/12859_2018_Article_2275.pdf) paper.

To reimplement this model, you are supposed to download the required material files as follows:

- [Pre-trained biomedical embeddings (BE)](https://drive.google.com/file/d/1IefJDnse6Y0lFlF-Nxtrhsi2l2bsa9AZ/view?usp=sharing) from `Training Word Embeddings for Deep Learning in Biomedical Text Mining Tasks` paper

And update the path setting variables: `DATA_DIR_PATH`, `LI_W2V_MODEL_PATH`, and `OUTPUT_DIR_PATH`, in `constants.py` to your local environments.

After that, to create the predictions, run the following command:

```
python main.py
```

Please note that how to reimplement the pre-trained biomedical embeddings is describe in `domain-oriented word2vec` folder.
