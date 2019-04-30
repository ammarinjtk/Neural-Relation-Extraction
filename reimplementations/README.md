## Reimplemented Existing Relation Extractions

As there was no comparison of existing models with the mean F1-score, it was necessary for us to reimplement other existing models with the random seeds to compare with our model. We used Pytorch library to reimplement two existing models: TurkuNLP [16] as the deep learning baseline and BGRU-Attn [18] as the best system.
To reimplement TurkuNLP model, as described in [16]. However, instead of constructing the ensemble of 15 deep learning models, we built only the best-performed model based on the validation dataset. The model was trained for 4 epochs using Adam optimizer. Batch size was set to 1 and other hyper-parameters were set as in the paper. The reimplemented model achieved the maximum F1-score of 51.99% compared to 52.10% which was outlined in [16].
To reimplement the BGRU-Attn model, we utilized a Bidirectional GRU based on Additive attention mechanism over the dynamic extended trees with words, POS tags, and relative distances as the inputs. Since the method to train domain-oriented word embedding model is not provided in the paper, we employed the Skip-gram word embedding model based on biomedical entities and syntactic chunks using hierarchical softmax method described in the author’s previous paper [54]. Also, due to the fact that some of the hyper-parameters were also not given in the paper, we empirically chose the hyper-parameters using 3-fold cross-validation. The hidden unit number of GRU was 256 and the number of layers was 3. To avoid overfitting when training, we performed early stopping where the performance was evaluated on the development data. The reimplemented BGRU-Attn model achieved the maximum F1-score of 55.54% compared to 57.42%, presented in the paper.


16.	Mehryary, F., Bj ̈orne, J., Pyysalo, S., Salakoski, T., Ginter, F.: Deep Learning with Minimal Training Data: TurkuNLP Entry in the BioNLP Shared Task 2016. Acl 2016, 73 (2016) 

18.	Li, L., Wan, J., Zheng, J., Wang, J.: Biomedical event extraction based on GRU integrating attention mechanism. BMC Bioinformatics (2018). doi:10.1186/s12859-018-2275-2 

54.	Jiang, Z., Li, L., Huang, D., Jin, L.: Training word embeddings for deep learning in biomedical text mining tasks. In: Proceedings - 2015 IEEE International Conference on Bioinformatics and Biomedicine, BIBM 2015 (2015). doi:10.1109/BIBM.2015.7359756 

1. [TurkuNLP](https://github.com/ammarinjtk/Neural-Relation-Extraction/tree/master/reimplementations/TurkuNLP)
1. [Li](https://github.com/ammarinjtk/Neural-Relation-Extraction/tree/master/reimplementations/Li)
