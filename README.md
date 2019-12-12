# Relation extraction between bacteria and biotopes from biomedical texts with attention mechanisms and domain-specific contextual representations

The repository contains the code used for `Relation extraction between bacteria and biotopes from biomedical texts with attention mechanisms and domain-specific contextual representations` paper on `BMC Bioinformatics` publication (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3217-3).
These following instructions will get you a copy of the project up and running on your local machine for reimplementation the proposed model.

## Installing

In this project, python 3.6 (or later) and PyTorch 0.4 are mainly required for the current codebase. You need to clone the project:

```
git clone https://github.com/ammarinjtk/Neural-Relation-Extraction.git
```

and then install the dependencies by running:

```
pip install -r requirements.txt
```

## Prerequisites

To get started with the project, you are supposed to download the required material files as follows:

- [Pre-trained word2vec](https://drive.google.com/file/d/1eXeHKZh_PhxA2hf0NRDpODBZ4zk6L711/view?usp=sharing)
- [Pre-trained domain-specific ELMo](https://drive.google.com/drive/folders/1tnSlCfgFgcJgkWTn-xzAohPGjSAiJOm8?usp=sharing)
- [Pre-trained BERT](https://drive.google.com/drive/folders/1ySPqub2DRFHYqyfh9qWS0QeJApfrKWwQ?usp=sharing)

Please note that these are very large files, so downloading may take quite a while.

After you have downloaded the files, unzip them and update the path setting variables: `DATA_DIR_PATH`, `W2V_MODEL_PATH`, `ELMO_MODEL_PATH`, `ELMO_OPTIONS_PATH`, `BERT_FEATURES_PATH`, `BERT_TKN_PATH`, and `OUTPUT_DIR_PATH`, in `constants.py` to your local environments.

## Running the system

To reproduce results from the paper, you should run the following command (it will take 3-4 hours to run on GPU (GTX 1080 Ti), as it iteratively trains the model for many times):

```
python main.py
```

Please note that you can use tmux as a screen multiplexer in the running process since it can resume the session at any time you desire without worrying about the session being logged out or terminal being closed.

## Evaluation

The model results, shown in `final_predictions` folder, can be evaluated with the online evaluation service provided by [BioNLP-ST’16](http://bibliome.jouy.inra.fr/demo/BioNLP-ST-2016-Evaluation/index.html).

Please note that these results (zip files) will be stored in your `OUTPUT_DIR_PATH`.

## Reimplementations

Since there has been no comparison of existing models with our bootstrap method to evaluate the model performance, it is necessary for us to reimplement other existing models to compare with our proposed model. So that, we reimplemented two existing models, [TurkuNLP](http://aclweb.org/anthology/W16-3009) as the deep learning baseline and the best system [Li](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6101075/pdf/12859_2018_Article_2275.pdf) in the `reimplementations` folder.

## Authors

- **Amarin Jettakul** - _Initial work_ - [ammarinjtk](https://github.com/ammarinjtk)

See also the list of [contributors](https://github.com/ammarinjtk/Neural-Relation-Extraction/graphs/contributors) who participated in this project.

## Acknowledgments

- All additional resources are released at [Google Drive](https://drive.google.com/drive/folders/1u7e86ZwqSNERDXjR5tec63Id0nTJBTVO?usp=sharing).
- The author received the scholarship from Chula Computer Engineering Graduate Scholarship for CP Alumni.
