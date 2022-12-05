<img src="https://github.com/biomed-AI/SPROF-GO/blob/main/image/logo.png" width = "150" height = "150" alt="logo" align=left />  
# Introduction
SPROF-GO is an alignment-free sequence-based protein function predictor through pretrained language model and homology-based label diffusion. SPROF-GO is easy to install and run, and is also accurate (surpassing the state-of-the-art sequence-based and even network-based methods) and really fast. Empirically, prediction on the three ontologies for 1000 sequences with an average length of 500 only takes about 7 minutes using  an Nvidia GeForce RTX 3090 GPU. If your input is small, you can also use our [SPROF-GO web server](http://bio-web1.nscc-gz.cn/app/SPROF-GO).
![overview](https://github.com/biomed-AI/SPROF-GO/blob/main/image/overview.png)

# System requirement
SPROF-GO is developed under Linux environment with:  
python  3.8.5  
numpy  1.19.1  
scipy  1.5.2  
torch  1.13.0  
sentencepiece  0.1.96  
transformers  4.17.0  
tqdm  4.59.0  

# Set up SPROF-GO
1. Clone this repository by `git clone git@github.com:biomed-AI/SPROF-GO.git` (~ 750 MB) or download the code in ZIP archive (~ 630 MB)
2. Download the pretrained ProtT5-XL-UniRef50 model in [here](https://zenodo.org/record/4644188) (~ 5.3 GB)
3. Set the path variable `ProtTrans_path` in `./script/predict.py`
4. Add permission to execute for DIAMOND by  `chmod +x ./script/diamond`

# Run SPROF-GO for prediction
Simply run:
```
python ./script/predict.py --fasta ./example/demo.fa --outpath ./example/
```
And the prediction results will be saved in `demo_top_preds.txt` and `demo_all_preds.txt` under `./example/`. Here we provide the corresponding canonical input and prediction results under `./example/` for your reference.

Other parameters:
```
--top           Besides the full predictions, also show the terms with top K predictive scores, default=20
--feat_bs       Batch size for ProtTrans feature extraction, default=8
--pred_bs       Batch size for SPROF-GO prediction, default=8
--save_feat     Save intermediate ProtTrans features
--gpu           Use GPU for feature extraction and SPROF-GO prediction
```

# Dataset and model
We provide the datasets and the trained models here for those interested in reproducing our paper.  
The protein function datasets used in this study are stored in `./datasets/` as ZIP archives.  
The trained SPROF-GO models can be found under `./model/`.  

# Citation and contact
Citation:  
```bibtex
@article{10.1093/bib/bbac444,
    author = {Yuan, Qianmu and Chen, Sheng and Wang, Yu and Zhao, Huiying and Yang, Yuedong},
    title = "{Alignment-free metal ion-binding site prediction from protein sequence through pretrained language model and multi-task learning}",
    journal = {Briefings in Bioinformatics},
    year = {2022},
    month = {10},
    issn = {1477-4054},
    doi = {10.1093/bib/bbac444},
    url = {https://doi.org/10.1093/bib/bbac444},
}
```

Contact:  
Qianmu Yuan (yuanqm3@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)
