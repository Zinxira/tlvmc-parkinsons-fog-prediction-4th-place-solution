# 4th Place Solution: a MultiLayer Bidirectional GRU with Residual Connections
This archive contains the 4th place solution for the "Parkinson's Freezing of Gait Prediction" Kaggle competition (https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction). See this post https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/discussion/416410 for a more complete view of this solution. 

## HARDWARE
The following specs were used to create the original solution:
- OS: Microsoft Windows 10 Family, Version 10.0.19045 Build 19045
- CPU: Processor Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz, 3696 Mhz, 10 Cores, 20 Logical Processors
- GPU: 1 x NVIDIA GeForce RTX 3080
- Memory : 1TB
- RAM: 64 GB

## SOFTWARE 
- Python 3.10.10
- CUDA 11.7

The full list of the python packages we used along with their versions can be found in the "requirements.txt" file. We also recommend to create a virtual environment dedicated to this project (for instance with [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)), with the same python packages versions as the ones we used. We can not guarantee the normal operation of our code otherwise. 

To install torch with CUDA enabled, we used the following command: ```pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117```; however you might want to adapt it to your needs. You can find additional information here: https://pytorch.org/get-started/previous-versions/

## DATA SETUP 
(assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)

Below are the shell commands used to download the data from Kaggle, as run from the top level directory:
```
mkdir input
cd input
kaggle competitions download -c tlvmc-parkinsons-freezing-gait-prediction
unzip tlvmc-parkinsons-freezing-gait-prediction.zip
```

## MODEL TRAINING
To reproduce the results obtained during the competition and obtain the same model as the one which scored 0.417 on the private leaderboard, you can execute the cell of the "Training" section of the tlvmc-parkinsons-freezing-gait-prediction.ipynb notebook. Note that all the preprocessing steps are done on the fly during the execution of this cell. 

You can also try different parameters: for instance if you want to train a simpler version of the model which only uses 1 layer, you can change the NL parameter in the PARAMS dict to 1. This corresponds to the simplified model described in the model summary, which obtained 0.388 on the private leaderboard. 

The model along with the training logs are saved in models/MultiResidualBiGRU_\[TIMESTAMP\]. Note that we only keep the version of the model that has the best validation loss, which is often before the last epoch. 

## USAGE OF PRETRAINED MODELS
If you do not want to retrain a model by yourself but only want to perform inference, you can skip the "Training" section and only consider the "Loading a model & Inference" section of the tlvmc-parkinsons-freezing-gait-prediction.ipynb notebook. Here you will need to change the model_id variable to the name of the folder containing the model you wish to use. We provide two pretrained models in .pth files in the folder "models":
- best_model: the model that achieved a score of 0.417 on the private leaderboard;
- simplified_model: the simplified version of the best model, which uses only 1 layer instead of 3. Achieved a score of 0.388 on the private leaderboard.

Note that as with model training, the pre and post processing steps are done on the fly. 

Once you have modified model_id, just execute all the cells of the "Loading a model & Inference" section: it will perform inference on the data stored in the "sample_submission.csv" file and produce a "submission.csv" file containing the corresponding predictions. 
