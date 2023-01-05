# Product Text Classification
## Project Structure

```
├── data
│   ├── checkpoints
│   │   └── brands
│   └── samples
│       └── samples.csv
├── models.py
├── product_classifier_infer.py
├── requirements.txt
├── train.py
└── utils.py
```
## Train

### Data
Sample data in `product_text_classifier/data/samples/samples.csv`

|text|label1|label2|label3|filename|
|---|---|---|---|---|
|grow 4 1 abbott weight helps abbott 900g higher intelligence 26 power nent foreng higher,|abbott|abbott grow|abbott grow 1|all_crop_data/abbott/abbott_grow/abbott_grow_1/0008.jpg

#### Data description
* **text**: product text content
* **label1**: brand name
* **label2**: intermediate level
* **label3**: step name
* **filename**: product crop image name

### Train
```
 python -m product_text_classifier.train --data_fn product_text_classifier/data/samples/samples_text.csv
```

# Text Recognition

## Project Structure
```
├── create_lmdb_dataset.py
├── data
│   ├── pretrained
│   │   └── TPS-ResNet-BiLSTM-Attn.pth
│   └── raw_data
├── dataset.py
├── demo.ipynb
├── demo.py
├── LICENSE.md
├── model.py
├── modules
│   ├── feature_extraction.py
│   ├── prediction.py
│   ├── sequence_modeling.py
│   └── transformation.py
├── README.md
├── requirements.txt
├── test.py
├── train.py
└── utils.py
```

## Data

Train and test data in `text_recognition/data/raw_data/train_labels.txt` and `text_recognition/data/raw_data/valid_labels.txt`

### Description
Data format
```
#path_to_file    text
data_1-2w_v3/out/13942.jpg	Bich
data_1-2w_v3/out/18503.jpg	Bênh
```

## Train
### Create dataset
```
# create train dataset
python create_lmdb_dataset.py \
--inputPath ./text_recognition/data/raw_data \
--gtFile ./text_recognition/data/raw_data/train_labels.txt 
--outputPath data/train_data

# create valid dataset
python create_lmdb_dataset.py \
--inputPath ./text_recognition/data/raw_data \
--gtFile ./text_recognition/data/raw_data/valid_labels.txt 
--outputPath data/valid_data
```

### train
```
python -W ignore train.py \
--train_data data/train_data \
--valid_data data/valid_data \
--Transformation TPS \
--FeatureExtraction ResNet \
--SequenceModeling BiLSTM \
--Prediction Attn \
--num_iter 100000 \
--batch_size 192 \
--imgW 100 \
--imgH 32 \
--workers 0 \
--batch_max_length 80 \
--valInterval 500 \
--exp_name TPS_ResNET_BiLSTM_Attn_additional \
--PAD \
--FT \
--saved_model text_recognition/data/pretrained/TPS-ResNet-BiLSTM-Attn.pth
```
