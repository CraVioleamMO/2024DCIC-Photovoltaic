

# Photovoltaic power generation output forecast

## 1. Enviroment setup
```bash
conda create -n DM python==3.8.0
conda activate DM
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. The directory structure
```bash
|- Input
  |- train # store the train data
    |- power.csv
    |- basic_information.csv
    |- weather.csv
  |- test # store the test data
|- Output # store the submission file
|- Inter_output # store the feature file like feature.csv
|- config # store the yaml configuration file
|- notebooks # store the data analysis result
|- src # the source code
```
When downloaded the data, you should place the data in the Input directory and rename the data as **basic_information.csv, power.csv and weather.csv**
## 3. Run the script
```bash
python feature_engineer.py # generate the feature file
python train.py # train the model
```
You can run the feature_engineer.py to generate the feature file and run the train.py to train the model and 
generate the submission file.

The feature configuration is in the config/fe.yaml file. The model configuration is in the config/lgbm.yaml file. 
More detail can be found in the config file.


