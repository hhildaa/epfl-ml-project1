# EPFL Machine Learning Course - PROJECT 1
## Fall semester, 2021
###

### Description
In this project, our task was to apply basic machine learning techniques to CERN particle accelerator data. The goal of the project was to recreate the process of finding Higgs particles. Our best approach was regularized logistic regression with splitting on number of the jets, feature expansion, median imputing and outlier bounding. We could achive 0.831% with this approach on the AIcrowd platform of this challenge (https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/leaderboards). This project was a part of our Machine Learning Course at EPFL in 2021 Fall semester.

### Table of contents
- Technology requirements
- Documentation
- Data
- Running the code

### Technology requirements
For working with the code, the following technologies are required:
- Python ???
- Numpy ???
- Matplotlib

We were not allowed to use any other external libraries. 

### Documentation



### Data
The data is available in the data folder compressed. Before working with the code, it should be uncompressed.

### Running the code

Move to the root folder. There executing 
```
python run.py
```
In this way the code will run with default settings. 

If you want to run with other setting write a similar code as below.
```
python run.py --verbose --feature_expansion --remove_outliers --split_jet --algorithm reg_logistic
```

The terms mean the following:
- verbose: allows detailed logs
- feature_expansion: allows adding additional features
- remove_outliers: allows outlier removing
- split_jet: allows splitting data according to jet number
- algorithm: here you can choose the type of the used algorithm

