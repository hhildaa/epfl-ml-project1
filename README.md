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
We executed and tested our code on the following libraries and resp. versions:
- Python 3.8
- Numpy 1.21.2
- Matplotlib 3.4.3

We were not allowed to use any other external libraries. 

### Documentation

The documentation of the project can be found under documentation/The_Higgs_Boson_Machine_Learning_Challenge.pdf. More information about our task can be found also in this folder under the name of project_description.pdf.

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
python run.py --verbose --feature_expansion --bound_outliers --impute_median  --split_jet --algorithm reg_logistic
```

The terms mean the following:
- verbose: allows detailed logs
- feature_expansion: allows adding additional features
- bound_outliers: allows outlier bounding
- split_jet: allows splitting data according to jet number
- algorithm: here you can choose the type of the used algorithm (one of `['reg_logistic', 'logistic', 'least_squares_GD', 'least_squares_SGD']`)
- output_file: add the name of the file to save the output to
- impute_median: impute the median for missing values
- k_folds: change the number of k-folds for cross-validation
- max_iters: change the number of iterations in (stochastic) gradient descent
- batch_size: change the batch size for stochastic gradient descent
- gamma: change the learning rate for (stochastic) gradient descent

