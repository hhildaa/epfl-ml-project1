from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from scripts.implementations import *

from dataprocess.process import *
from model.cross_validation import *

from scripts.proj1_helpers import *
from scripts.argument_parser import *



def main(**params):
    params = dict(
        default_params,
        **params
    )
    # Set random seed
    np.random.seed(params['seed'])


    DATA_TRAIN_PATH = './data/train.csv'
    y_train, X_train, ids_train = load_csv_data(DATA_TRAIN_PATH, sub_sample=True)
    y_train_whole, X_train_whole, ids_train_whole = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

    DATA_TEST_PATH = './data/test.csv'
    _, X_test, ids_test = load_csv_data(DATA_TEST_PATH)

    if(params['verbose']):
            print("DATA READ")


    feature_list = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_deltaeta_jet_jet", "DER_mass_jet_jet",
                    "DER_prodeta_jet_jet", "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau", "DER_met_phi_centrality",
                    "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi", "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi",
                    "PRI_met", "PRI_met_phi", "PRI_met_sumet", "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta",
                    "PRI_jet_leading_phi", "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "PRI_jet_all_pt"]
    feature_ids = {feature:i for i, feature in enumerate(feature_list)}


    # TODO: REMOVING FEATURES OF TRAINING DATA FOR CROSS VALIDATION
    #       - BE CAREFUL THAT AFTER SPLITTING ON JET, INDICES OF FEATURES CHANGE



    # Without splitting on JET
    data_splits_train = {-1:(X_train, y_train, ids_train)}
    data_splits_train_whole = {-1:(X_train_whole, y_train_whole, ids_train_whole)}
    data_splits_test = {-1:(X_test, None, ids_test)}

    # With splitting on JET
    if(params['split_jet']):
        data_splits_train = split_data_by_feature(y_train, X_train, ids_train, feature_ids["PRI_jet_num"], train=True)
        data_splits_train_whole = split_data_by_feature(y_train_whole, X_train_whole, ids_train_whole, feature_ids["PRI_jet_num"], train=True)
        data_splits_test = split_data_by_feature(None, X_test, ids_test, feature_ids["PRI_jet_num"], train=False)


    if(params['verbose']):
            print("CROSS VALIDATION FOR PARAMETER ESTIMATION")


    best_degrees = {}
    best_lambdas = {}

    # Cross validation
    for jet in data_splits_train:
        X_train_jet, y_train_jet, _ = data_splits_train[jet]
        
        if(params['verbose']):
            print(f"JET NUMBER {int(jet)}:")     

        best_accuracy, best_degree, best_lambda, _, _ = k_fold_cross_validation(y_train_jet, X_train_jet, params['k_folds'],
                                                                                lambdas=np.logspace(-10, -2, 9),
                                                                                degrees=range(8, 12),
                                                                                max_iters=params['max_iters'],
                                                                                batch_size=params['batch_size'],
                                                                                gamma=params['gamma'],
                                                                                verbose=params['verbose'], 
                                                                                algorithm=params['algorithm'],
                                                                                params=params)
        best_degrees[jet] = best_degree
        best_lambdas[jet] = best_lambda

        if(params['verbose']):
            print(f"Best accuracy for JET {int(jet)}: {best_accuracy}")


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time: ", current_time)

######################################################################################################################
    
    # Cross validation on the whole training dataset (for the purpose of report)

    if(params['verbose']):
        print("CROSS VALIDATION ON THE WHOLE TRAINING DATASET (FOR THE PURPOSE OF REPORT)")

    all_preds = {}
    all_labels = {}
    for jet in data_splits_train_whole:
        X_train_whole_jet, y_train_whole_jet, _ = data_splits_train_whole[jet]
         

        _, _, _, preds, labels = k_fold_cross_validation(y_train_whole_jet, X_train_whole_jet, params['k_folds'],
                                                         lambdas=np.array([best_lambdas[jet]]),
                                                         degrees=np.array([best_degrees[jet]]),
                                                         max_iters=params['max_iters'],
                                                         batch_size=params['batch_size'],
                                                         gamma=params['gamma'],
                                                         verbose=params['verbose'], 
                                                         algorithm=params['algorithm'],
                                                         params=params)
        all_preds[jet] = preds
        all_labels[jet] = labels


    accs_test_whole = np.zeros(params['k_folds'])
    for k in range(params['k_folds']):
        fold_labels = []
        fold_preds = []
        for jet in data_splits_train_whole:
            fold_preds.append(all_preds[jet][k])
            fold_labels.append(all_labels[jet][k])

        fold_preds = np.concatenate(fold_preds, axis=0)
        fold_labels = np.concatenate(fold_labels, axis=0)
        accs_test_whole[k] = accuracy(fold_preds, fold_labels)

    print(f"FINAL Test (on the whole dataset): {accs_test_whole.mean():.4f} +- {accs_test_whole.std():.4f}")



    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time: ", current_time)


######################################################################################################################

    # Data preprocessing of the whole training dataset and testing dataset


    # Without splitting on JET
    data_splits_train_whole = {-1:(X_train_whole, y_train_whole, ids_train_whole)}
    data_splits_test = {-1:(X_test, None, ids_test)}

    # With splitting on JET
    if(params['split_jet']):
        data_splits_train_whole = split_data_by_feature(y_train_whole, X_train_whole, ids_train_whole, feature_ids["PRI_jet_num"], train=True)
        data_splits_test = split_data_by_feature(None, X_test, ids_test, feature_ids["PRI_jet_num"], train=False)



    if(params['verbose']):
        print("DATA PREPROCESSING OF THE WHOLE TRAINING DATASET AND TESTING DATASET")

    for jet in data_splits_train_whole:
        X_train_whole_jet, y_train_whole_jet, ids_train_whole_jet = data_splits_train_whole[jet]
        X_test_jet, y_test_jet, ids_jet = data_splits_test[jet]



        # TODO: REMOVING FEATURES - BE CAREFUL THAT AFTER SPLITTING ON JET, INDICES OF FEATURES CHANGE


        # Remove degenerated features
        X_train_whole_jet, removed_features_jet = remove_tiny_features(X_train_whole_jet)
        X_test_jet = remove_custom_features(X_test_jet, custom_feature_ids=removed_features_jet)
            


        # Imputing median
        if(params['impute_median']):
            X_train_whole_jet, median = impute_median(X_train_whole_jet, None)
            X_test_jet, _ = impute_median(X_test_jet, median)

        # Remove outliers
        if(params['remove_outliers']):
            X_train_whole_jet, upper_quart, lower_quart = bound_outliers(X_train_whole_jet, None, None)
            X_test_jet, _, _ = bound_outliers(X_test_jet, upper_quart, lower_quart)

        # Feature expansion
        if(params['feature_expansion']):
            X_train_whole_jet = build_poly(X_train_whole_jet, best_degrees[jet])
            X_test_jet = build_poly(X_test_jet, best_degrees[jet])

        # Standardize features
        X_train_whole_jet, mean, std = standardize(X_train_whole_jet, None, None)
        X_test_jet, _, _ = standardize(X_test_jet, mean, std)

        # Add bias
        bias_train = np.ones((X_train_whole_jet.shape[0], 1))
        X_train_whole_jet = np.concatenate((X_train_whole_jet, bias_train), axis=1)

        bias_test = np.ones((X_test_jet.shape[0], 1))
        X_test_jet = np.concatenate((X_test_jet, bias_test), axis=1)

        # Save cleaned data
        data_splits_train_whole[jet] = (X_train_whole_jet, y_train_whole_jet, ids_train_whole_jet) 
        data_splits_test[jet] = (X_test_jet, y_test_jet, ids_jet) 


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time: ", current_time)


######################################################################################################################

    # Training on the whole dataset (for the purpose of submission)

    if(params['verbose']):
        print("TRAINING ON THE WHOLE DATASET (FOR THE PURPOSE OF SUBMISSION)")

    weights = {}

    all_preds = []
    all_labels = []
    for jet in data_splits_train_whole:
        X_train_whole_jet, y_train_whole_jet, _ = data_splits_train_whole[jet]
        
        num_batches = max(1, int(X_train_whole_jet.shape[0] / params['batch_size']))
        w0 = np.zeros(X_train_whole_jet.shape[1])
        w, loss = None, None

        if(params['algorithm'] == 'reg_logistic'):
            w, loss = reg_logistic_regression(y=y_train_whole_jet, 
                                              tx=X_train_whole_jet, 
                                              lambda_=best_lambdas[jet], 
                                              initial_w=w0, 
                                              max_iters=params['max_iters'], 
                                              gamma=params['gamma'], 
                                              batch_size=params['batch_size'],
                                              num_batches=num_batches)

        if(params['algorithm'] == 'logistic'):
            w, loss = logistic_regression(y=y_train_whole_jet, 
                                          tx=X_train_whole_jet, 
                                          initial_w=w0, 
                                          max_iters=params['max_iters'], 
                                          gamma=params['gamma'], 
                                          batch_size=params['batch_size'],
                                          num_batches=num_batches)

        if(params['algorithm'] == 'least_squares_GD'):
            w, loss = least_squares_GD(y=y_train_whole_jet, 
                                       tx=X_train_whole_jet,  
                                       initial_w=w0, 
                                       max_iters=params['max_iters'], 
                                       gamma=params['gamma'])


        if(params['algorithm'] == 'least_squares_SGD'):
            w, loss = least_squares_SGD(y=y_train_whole_jet, 
                                        tx=X_train_whole_jet,  
                                        initial_w=w0, 
                                        max_iters=params['max_iters'], 
                                        gamma=params['gamma'])
        

        weights[jet] = w
                
        y_train_pred = predict_labels(w, X_train_whole_jet)
        acc_train = accuracy(y_train_pred, y_train_whole_jet)
        if(params['verbose']):
            print(f"JET {jet}: Train accuracy: {acc_train:.4f}")
            
        all_preds.append(y_train_pred)
        all_labels.append(y_train_whole_jet)

        

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc_train_whole = accuracy(all_preds, all_labels)
    if(params['verbose']):
        print(f"Train accuracy on the whole train data: {acc_train_whole:.4f}")

    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time: ", current_time)


######################################################################################################################

    # Making predictions on test data

    if(params['verbose']):
        print("MAKING PREDICTIONS ON TEST DATA")


    all_preds = []
    all_ids = []
    for jet in data_splits_test.keys():
        X_test_jet, _, ids_jet = data_splits_test[jet]
        y_test_pred = predict_labels(weights[jet], X_test_jet, competition=True)
        
        all_preds.append(y_test_pred)
        all_ids.append(ids_jet)

    all_preds = np.concatenate(all_preds, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    OUTPUT_PATH = f"./predictions/{params['output_file']}.csv"
    create_csv_submission(all_ids, all_preds, OUTPUT_PATH)






if __name__ == '__main__':
    main(**vars(parse_arguments()))