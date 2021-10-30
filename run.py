import os
os.sys.path.append('./scripts')
os.sys.path.append('./model')
os.sys.path.append('./dataprocess')
os.sys.path.append('./visualization')

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from implementations import *

from process import *
from cross_validation import *

from proj1_helpers import *
from argument_parser import *



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


    feature_list = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_deltaeta_jet_jet", "DER_mass_jet_jet",
                    "DER_prodeta_jet_jet", "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau", "DER_met_phi_centrality",
                    "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi", "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi",
                    "PRI_met", "PRI_met_phi", "PRI_met_sumet", "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta",
                    "PRI_jet_leading_phi", "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "PRI_jet_all_pt"]
    feature_ids = {feature:i for i, feature in enumerate(feature_list)}


    # TODO: REMOVING FEATURES OF TRAINING DATA FOR CROSS VALIDATIO
    #       - BE CAREFUL THAT AFTER SPLITTING ON JET, INDICES OF FEATURES CHANGE



    # Without splitting on JET
    data_splits_train = {-1:(X_train, y_train, ids_train)}
    data_splits_train_whole = {-1:(X_train_whole, y_train_whole, ids_train_whole)}
    data_splits_test = {-1:(X_test, y_test, ids_test)}

    # With splitting on JET
    if(params['split_jet']):
        data_splits_train = split_data_by_feature(y_train, X_train, ids_train, feature_ids["PRI_jet_num"], train=True)
        data_splits_train_whole = split_data_by_feature(y_train_whole, X_train_whole, ids_train_whole, feature_ids["PRI_jet_num"], train=True)
        data_splits_test = split_data_by_feature(None, X_test, ids_test, feature_ids["PRI_jet_num"], train=False)



    best_degrees = {}
    best_lambdas = {}

    # Cross validation
    for jet in data_splits_train:
        X_train_jet, y_train_jet, _ = data_splits_train[jet]
        
        if(params['verbose']):
            print(f"JET NUMBER {int(jet)}:")     

        best_accuracy, best_degree, best_lambda = k_fold_cross_validation(y_train_jet, X_train_jet, params['k-fold'],
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




######################################################################################################################

    # Data preprocessing for whole training dataset and testing dataset

    for jet in data_splits_train_whole:
        X_train_whole_jet, y_train_whole_jet, ids_train_whole_jet = data_splits_train_whole[jet]
        X_test_jet, y_test_jet, ids_jet = data_splits_test[jet]



        # TODO: REMOVING FEATURES - BE CAREFUL THAT AFTER SPLITTING ON JET, INDICES OF FEATURES CHANGE


        # Remove degenerated features
        X_train_whole_jet, removed_features_jet = remove_tiny_features(X_train_whole_jet)
        X_test_jet = remove_custom_features(X_test_jet, custom_feature_ids=removed_features_jet)
            


        # Imputing median
        if(params['impute_median']):
            # TODO !!!!!
            """
            X_train_whole_jet, median = impute_median(X_train_whole_jet, None)
            X_test_jet, _, _ = impute_median(X_test_jet, median)
            """

        # Remove outliers
        if(params['remove_outliers']):
            # TODO !!!!!
            pass

        # Feature expansion
        if(params['feature_expansion']):
            X_train_whole_jet = build_poly(X_train_whole_jet, best_degrees[jet])
            X_test_jet = build_poly(X_test_jet, best_degrees[jet])

        # Standardize features
        X_train_whole_jet, mean, std = standardize(X_train_whole_jet, None, None)
        X_test_jet, _, _ = standardize(X_test_jet, mean, std)

        # Save cleaned data
        data_splits_train_whole[jet] = (X_train_whole_jet, y_train_whole_jet, ids_train_whole_jet) 
        data_splits_test[jet] = (X_test_jet, y_test_jet, ids_jet) 


######################################################################################################################

    # Training on the whole dataset

    weights = {}

    all_preds = []
    all_labels = []
    for jet in data_splits_train_whole.keys():
        X_train_whole_jet, y_train_whole_jet, _ = data_splits_train_whole[jet]
                
        
        w0 = np.zeros(X_train.shape[1])
        w, loss = None, None

        if(algorithm == 'reg_logistic'):
            w, loss = reg_logistic_regression(y=y_train_whole_jet, 
                                              tx=X_train_whole_jet, 
                                              lambda_=lambda_, 
                                              initial_w=w0, 
                                              max_iters=max_iters, 
                                              gamma=gamma, 
                                              batch_size=batch_size)

        if(algorithm == 'logistic'):
            w, loss = logistic_regression(y=y_train_whole_jet, 
                                          tx=X_train_whole_jet, 
                                          initial_w=w0, 
                                          max_iters=max_iters, 
                                          gamma=gamma, 
                                          batch_size=batch_size)

        if(algorithm == 'least_squares_GD'):
            w, loss = least_squares_GD(y=y_train_whole_jet, 
                                       tx=X_train_whole_jet,  
                                       initial_w=w0, 
                                       max_iters=max_iters, 
                                       gamma=gamma)


        if(algorithm == 'least_squares_SGD'):
            w, loss = least_squares_SGD(y=y_train_whole_jet, 
                                        tx=X_train_whole_jet,  
                                        initial_w=w0, 
                                        max_iters=max_iters, 
                                        gamma=gamma)
        

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
        print(f"TOGETHER: Train accuracy on train data: {acc_train_whole:.4f}")



######################################################################################################################

    # Testing and making predictions

    all_preds = []
    all_ids = []
    for jet in data_splits_test.keys():
        X_test_jet, _, ids_jet = data_splits_test[jet]
        y_test_pred = predict_labels(weights[jet], X_test_jet, competition=True)
        
        all_preds.append(y_test_pred)
        all_ids.append(ids_jet)

    all_preds = np.concatenate(all_preds, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    OUTPUT_PATH = f'./predictions/{params['output_file']}.csv'
    create_csv_submission(all_ids, all_preds, OUTPUT_PATH)


if __name__ == '__main__':
    main(**vars(parse_arguments()))