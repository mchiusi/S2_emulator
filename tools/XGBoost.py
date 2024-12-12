import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import json
from sklearn.model_selection import train_test_split

def json_to_dataframe(data):
    rows = []
    
    for particle, variables in data.items():
        transposed = {key: list(values) for key, values in variables.items()}
        df_particle = pd.DataFrame(transposed)
        df_particle["particle"] = particle     
        rows.append(df_particle)

    full_df = pd.concat(rows, ignore_index=True)
    return full_df

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return {}

def load_data():
    filename = 'plots/data/shape_variables.json'
    particle_data = load_json(filename)
    return json_to_dataframe(particle_data)

def train(data):
    feature_names = data.columns[:-1]  # we skip the last column because is particle = label
    train = xgb.DMatrix(data=data_train[feature_names],label=data_train.particle.cat.codes,
                    missing=-999.0,feature_names=feature_names)
    test = xgb.DMatrix(data=data_test[feature_names],label=data_test.particle.cat.codes,
                    missing=-999.0,feature_names=feature_names)
    
    param = {}

    # Booster parameters
    param['eta']              = 0.1 # learning rate
    param['max_depth']        = 6  # maximum depth of a tree
    param['subsample']        = 0.8 # fraction of events to train tree on
    param['colsample_bytree'] = 0.8 # fraction of features to train tree on

    # Learning task parameters
    param['objective']   = 'binary:logistic' # objective function
    param['eval_metric'] = 'error'           # evaluation metric for cross validation
    param = list(param.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]

    num_trees = 100  # number of trees to make
    return xgb.train(param,train,num_boost_round=num_trees), test

def plot_taining_predictions(predictions, test):
    # plot all predictions (both signal and background)
    plt.figure()
    plt.hist(predictions,bins=np.linspace(0,1,30),histtype='step',color='darkgreen',label='All events')
    plt.xlabel('Prediction from BDT',fontsize=12)
    plt.ylabel('Events',fontsize=12)
    plt.legend(frameon=False)
    plt.savefig('plots/LLPs_all_event.pdf')
    plt.savefig('plots/LLPs_all_event.png')
    plt.clf()

    # plot signal and background separately
    plt.figure()
    plt.hist(predictions[test.get_label().astype(bool)],bins=np.linspace(0,1,30),
            histtype='step',color='midnightblue',label='signal')
    plt.hist(predictions[~(test.get_label().astype(bool))],bins=np.linspace(0,1,30),
            histtype='step',color='firebrick',label='background')
    # make the plot readable
    plt.xlabel('Prediction from BDT',fontsize=12)
    plt.ylabel('Events',fontsize=12)
    plt.legend(frameon=False)
    plt.savefig('plots/LLPs_signal_bkg.pdf')
    plt.savefig('plots/LLPs_signal_bkg.png')
    plt.clf()
    
def compute_ROC(predictions):
    # choose score cuts:
    cuts = np.linspace(0,1,10)
    nsignal = np.zeros(len(cuts))
    nbackground = np.zeros(len(cuts))
    for i,cut in enumerate(cuts):
        nsignal[i] = len(np.where(predictions[test.get_label().astype(bool)] > cut)[0])
        nbackground[i] = len(np.where(predictions[~(test.get_label().astype(bool))] > cut)[0])

    # plot efficiency vs. purity (ROC curve)
    plt.figure()
    plt.plot(nsignal/len(data_test[data_test.particle == 'LLPs']), nsignal/(nsignal + nbackground), \
             'o-', color='blueviolet')
    plt.xlabel('Efficiency',fontsize=12)
    plt.ylabel('Purity',fontsize=12)
    plt.legend(frameon=False)
    plt.savefig('plots/LLPs_ROC_curve.pdf')
    plt.savefig('plots/LLPs_ROC_curve.png')
    plt.clf()
    

if __name__ == '__main__':
    ''' python tools/XGBoost.py following the tutorial at 
        https://github.com/k-woodruff/bdt-tutorial/blob/master/bdt_tutorial.ipynb '''

    data = load_data()

    print('Number of signal events: {}'.format(len(data[data.particle == 'LLPs'])))
    print('Number of background events: {}'.format(len(data[data.particle == 'photons'])))
    print('Fraction signal: {}'.format(len(data[data.particle == 'LLPs'])/(float)(len(data[data.particle == 'LLPs']) \
                                     + len(data[data.particle == 'photons']))))

    # splitting testing and training samples 
    data['particle'] = data.particle.astype('category')
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

    print('Number of training samples: {}'.format(len(data_train)))
    print('Number of testing samples: {}'.format(len(data_test)))

    print('\nNumber of signal events in training set: {}'.format(len(data_train[data_train.particle == 'LLPs'])))
    print('Number of background events in training set: {}'.format(len(data_train[data_train.particle == 'photons'])))
    print('Fraction signal: {}'.format(len(data_train[data_train.particle == 'LLPs'])/(float)(len(data_train[data_train.particle == 'LLPs']) \
                                     + len(data_train[data_train.particle == 'photons']))))

    # training
    booster, test = train(data_train)
    predictions = booster.predict(test)
    plot_taining_predictions(predictions, test)

    # ROC curve
    compute_ROC(predictions)
    xgb.plot_importance(booster,grid=False)
    plt.savefig("plots/feature_importance.png", dpi=300, bbox_inches='tight')
