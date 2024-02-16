import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from skg.nsphere import nsphere_fit
import pickle
from sklearn.preprocessing import MinMaxScaler
import random 

def get_rings(file, add_circle_fit_at_start=True):
    
    print("Getting some rings boiiiiss")
    
    hit_info = np.loadtxt(file, delimiter=',')
    # hit_info = np.loadtxt('rings.txt', delimiter=',')
    labels = np.unique(hit_info[:,0])


    if add_circle_fit_at_start:
        organised_hit_info = {'p':[], 'pt':[], 'px':[], 'py':[], 'pz':[], 'pid':[], 'eta':[], 'track_x':[], 'track_y':[], 'track_z':[], 'n_hits':[], 'x':[], 'y':[], 'r':[], 'c_x':[], 'c_y':[]}
    else:
        organised_hit_info = {'p':[], 'pt':[], 'px':[], 'py':[], 'pz':[], 'pid':[], 'eta':[], 'track_x':[], 'track_y':[], 'track_z':[], 'n_hits':[], 'x':[], 'y':[]}

    for label in labels:

        hit_info_i = hit_info[np.where(hit_info[:,0]==label)][:,1:]

        events = np.unique(hit_info_i[:,0])

        for event in events:

            hit_info_i_event = hit_info_i[np.where(hit_info_i[:,0]==event)][:,1:]

            hit_info_i_event_info = hit_info_i_event[np.where(hit_info_i_event[:,0]==1)][:,1:]
            hit_info_i_event_hits = hit_info_i_event[np.where(hit_info_i_event[:,0]==0)][:,1:]

            organised_hit_info['p'].append(hit_info_i_event_info[0][0])
            organised_hit_info['pt'].append(hit_info_i_event_info[1][0])
            organised_hit_info['px'].append(hit_info_i_event_info[2][0])
            organised_hit_info['py'].append(hit_info_i_event_info[3][0])
            organised_hit_info['pz'].append(hit_info_i_event_info[4][0])
            organised_hit_info['pid'].append(hit_info_i_event_info[5][0])
            organised_hit_info['eta'].append(hit_info_i_event_info[6][0])
            organised_hit_info['track_x'].append(hit_info_i_event_info[7][0])
            organised_hit_info['track_y'].append(hit_info_i_event_info[8][0])
            organised_hit_info['track_z'].append(hit_info_i_event_info[9][0])

            organised_hit_info['x'].append(hit_info_i_event_hits[:,0]/10.)
            organised_hit_info['y'].append(hit_info_i_event_hits[:,1]/10.)

            organised_hit_info['n_hits'].append(np.shape(hit_info_i_event_hits[:,1])[0])

            if add_circle_fit_at_start:
                ring_data = np.swapaxes(np.asarray([hit_info_i_event_hits[:,0]/10., hit_info_i_event_hits[:,1]/10.]),0,1)
                r, c = nsphere_fit(ring_data)
                organised_hit_info['r'].append(r)
                organised_hit_info['c_x'].append(c[0])
                organised_hit_info['c_y'].append(c[1])

    organised_hit_info_df = pd.DataFrame.from_dict(organised_hit_info)

    return organised_hit_info_df


def main():
    print("Im the BOSS")
    pd.set_option('display.max_columns', None)
    organised_hit_info_df = get_rings('example.txt')
    
    train_df = organised_hit_info_df.query("r>0 and r<10 and n_hits >= 5")
    train_df = train_df.loc[abs(train_df['pid']) == 211] # get dem pions
    train_df = train_df.drop(columns = ['pid'], axis = 1)
 
    ####################### Calculate phi and log of p ######################
    ratio = train_df["px"]/train_df["py"]
    train_df["phi"] = 2*np.arctan(ratio)/np.pi
    
    train_df["logp"] = np.log10(train_df["p"].values)

    
    ####################### cut out shit ########################
    print(train_df.shape[0])
    train_df = train_df.query("track_x>-0.25 and track_x<0.25 and track_y>-0.25 and track_y<0.25")
    print(train_df.shape[0])
    train_df = train_df.query("logp>3.4 and logp<4.5")
    print(train_df.shape[0])
    train_df = train_df.query("track_z>-200 and track_z<200")
    print(train_df.shape[0])
    
    
    ##################### Setup shit #######################
    conds_df = train_df[['p','eta','phi','track_x','track_y','track_z', 'logp', 'n_hits']]
    rnc_df = train_df[['r','c_x','c_y']]
    hits_df = train_df[['x','y']]
    
    rand5 = hits_df.apply(lambda x: random.sample(x, 5), axis = 1)
    sys.exit()
    
    print(rand5)
    ################# normalize that shit ##################
    scaler = MinMaxScaler(feature_range=(-1, 1))
    conds_df_normalized = pd.DataFrame(scaler.fit_transform(conds_df), columns=conds_df.columns)
    
    
    conds_df_normalized.reset_index(drop=True, inplace=True)
    rnc_df.reset_index(drop=True, inplace=True)
    rnc_data = pd.concat([conds_df_normalized, rnc_df], axis=1) #data for rnc GAN
    
    
    ############### plot some shit ##################
    """
    
    for col in conds_df.columns:
        col_max = conds_df[col].max()
        col_min = conds_df[col].min()
        plt.hist(conds_df[col], bins = 75)
        plt.title(f"Histogram for {col}  range: {col_min} >< {col_max}")
        plt.show()
    
    
    for col in conds_df_normalized.columns:
        
        plt.hist(conds_df_normalized[col], bins = 75)
        plt.title(f"Normalized histogram for {col}")
        plt.show()
        
    
    print("====================================")
    print(conds_df_normalized.shape[0])

    """
    #############################just check eta is right #########################
    """
    theta = np.arctan(((train_df['px']**2 + train_df['py']**2)**0.5)/train_df['pz'])
    
    psrap = -np.log(np.tan(theta/2))
    
    diff = np.asarray(train_df['eta']) - psrap
    
    plt.hist(diff, bins = 75)
    plt.title('eta-psrap')
    plt.show()
    
    plt.hist(psrap, bins = 75)
    plt.title('eta')
    plt.show()
    """
    ############################## bleddy is! ################## off by ~10E-5
    
    
    with open('pions_rnc.pkl', 'wb') as file:
        pickle.dump(rnc_data, file)


    ############################# right lets deal with hit points ###################
    
    max_len_x = max(len(lst) for lst in hits_df['x'])
    max_len_y = max(len(lst) for lst in hits_df['y'])
    
    #print(max_len_x, max_len_y)
    
    # Create new columns for 'x' and 'y'
    new_columns_x = [f'x{i}' for i in range(max_len_x)]
    new_columns_y = [f'y{i}' for i in range(max_len_y)]
    
    hits_df = pd.concat([
        hits_df.drop(['x', 'y'], axis=1),
        pd.DataFrame(hits_df['x'].to_list(), columns=new_columns_x),
        pd.DataFrame(hits_df['y'].to_list(), columns=new_columns_y)
    ], axis=1)
    
    #print(hits_df)

if __name__ == "__main__":
    main()

