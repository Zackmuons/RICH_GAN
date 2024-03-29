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
import torch
import seaborn as sns


def get_rings(file, add_circle_fit_at_start=True):

    print("Getting some rings boiiiiss")

    hit_info = np.loadtxt(file, delimiter=',')
    # hit_info = np.loadtxt('rings.txt', delimiter=',')
    labels = np.unique(hit_info[:, 0])

    if add_circle_fit_at_start:
        organised_hit_info = {'p': [], 'pt': [], 'px': [], 'py': [], 'pz': [], 'pid': [], 'eta': [], 'track_x': [
        ], 'track_y': [], 'track_z': [], 'n_hits': [], 'x': [], 'y': [], 'r': [], 'c_x': [], 'c_y': []}
    else:
        organised_hit_info = {'p': [], 'pt': [], 'px': [], 'py': [], 'pz': [], 'pid': [], 'eta': [
        ], 'track_x': [], 'track_y': [], 'track_z': [], 'n_hits': [], 'x': [], 'y': []}

    for label in labels:

        hit_info_i = hit_info[np.where(hit_info[:, 0] == label)][:, 1:]

        events = np.unique(hit_info_i[:, 0])

        for event in events:

            hit_info_i_event = hit_info_i[np.where(
                hit_info_i[:, 0] == event)][:, 1:]

            hit_info_i_event_info = hit_info_i_event[np.where(
                hit_info_i_event[:, 0] == 1)][:, 1:]
            hit_info_i_event_hits = hit_info_i_event[np.where(
                hit_info_i_event[:, 0] == 0)][:, 1:]

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

            organised_hit_info['x'].append(hit_info_i_event_hits[:, 0]/10.)
            organised_hit_info['y'].append(hit_info_i_event_hits[:, 1]/10.)

            organised_hit_info['n_hits'].append(
                np.shape(hit_info_i_event_hits[:, 1])[0])

            if add_circle_fit_at_start:
                ring_data = np.swapaxes(np.asarray(
                    [hit_info_i_event_hits[:, 0]/10., hit_info_i_event_hits[:, 1]/10.]), 0, 1)
                r, c = nsphere_fit(ring_data)
                organised_hit_info['r'].append(r)
                organised_hit_info['c_x'].append(c[0])
                organised_hit_info['c_y'].append(c[1])

    organised_hit_info_df = pd.DataFrame.from_dict(organised_hit_info)

    return organised_hit_info_df


def rand5indices(row):

    x_values = row['x']
    y_values = row['y']

    min_length = min(len(x_values), len(y_values))

    random_indices = random.sample(range(min_length), 5)
    row['x'] = [x_values[i] for i in random_indices]
    row['y'] = [y_values[i] for i in random_indices]

    return row


def main():
    print("Im the BOSS")
    pd.set_option('display.max_columns', None)
    organised_hit_info_df = get_rings('example.txt')
    
    organised_hit_info_df['n_hits'].plot(kind = 'hist')
    plt.show()

    train_df = organised_hit_info_df.query("r>0 and r<10 and n_hits >= 5")
    train_df = train_df.loc[abs(train_df['pid']) == 211]  # get dem pions
    #train_df = train_df.drop(columns=['pid'], axis=1)

    ####################### Calculate phi and log of p ######################
    
    train_df["phi"] = np.arctan2(train_df["py"],train_df["px"])

    train_df["logp"] = np.log10(train_df["p"].values)
    
    ####################### cut out shit ########################
    print(train_df.shape[0])
    train_df = train_df.query(
        "track_x>-0.25 and track_x<0.25 and track_y>-0.25 and track_y<0.25")
    print(train_df.shape[0])
    train_df = train_df.query("logp>3.4 and logp<4.5")
    print(train_df.shape[0])
    train_df = train_df.query("track_z>-200 and track_z<200")
    print(train_df.shape[0])

    correlation_df = train_df[['p', 'logp','eta','phi','pt',
                               'px', 'py', 'pz',
                               'track_x', 'track_y', 'track_z', 
                               'n_hits', 'r', 'c_x', 'c_y']]
    ##################### Setup shit #######################
    conds_df = train_df[['eta', 'phi', 'track_x',
                         'track_y', 'track_z', 'logp']]
    #rnc_df = train_df[['r', 'c_x', 'c_y']]
    hits_df = train_df[['x', 'y']]
    all_hits = train_df[['x','y', 'n_hits']]
    print(all_hits['n_hits'].min())
    all_hits.reset_index(drop=True,inplace=True)
    df = train_df
    
    fig , axes = plt.subplots(2,2, figsize=(8,8))
    axes[0,0].plot(df['phi'],df['c_y'],'.')
    axes[0,1].plot(df['phi'],df['c_x'],'.')
    axes[1,0].plot(df['eta'],df['c_y'],'.')
    axes[1,1].plot(df['eta'],df['c_x'],'.')
    plt.show()
   
    
    
    
    ################# normalize that shit ##################
    centered_hits = train_df[['x', 'y', 'c_x', 'c_y', 'r', 'n_hits']]
    centered_hits.reset_index(drop=True, inplace=True)
    centered_hits['x'] = centered_hits.apply(lambda row: row['x'] -  row['c_x'] , axis = 1)
    centered_hits['y'] = centered_hits.apply(lambda row: row['y'] -  row['c_y'] , axis = 1)
    
    print(centered_hits['n_hits'].sum())
    
    
    
    
        
    def explode_rows(row):
        x_values = row['x']
        y_values = row['y']
        r = row['r']
        c_x = row['c_x']
        c_y = row['c_y']
        n_hits = row['n_hits']
        
        new_rows = []
        for i in range(len(x_values)):
            new_rows.append({'x': x_values[i], 'y': y_values[i], 'r': r, 'c_x' : c_x, 'c_y' : c_y, 'n_hits' : n_hits, 'index': 2*((i)/n_hits)-1})
        return new_rows

    # Apply the function to each row and concatenate the results
    new_rows = []
    for _, row in centered_hits.iterrows():
        new_rows.extend(explode_rows(row))
    
    # Create a new DataFrame with exploded rows
    c_df = pd.DataFrame(new_rows)
    c_df.reset_index(drop=True, inplace=True)


    max_hit = max(c_df['x'].max(), c_df['y'].max())
    max_center = max(c_df['c_x'].max(), c_df['c_y'].max())
    c_df[['x','y']] = c_df[['x', 'y']]/max_hit
    c_df[['c_x','c_y']] = c_df[['c_x', 'c_y']]/max_center
    radius_scalar = MinMaxScaler(feature_range=(-1,1))
    c_df['r'] = radius_scalar.fit_transform(c_df['r'].values.reshape(-1,1))
    n_hits_scalar = MinMaxScaler(feature_range=(-1,1))
    c_df['n_hits'] = n_hits_scalar.fit_transform(c_df['n_hits'].values.reshape(-1,1))
    print(c_df)
    
    center_tensor = torch.Tensor(c_df.values)
    torch.save(center_tensor, "centered_hits.pt")
    
    c_df['residual'] = np.sqrt(c_df['x']**2 + c_df['y']**2)-c_df['r']
    
    fig, axes = plt.subplots(1, figsize= (8,8))
    axes.hist(c_df['residual'], bins = 50)
    plt.show()
    
    centered_scalar = {"radius": radius_scalar,
                       "max_hit": max_hit,
                       "max_center": max_center,
                       "n_hits": n_hits_scalar}
    
    with open('centered_scalar.pkl', 'wb') as file:
        pickle.dump(centered_scalar, file)
        
    sys.exit()
    
        
    
    max_len_x = max(len(lst) for lst in centered_hits['x'])
    max_len_y = max(len(lst) for lst in centered_hits['y'])
    max_val_x = max(max(abs(lst)) for lst in centered_hits['x'])
    max_val_y = max(max(abs(lst)) for lst in centered_hits['y'])
    max_val = max(max_val_x, max_val_y)
    
    
    
    # Create new columns for 'x' and 'y'
    new_columns_x = [f'x{i}' for i in range(max_len_x)]
    new_columns_y = [f'y{i}' for i in range(max_len_y)]
    
    x_columns = pd.DataFrame(centered_hits['x'].to_list(), columns=new_columns_x)
    y_columns = pd.DataFrame(centered_hits['y'].to_list(), columns=new_columns_y)

    centered_hits = pd.concat([ x_columns, y_columns, centered_hits.drop(['x', 'y'], axis=1)], axis=1)
    centered_hits.fillna(0, inplace= True)
    print(centered_hits)
    sys.exit()
    
    
    """
    centered_hits['max'] = centered_hits.apply(lambda row: min(np.sqrt(row['x']**2 + row['y']**2)) , axis = 1) 
    centered_hits['frac'] = centered_hits['max']/centered_hits['r']
    
    fig,axes = plt.subplots(1, figsize=(8,8))
    for i in range(6036):
        axes.plot(centered_hits['x'].values[i],centered_hits['y'].values[i], '.')
    plt.show()
    sys.exit()
    """
    
    conds_scaler = MinMaxScaler(feature_range=(-1, 1))
    conds_df_normalized = pd.DataFrame(
        conds_scaler.fit_transform(conds_df), columns=conds_df.columns)
    
    rnc_df_normalized = pd.DataFrame(columns = ['r', 'cx', 'cy','n_hits'])
    r_df = train_df['r']
    c_df = train_df[['c_x','c_y']]
    n_df = train_df['n_hits']
    
    r_scaler = MinMaxScaler(feature_range=(-1, 1))
    r_df_normalized = r_scaler.fit_transform(r_df.values.reshape(-1, 1))
    c_df.reset_index(drop = True, inplace = True)
    
    n_scaler = MinMaxScaler(feature_range=(-1,1))
    n_df_normalized = n_scaler.fit_transform(n_df.values.reshape(-1,1))
    
    # Update 'r' column in rnc_df_normalized
    rnc_df_normalized['r'] = r_df_normalized.flatten()
    max_center = np.abs(c_df).max().max()
    
    c_df = c_df/max_center
    rnc_df_normalized['r'] = r_df_normalized
    rnc_df_normalized[['cx','cy']] = c_df
    rnc_df_normalized['n_hits'] = n_df_normalized
    
    print(rnc_df_normalized)

    
    #print(train_df[['r','c_x','c_y']])
    #print(rnc_df_normalized)

    
    conds_df_normalized.reset_index(drop=True, inplace=True)
    rnc_df_normalized.reset_index(drop=True, inplace=True)
    
    
    #rand5 = pd.DataFrame(columns = ['x','y'])
    rand5 = hits_df.apply(rand5indices, axis=1)
    
    
    new_columns_x = ['x1', 'x2', 'x3', 'x4', 'x5']
    new_columns_y = ['y1', 'y2', 'y3', 'y4', 'y5']

    rand5 = pd.concat([
        pd.DataFrame(rand5['x'].to_list(), columns=new_columns_x),
        pd.DataFrame(rand5['y'].to_list(), columns=new_columns_y)
    ], axis=1)

    
    rand5_10 = rand5
    
    
    
    for i in range(10):
        new5 = hits_df.apply(rand5indices, axis=1)
        new5 = pd.concat([
            pd.DataFrame(new5['x'].to_list(), columns=new_columns_x),
            pd.DataFrame(new5['y'].to_list(), columns=new_columns_y)
        ], axis=1)
        
        rand5_10 = pd.concat([rand5_10,new5], axis = 0, ignore_index=True)
    
    
    
    max_len_x = max(len(lst) for lst in all_hits['x'])
    max_len_y = max(len(lst) for lst in all_hits['y'])
    max_val_x = max(max(abs(lst)) for lst in train_df['x'])
    max_val_y = max(max(abs(lst)) for lst in train_df['y'])
    max_val = max(max_val_x, max_val_y)
    
    
    
    # Create new columns for 'x' and 'y'
    new_columns_x = [f'x{i}' for i in range(max_len_x)]
    new_columns_y = [f'y{i}' for i in range(max_len_y)]
    
    x_columns = pd.DataFrame(all_hits['x'].to_list(), columns=new_columns_x)
    y_columns = pd.DataFrame(all_hits['y'].to_list(), columns=new_columns_y)

    all_hits = pd.concat([ x_columns, y_columns, all_hits.drop(['x', 'y'], axis=1)], axis=1)
    all_hits.fillna(0, inplace= True)
    print(all_hits)
    
    print(all_hits['n_hits'].min())
    all_hits = all_hits/max_val
    all_hits['n_hits'] = all_hits['n_hits']*max_val
    #all_hits['n_hits'] = all_hits.apply(lambda row: row[:88].count(), axis = 1)
    #all_hits.iloc[:,:] = all_hits.iloc[:,:].fillna(0)
    #print(all_hits)
    
    all_hits['n_hits'].plot(kind = 'hist')
    plt.show()
    
    
    scaling_dict = {"conds": conds_scaler, 
                    "radius": r_scaler, 
                    "hits": max_val,
                    "centers": max_center,
                    "n_scaler": n_scaler}
    
    with open('scaling_pions.pkl', 'wb') as file:
        pickle.dump(scaling_dict, file)
        
    
    
    rand5_normalized = rand5/max_val
    rand5_10_normalized = rand5_10/max_val
    print(rand5_normalized.min().min())
    print(rand5_normalized.max().max())
    rand5_normalized.reset_index(drop=True, inplace=True)
    
    ############### plot some shit ##################
    
    fig, axes = plt.subplots(2,2, figsize = (8,6))
    
    axes[0,0].plot(train_df["phi"] ,train_df['c_x']  ,'.')
    axes[0,0].set_title("phi c_x")
    axes[0,1].plot(train_df["phi"], train_df['c_y'], '.')
    axes[0,1].set_title("phi c_y")
    axes[1,0].plot(train_df['py'], train_df['c_x'], '.')
    axes[1,0].set_title('eta c_x')
    axes[1,1].plot(train_df['py'], train_df['c_y'], '.')
    axes[1,1].set_title('eta c_y')
    plt.show()
    
    fig, axes = plt.subplots(1, figsize= (8,6))
    axes.plot(x_columns, y_columns, '.')
    plt.show()
    
    """
    x_s = train_df['c_x'].values
    y_s = train_df['c_y'].values
    plt.plot(x_s,y_s,'.')
    plt.show()
    
    
    plt.plot
    
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
    
    sns.set_theme(style="white")


    xs = train_df['x'].values
    ys = train_df['y'].values
    

    plt.subplots(1, figsize =(8,6))
    plt.plot(train_df["p"].values,train_df["n_hits"].values,'.')
    
    plt.show()
    sys.exit()


    # Compute the correlation matrices
    corr = correlation_df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(1, figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    
    # bleddy is! ################## off by ~10E-5

    """
    with open('pions_rnc.pkl', 'wb') as file:
        pickle.dump(rnc_data, file)
    """

    ############################# right lets deal with hit points ###################
    """
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
    """
    
    ######################### make the data files #####################################
    
    
    
    
    
    rnc_data = pd.concat([conds_df_normalized, rnc_df_normalized],
                         axis=1)  # data for rnc GAN
    conds_rand5 = pd.concat([conds_df_normalized, rand5_normalized], 
                           axis = 1) # data for conds gan
    
    rand5_10_tensor = torch.Tensor(rand5_10_normalized.values)
    rnc_data = torch.Tensor(rnc_data.values)
    conds_rand5 = torch.Tensor(conds_rand5.values)
    rand5_normalized = torch.Tensor(rand5_normalized.values)
    all_hits_tensor = torch.Tensor(all_hits.values)
    
    print(all_hits_tensor)
    torch.save(rand5_10_tensor, "rand5_10.pt")
    torch.save(all_hits_tensor, "all_hits.pt")
    torch.save(rnc_data, "rnc_data.pt")
    torch.save(conds_rand5,"conds_rand5.pt")
    torch.save(rand5_normalized,"rand5_normalized.pt")
    
    print("pickle files made")
    


if __name__ == "__main__":
    main()
