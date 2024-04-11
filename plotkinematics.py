import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skg.nsphere import nsphere_fit
import sys
sns.set_palette('viridis')


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

organised_hit_info_df = get_rings('example.txt')
pd.set_option('mode.use_inf_as_na', True)
train_df = organised_hit_info_df

train_df["phi"] = np.arctan2(train_df["py"],train_df["px"])

train_df["logp"] = np.log10(train_df["p"].values)

train_df = organised_hit_info_df.query("r>0 and r<10 and n_hits >= 5")
plot_arr = np.array([['p','phi','c_x','c_y'],
                     ['px','py','pz','pt'],
                     ['n_hits','track_x','track_y','track_z'],
                     ['r','p','logp','eta']])  

fig, axes = plt.subplots(4,4, figsize = (16,16))

for i in range(4):
    for j in range(4):
        axes[i,j].hist(train_df[plot_arr[i,j]], bins = 30)
        axes[i,j].set_title(plot_arr[i,j])
    
plt.show() 



train_df = train_df.loc[abs(train_df['pid']) == 211]  # get dem pions
#train_df = train_df.drop(columns=['pid'], axis=1)

####################### Calculate phi and log of p ######################



####################### cut out shit ########################
print(train_df.shape[0])
train_df = train_df.query(
    "track_x>-0.25 and track_x<0.25 and track_y>-0.25 and track_y<0.25")
print(train_df.shape[0])
train_df = train_df.query("logp>3.4 and logp<4.5")
print(train_df.shape[0])
train_df = train_df.query("track_z>-200 and track_z<200")
print(train_df.shape[0])
train_df['bigr'] = np.sqrt(train_df['c_x']**2 + train_df['c_y']**2 )
train_df['theta'] = np.arctan2(train_df['c_y'],train_df['c_x'])
df = train_df[[ 'logp','eta','phi','track_x', 'track_y', 
                           'n_hits', 'r', 'c_x', 'c_y', 'bigr', 'theta']]

m_df = train_df[['logp','eta','phi', 'r', 'bigr', 'theta']]
p_df = train_df[['logp', 'eta','phi', 'r', 'c_x','c_y']]
t_df = train_df[['track_x', 'track_y', 'c_x', 'c_y', 'bigr']]
print(train_df.keys())

"""
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
# Create a pairplot using Seaborn
sns.pairplot(m_df, dropna = True, kind='hist')
plt.show()
sns.pairplot(p_df, dropna = True, kind='hist')
plt.show()
sns.pairplot(t_df, dropna = True, kind='hist')
plt.show()
sns.pairplot(train_df.drop(['p', 'pt', 'px', 'py', 'pz', 'pid', 'eta', 'track_x', 'track_y',
       'track_z', 'n_hits', 'r', 'c_x', 'c_y', 'phi', 'logp', 'bigr',
       'theta']), dropna = True, kind='hist')
plt.show()
"""
lst_frac_diff = []
for index, row in train_df.iterrows():
    xs = row['x']
    ys = row['y']
    r = row['r']
    cx = row['c_x']
    cy = row['c_y']
    
    for i in range(len(xs)):
        rad = np.sqrt((xs[i]-cx)**2+(ys[i]-cy)**2)
        diff = rad-r
        lst_frac_diff.append(diff/r)
        
print(max(lst_frac_diff)) 
print(min(lst_frac_diff))

plt.hist(lst_frac_diff, bins = 100)
plt.show()

print(np.std(lst_frac_diff))
print(np.mean(lst_frac_diff))
print('=======================')
centers_diff = []
for index, row in train_df.iterrows():
    x = np.mean(row['x'])
    y = np.mean(row['y'])
    cx = row['c_x']
    cy = row['c_y']
    x_diff = x-cx
    y_diff = y-cy
    diff = np.sqrt((x_diff)**2+(y_diff)**2)
    centers_diff.append(diff)
print(max(centers_diff)) 
print(min(centers_diff))

plt.hist(centers_diff, bins = 100)
plt.show()

print(np.std(centers_diff)) 


    
fig, axes = plt.subplots(4,4, figsize = (16,16))

for i in range(4):
    for j in range(4):
        axes[i,j].hist(train_df[plot_arr[i,j]], bins = 30)
        axes[i,j].set_title(plot_arr[i,j])
    
plt.show() 
    
    
    
    
    
    
    
    
