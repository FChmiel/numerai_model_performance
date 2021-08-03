"""
Code for plotting an embedding of models submitted to the numerai
tournament.


Copyright (C) 2021  Francis P Chmiel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import requests
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap # pip install umap-learn

# You can change what rounds to look at here
LAST_ROUND = 245 # Last round to include in creating embedded space
FIRST_ROUND = 221

# coords are position to plot text relative to model position
KNOWN_MODELS = {'budbot_7':(-2,-2),
                'integration_test_7':(2,0.5),
                'krat':(-0.25,-2.5),
                'trivial':(-2,-2)
               }

get_data() # this is defined below

df = pd.read_csv('round_data.csv')
pivoted_df = pd.pivot(df, 
                      index='model', 
                      columns='round', 
                      values=['corr','mmc'])

round_corrs = pivoted_df['corr'] # model per row, round per column
round_mmcs = pivoted_df['mmc']

# calculate the embedding in end-of-round correlation space
X = round_corrs.loc[:,FIRST_ROUND:LAST_ROUND]

# UMAP doen't like NaNs, either impute or remove rows models with NaN
nan_mask = X.isna().sum(axis=1)==0
X = X[nan_mask]

embedder = umap.UMAP(random_state=42, n_neighbors=30)
X_emb = embedder.fit_transform(X.to_numpy())
mean_corr_by_model = np.mean(X.to_numpy(), axis=1)

fig, ax = plt.subplots()
cax = ax.scatter(X_emb[:,0], 
                 X_emb[:,1], 
                 c=mean_corr_by_model,
                 cmap='RdBu',
                 s=18,
                 vmin=-0.03,
                 vmax=0.03)

cb = fig.colorbar(cax, 
                  ax=ax, 
                  label='Mean end-of-round correlation',
                  fraction=0.03)
cb.ax.tick_params(labelsize=8)

# Add annotation of known model names
for key, value in KNOWN_MODELS.items():
        try:
            annotate_model(ax, X, X_emb, key, value)
        except:
            pass
        
ax.axis('off')

def annotate_model(ax, X, X_emb, model_name, xytext=(-2,-2)):
    """
    Adds model name annotation to ax.

    Parameters:
    -----------
        ax : matplotlib.Axes, 
        Axes to plot to.
    
    X : pd.DataFrame,
        The round correlations, with model names as index.
    
    X_emb: np.array, 
        The 2D embedding.
        
    model_name : str, 
        Name of the model to annotate
        
    xytext : tuple, 
        Coords relative to point to plot.
    """
    mask = X.index==model_name
    coords = X_emb[mask][0]
    
    ax.annotate(model_name,
                xy=coords, 
                xycoords='data',
                xytext=(coords[0]+xytext[0], coords[1]+xytext[1]),
                textcoords='data',
                size=9,
                va="center",
                ha="center",
                arrowprops=dict(arrowstyle="simple",
                                connectionstyle="arc3,rad=0.2",
                                color='k'))
    
def get_data(data_url=None):
    """
    Checks if round_data is available and if not loads the data from 
    "ia_ai_Joe's" webpage (https://www.jofaichow.co.uk/numerati/).
    
    Parameters:
    -----------
    data_url : str (default=None),
        URL where the round data is held. If None default is used.
    """
    if data_url is None:
        data_url = ('https://raw.githubusercontent.com'
                           '/woobe/numerati/master/data.csv')
        
    if not os.path.isfile('round_data.csv'):
        # get data from ia_ai_Joe's webpage
        resp = requests.get(data_url)
        with open('round_data.csv', 'w') as f:
            writer = csv.writer(f)
            for line in resp.iter_lines():
                writer.writerow(line.decode('utf-8').split(','))
    else: 
        print('Data already downloaded.')
