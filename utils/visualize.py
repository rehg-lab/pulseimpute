import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def convert_task_to_path():
# Define the dictionaries for task and ptbxl paths
    task_dict = {
        'ECG': 'mimic_ecg_test',
        'PPG': 'mimic_ppg_test',
        'PTBXL': 'ptbxl_',
        'custom': 'custom_test'
    }
    ptbxl_dict1 = {
        'Extended': 'testextended_',
        'Transient': 'testtransient_',
        'Extracted mHealth Missingness Patterns': '',
        '---': ''
    }
    ptbxl_dict2 = {
        '10%': '10percent',
        '20%': '20percent',
        '30%': '30percent',
        '40%': '40percent',
        '50%': '50percent',
        '---': ''
    }
    return task_dict, ptbxl_dict1, ptbxl_dict2

def visualize(task, ptbxl1, ptbxl2, models, sample_index, x_range, save_path=None):
    task_dict, ptbxl_dict1, ptbxl_dict2 = convert_task_to_path()
    # Build the file path using the task, ptbxl1, and ptbxl2 information
    base_path = os.path.join('out/out_test', task_dict[task] + ptbxl_dict1[ptbxl1] + ptbxl_dict2[ptbxl2])
    
    # Load the data from numpy files
    #print(base_path)
    print(os.path.join(base_path, models[0], 'original.npy'))
    original = np.load(os.path.join(base_path, models[0], 'original.npy'))
    target_seq = np.load(os.path.join(base_path, models[0], 'target_seq.npy'))

    # have imputation only be for target region
    imputation_dict = {}
    for model in models:
        imputation = np.load(os.path.join(base_path, model, 'imputation.npy'))
        imputation[np.isnan(target_seq)] = np.nan
        imputation_dict[model] = imputation


    x_range = (0, min(len(original[sample_index]), x_range))
    # Start plotting
    fig = make_subplots(rows=1, cols=1)

    for model in models:
        fig.add_trace(go.Scatter(
            x=np.arange(len(imputation_dict[model][sample_index])),
            y=imputation_dict[model][sample_index].flatten(),
            mode='lines',
            name=model,
            line=dict(color=np.random.choice(['blue', 'green', 'red', 'purple', 'orange']), width=3)
        ))
        
    fig.add_trace(go.Scatter(
        x=np.arange(len(original[sample_index])),
        y=original[sample_index].flatten(),
        mode='lines',
        name='Original',
        line=dict(color='black')
    ))

    fig.add_trace(go.Scatter(
        x=np.arange(len(target_seq[sample_index])),
        y=target_seq[sample_index].flatten(),
        mode='lines',
        name='Target Sequence',
        line=dict(color='lightgray')
    ))

    # Update axes and layout
    fig.update_xaxes(range=x_range, showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        title="Data Visualization",
        xaxis_title="Data Points",
        yaxis_title="Values",
        height=600,
        width=800
    )
    
    # Save to file
    if save_path is None:
        save_path = base_path
    fig.write_image(os.path.join(save_path, 'plot.png'))
    print('Saved to: ' + str(os.path.join(save_path, 'plot.png')))


visualize(task='custom', ptbxl1='---', ptbxl2='---', models=['FFT_custom'],
                                                                 sample_index=0, x_range=30000, save_path='')
#visualize(task='ECG', ptbxl1='---', ptbxl2='---', models=['mean'],
#                                                                 sample_index=0, x_range=30000, save_path='')