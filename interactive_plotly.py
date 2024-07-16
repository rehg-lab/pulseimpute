import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import collections

def hcl_to_rgb(hue=0, chroma=0, luma=0) :
    # Notes:
    #   coded from http://en.wikipedia.org/wiki/HSL_and_HSV#From_luma.2Fchroma.2Fhue
    #   with insights from gem.c in MagickCore 6.7.8
    #   http://www.imagemagick.org/api/MagickCore/gem_8c_source.html
    # Assume:
    #   h, c, l all in range 0 .. 1 (cylindrical coordinates)
    # Returns a tuple:
    #   r, g, b all in the range 0 .. 1 (cubic cartesian coordinates)

    # sanity checks
    hue = math.modf(float(hue))[0]
    if hue < 0 or hue >= 1 :
        raise ValueError('hue is a value greater than or equal to 0 and less than 1')
    chroma = float(chroma)
    if chroma < 0 or chroma > 1 :
        raise ValueError('chroma is a value between 0 and 1')
    luma = float(luma)
    if luma < 0 or luma > 1 :
        raise ValueError('luma is a value between 0 and 1')

    # do the conversion
    _h = hue * 6.0
    x = chroma * ( 1 - abs((_h % 2) - 1) )

    c = chroma
    if   0 <= _h and _h < 1 :
        r, g, b = (c, x, 0.0)
    elif 1 <= _h and _h < 2 :
        r, g, b = (x, c, 0.0)
    elif 2 <= _h and _h < 3 :
        r, g, b = (0.0, c, x)
    elif 3 <= _h and _h < 4 :
        r, g, b = (0.0, x, c)
    elif 4 <= _h and _h < 5 :
        r, g, b = (x, 0.0, c)
    elif 5 <= _h and _h <= 6 :
        r, g, b = (c, 0.0, x)
    else :
        r, g, b = (0.0, 0.0, 0.0)

    m = luma - (0.298839*r + 0.586811*g + 0.114350*b)
    z = 1.0
    if m < 0.0 :
        z = luma/(luma-m)
        m = 0.0
    elif m + c > 1.0 :
        z = (1.0-luma)/(m+c-luma)
        m = 1.0 - z * c
    (r, g, b) = (z*r+m, z*g+m, z*b+m)

    # clipping ...
    (r, g, b) = (min(r, 1.0), min(g, 1.0), min(b, 1.0))
    (r, g, b) = (max(r, 0.0), max(g, 0.0), max(b, 0.0))
    return (r, g, b)

def ggColorSlice(n=12, hue=(0.004,1.00399), chroma=0.8, luma=0.6, skipHue=True) :
    # Assume:
    #   n: integer >= 1
    #   hue[from, to]: all floats - red = 0; green = 0.33333 (or -0.66667) ; blue = 0.66667 (or -0.33333)
    #   chroma[from, to]: floats all in range 0 .. 1
    #   luma[from, to]: floats all in range 0 .. 1
    # Returns a list of #rgb colour strings:

    # convert stand alone values to ranges
    if not isinstance(hue, collections.abc.Iterable):
        hue = (hue, hue)
    if not isinstance(chroma, collections.abc.Iterable):
        chroma = (chroma, chroma)
    if not isinstance(luma, collections.abc.Iterable):
        luma = (luma, luma)

    # convert ints to floats
    hue = [float(hue[y]) for y in (0, 1)]
    chroma = [float(chroma[y]) for y in (0, 1)]
    luma = [float(luma[y]) for y in (0, 1)]

    # some sanity checks
    n = int(n)
    if n < 1 or n > 360 :
        raise ValueError('n is a value between 1 and 360')
    if any([chroma[y] < 0.0 or chroma[y] > 1.0 for y in (0, 1)]) :
        raise ValueError('chroma is a value between 0 and 1')
    if any([luma[y] < 0.0 or luma[y] > 1.0 for y in (0, 1)]) :
        raise ValueError('luma is a value between 0 and 1')

    # generate a list of hex colour strings
    x = n + 1 if n % 2 else n
    if n > 1 :
        lDiff = (luma[1] - luma[0]) / float(n - 1.0)
        cDiff = (chroma[1] - chroma[0]) / float(n - 1.0)
        if skipHue :
            hDiff = (hue[1] - hue[0]) / float(x)
        else :
            hDiff = (hue[1] - hue[0]) / float(x - 1.0)
    else:
        hDiff = 0.0
        lDiff = 0.0
        cDiff = 0.0

    listOfColours = []
    for i in range(n) :
        c = chroma[0] + i * cDiff
        l = luma[0] + i * lDiff
        h = math.modf(hue[0] + i * hDiff)[0]
        h = h + 1 if h < 0.0 else h
        (h, c, l) = (min(h, 0.99999999999), min(c, 1.0), min(l, 1.0))
        (h, c, l) = (max(h, 0.0), max(c, 0.0), max(l, 0.0))
        (r, g, b) = hcl_to_rgb(h, c, l)
        listOfColours.append( '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) )
    return listOfColours


import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets, Output, FloatRangeSlider, VBox, HBox
from IPython.display import display
from ipywidgets import Layout, Label

# task name -> path
task_dict = {
    'ECG': 'mimic_ecg_test',
    'PPG': 'mimic_ppg_test',
    'PTBXL': 'ptbxl_'
}
# ptbxl name -> path
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
model_orig_to_name_dict = {
    'fft': 'FFT',
    'mean': 'Mean',
    'lininterp': 'Linear Interpolation',
    'bdc883': 'BDC Transformer',
    'dcb883': 'BDC Transformer',
    'van': 'Van',
    'naomistep1': 'Naomi Step1',
    'naomistep64': 'Naomi Step64',
    'naomistep256': 'Naomi Step256',
    'conv9': 'Conv9',
    'deepmvi': 'DeepMVI',
    'brits': 'Brits'
}
model_name_to_orig_dict = {v: k for k, v in model_orig_to_name_dict.items()}

model_drop_delete_list = []
model_drop_delete_idx = 0


# Set up initial variables
out_folder = "out"
tasks = ['ECG', 'PPG', 'PTBXL']
current_task = tasks[0]
#tasks = sorted([task for task in os.listdir(out_folder) if not task.startswith('.')])
ptbxl1s = ['Extracted mHealth Missingness Patterns']
current_ptbxl1 = ptbxl1s[0]
ptbxl2s = ['---']
current_ptbxl2 = ptbxl2s[0]

ignores = ['Extracted mHealth Missingness Patterns', '---']

available_models = os.listdir(os.path.join(out_folder, 
                                           task_dict[current_task] + ptbxl_dict1[current_ptbxl1]))
current_models = [available_models[0 % len(available_models)], 
                  available_models[1 % len(available_models)]]  # Default models
current_sample_index = 0
change_flag = True

line_colors = ggColorSlice(n=10)
order = [5, 0, 3, 7, 1, 8, 6, 9, 4, 2]
curr_colors = [5, 0]

def model_path_to_alias(paths):
    ret = []
    for p in paths:
        ret.append(transform_model_path[p.split('_')[0]])
    return ret

def add_next_color():
    global curr_colors
    for i in order:
        if i not in curr_colors:
            curr_colors.append(i)
            return
    raise Exception("exceeded model dropdown capacity")

# Load data function
def load_data(task, ptbxl1, ptbxl2, models):
    task = task_dict[task]
    ptbxl1 = ptbxl_dict1[ptbxl1]
    ptbxl2 = ptbxl_dict2[ptbxl2]
    
    data = {}
    for model in models:
        if model != '---':
            task_path = os.path.join(out_folder, task + ptbxl1 + ptbxl2, model)
            data[model] = {
                'original': np.load(os.path.join(task_path, "original.npy")),
                'imputation': np.load(os.path.join(task_path, "imputation.npy")),
                'target_seq': np.load(os.path.join(task_path, "target_seq.npy"))
            }
    return data

# Get the default sample length for the current task
default_sample_length = len(load_data(current_task, current_ptbxl1, current_ptbxl2, current_models)[current_models[0]]['original'][current_sample_index])

# Update function
def update_plot(task, ptbxl1, ptbxl2, models, sample_index, x_range):
    # check if task updated
    '''
    tasks = {'ecg':'mimic_ecg', 'ppg':'mimic_ppg', 'ptbxl':'ptbxl'}
    if len(models) > 1:
        # Get the common tasks between the two strings
        common_tasks = set(list(tasks.keys())) & set(models[0].lower().split('_') + models[1].lower().split('_'))
        if len(common_tasks) > 1:
            print(4)
            return
    '''
    if not change_flag:
        return
                
    #data = load_data(task, ptbxl1, ptbxl2, models)
    available_models = os.listdir(os.path.join(out_folder, task_dict[task] + ptbxl_dict1[ptbxl1] + ptbxl_dict2[ptbxl2]))
    data = load_data(task, ptbxl1, ptbxl2, available_models)
    sample_length = len(data[available_models[0]]['original'][sample_index])
    for model in models:
        if model != '---':
            data[model]['original'][sample_index][data[model]['original'][sample_index] == 0] = np.nan
            data[model]['imputation'][sample_index][data[model]['imputation'][sample_index] == 0] = np.nan
            data[model]['target_seq'][sample_index][data[model]['target_seq'][sample_index] == 0] = np.nan

    with output:
        output.clear_output(wait=True)
        
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Scatter(x=np.arange(sample_length),
                                 y=data[available_models[0]]['original'][sample_index].flatten(),
                                 mode='lines',
                                 name='Original',
                                 line=dict(color='black'),
                                 ),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(sample_length),
                                 y=data[available_models[0]]['target_seq'][sample_index].flatten(),
                                 mode='lines',
                                 name='Target Sequence',
                                 line=dict(color='lightgray'),
                                 ),
                      row=1, col=1)
        for idx, model in enumerate(models):
            if model != '---':
                fig.add_trace(go.Scatter(x=np.arange(sample_length),
                                        y=data[model]['imputation'][sample_index].flatten(),
                                        mode='lines',
                                        name=model.split('_')[0],
                                        line=dict(color=line_colors[curr_colors[idx]], width=3),
                                        ),
                            row=1, col=1)

        fig.update_xaxes(range=x_range, showgrid=False)
        fig.update_yaxes(showgrid=False)

        fig.update_layout(
            title="Data Visualization",
            xaxis_title="Data Points",
            yaxis_title="Values",
            height=600,
            width=800
        )
        
        
        fig.show()

# Define dropdown, slider, and output widgets
task_dropdown = widgets.Dropdown(options=tasks, value=current_task, description='Task:')
ptbxl1_dropdown = widgets.Dropdown(options=ptbxl1s, 
                                   value=current_ptbxl1, description='miss type:')
ptbxl2_dropdown = widgets.Dropdown(options=ptbxl2s, 
                                   value=current_ptbxl2, description='ptbxl % miss:')
#ptbxl1_dropdown.layout.visibility = 'hidden'
ptbxl2_dropdown.layout.visibility = 'hidden'
model_dropdowns_container = VBox([])

model_dropdowns = [widgets.Dropdown(options=[(model_orig_to_name_dict[i.split('_')[0]], i) for i in os.listdir(os.path.join(out_folder, ptbxl_dict1[current_ptbxl1],
                                                                    ptbxl_dict2[current_ptbxl2],
                                                                    task_dict[current_task]))], 
                                    value=current_models[idx], 
                                    description=f'Model {idx + 1}:') for idx in range(len(current_models))]
model_dropdowns_container.children = model_dropdowns
sample_dropdown = widgets.Dropdown(options=list(range(10)), 
                                   value=current_sample_index, description='Signal idx:')
previous_sample_button = widgets.Button(description='Previous')
next_sample_button = widgets.Button(description='Next')
add_model_button = widgets.Button(description='Add Model')
delete_model_button = widgets.Button(description='Delete Model')

output = Output()

# Set up event handlers for previous and next buttons
def on_previous_sample_button_clicked(b):
    sample_dropdown.value = (sample_dropdown.value - 1) % len(sample_dropdown.options)

def on_next_sample_button_clicked(b):
    sample_dropdown.value = (sample_dropdown.value + 1) % len(sample_dropdown.options)

def delete_model_row(dl_list_idx):
    #print('Deleting ' + str(dl_list_idx))
    global model_dropdowns, delete_buttons, curr_colors
    global model_drop_delete_list, model_drop_delete_idx
    idx = model_drop_delete_list.index(dl_list_idx)
    model_drop_delete_list.remove(dl_list_idx)
    del curr_colors[idx]
    model_drop_delete_idx += 1

    # Remove the corresponding dropdown and delete button
    model_dropdowns[idx].close()
    delete_buttons[idx].close()

    # Remove event handlers
    model_dropdowns[idx].unobserve_all()

    # Delete the dropdown and delete button from the lists
    del model_dropdowns[idx]
    del delete_buttons[idx]

    '''
    # Update indices of remaining dropdowns and delete buttons
    for i in range(idx, len(model_dropdowns)):
        # Update the description of the dropdown
        model_dropdowns[i].description = f'Model {i + 1}:'
        # Update the description of the delete button
        delete_buttons[i].description = 'Delete'
    '''

    global change_flag
    change_flag = False

    # Update the plot and other functionalities
    generic_task_handler()

    # Reset the change flag
    change_flag = True

    '''
    # Update indices of remaining dropdowns and delete buttons
    for i in range(idx, len(model_dropdowns)):
        delete_buttons[i].on_click(lambda _, idx=i - 1: delete_model_row(i - 1))
        
    for i in range(len(model_dropdowns)):
        print(type(delete_buttons[i]))
    print('---------')
    '''

# Create delete buttons dynamically for each model dropdown
delete_buttons = []
for idx, dropdown in enumerate(model_dropdowns):
    global model_drop_delete_list, model_drop_delete_idx
    #delete_button = widgets.Button(description=f'Delete {model_drop_delete_idx}:')
    delete_button = widgets.Button(description=f'Delete')
    delete_buttons.append(delete_button)
    model_drop_delete_list.append(model_drop_delete_idx)
    delete_button.on_click(lambda _, idx=model_drop_delete_idx: delete_model_row(idx))
    model_drop_delete_idx += 1

# Update the model_dropdowns_container to include delete buttons
model_dropdowns_container = VBox([HBox([dropdown, delete_button]) for dropdown, delete_button in zip(model_dropdowns, delete_buttons)])

# Add Model button event handler
def on_add_model_button_clicked(b):
    global model_dropdowns_container
    global model_drop_delete_list, model_drop_delete_idx
    available_models = os.listdir(os.path.join(out_folder, 
                                                         task_dict[task_dropdown.value] + ptbxl_dict1[ptbxl1_dropdown.value] + ptbxl_dict2[ptbxl2_dropdown.value]))
    model_dropdown_tuples = [(model_orig_to_name_dict[i.split('_')[0]], i) for i in available_models]
    new_model_dropdown = widgets.Dropdown(options=[('---', '---')] + model_dropdown_tuples, 
                                          value='---', 
                                          description=f'Model {len(model_dropdowns) + 1}:')
    new_model_dropdown.observe(model_dropdown_eventhandler, names='value')
    model_dropdowns.append(new_model_dropdown)
    # Create delete button for the new dropdown
    #delete_button = widgets.Button(description=f'Delete {model_drop_delete_idx}:')
    delete_button = widgets.Button(description=f'Delete')
    model_drop_delete_list.append(model_drop_delete_idx)
    add_next_color()
    delete_button.on_click(lambda _, idx=model_drop_delete_idx: delete_model_row(idx))
    model_drop_delete_idx += 1
    delete_buttons.append(delete_button)
    # Update model_dropdowns_container to include the new dropdown and delete button
    model_dropdowns_container.children = tuple(model_dropdowns_container.children) + (HBox([new_model_dropdown, delete_button]),)

    update_plot(task_dropdown.value, ptbxl1_dropdown.value, ptbxl2_dropdown.value,
                [dropdown.value for dropdown in model_dropdowns], sample_dropdown.value, x_range_slider.value)

# Attach event handlers to buttons
add_model_button.on_click(on_add_model_button_clicked)

    
# Attach event handlers to buttons
previous_sample_button.on_click(on_previous_sample_button_clicked)
next_sample_button.on_click(on_next_sample_button_clicked)
add_model_button.on_click(on_add_model_button_clicked)

# Set up x-axis range slider with default range based on the current task
x_range_slider = FloatRangeSlider(
    value=[0, default_sample_length - 1], 
    min=0, 
    max=default_sample_length - 1, 
    step=1, 
    description='X-axis Range', 
    continuous_update=False
)
    
def generic_task_handler():
    global change_flag
    available_models = os.listdir(os.path.join(out_folder, 
                                               task_dict[task_dropdown.value] + ptbxl_dict1[ptbxl1_dropdown.value] + ptbxl_dict2[ptbxl2_dropdown.value]))
    model_dropdown_tuples = [(model_orig_to_name_dict[i.split('_')[0]], i) for i in available_models]
    for d in range(len(model_dropdowns)):
        dropdown = model_dropdowns[d]
        old_dropdown = dropdown.value.split('_')[0]
        # for some reason, dropdown.value gets set to first option after this line
        dropdown.options = [('---', '---')] + model_dropdown_tuples
        # Attempt to set to same model in new task
        # Check if any string in the list starts with model name
        selected_option = next((s[1] for s in dropdown.options if s[1].startswith(old_dropdown)), '---')
        dropdown.value = selected_option
        
    change_flag = True
    update_plot(task_dropdown.value, ptbxl1_dropdown.value, ptbxl2_dropdown.value,
                [dropdown.value for dropdown in model_dropdowns], sample_dropdown.value, x_range_slider.value)
    
    # Reset slider to the min and max x values
    base_data = load_data(task_dropdown.value, ptbxl1_dropdown.value, ptbxl2_dropdown.value, available_models)
    sample_length = len(base_data[available_models[0]]['original'][sample_dropdown.value])
    
    #sample_length = len(load_data(change.new, [dropdown.value for dropdown in model_dropdowns])[model_dropdowns[0].value]['original'][sample_dropdown.value])
    x_range_slider.min = 0
    x_range_slider.max = sample_length - 1
    x_range_slider.value = [0, sample_length - 1]  # Reset slider range
    
# Define event handlers
def task_dropdown_eventhandler(change):
    global change_flag
    change_flag = False
    
    # change ptbxl dropdowns first
    old_ptbxl1 = ptbxl1_dropdown.value
    old_ptbxl2 = ptbxl2_dropdown.value
    # for some reason, dropdown.value gets set to first option after this line
    if 'ptbxl' in change.new.lower():
        ptbxl1_dropdown.layout.visibility = 'visible'
        ptbxl1_dropdown.options = [k for k in ptbxl_dict1.keys() if k not in ignores]
    else:
        ptbxl1_dropdown.layout.visibility = 'visible'
        ptbxl1_dropdown.options = ['Extracted mHealth Missingness Patterns']
    ptbxl1_dropdown.value = ptbxl1_dropdown.options[0]
    
    generic_task_handler()
    
def ptbxl1_dropdown_eventhandler(change):
    global change_flag
    change_flag = False
    
    old_drop2 = ptbxl2_dropdown.value
    if change.new not in ignores:
        ptbxl2_dropdown.layout.visibility = 'visible'
        ptbxl2_dropdown.options = [k for k in ptbxl_dict2.keys() if k != '---']
    else:
        ptbxl2_dropdown.layout.visibility = 'hidden'
        ptbxl2_dropdown.options = ['---']
    if old_drop2 in ptbxl2_dropdown.options:
        ptbxl2_dropdown.value = old_drop2
    else:
        ptbxl2_dropdown.value = ptbxl2_dropdown.options[0]
    
    generic_task_handler()
    
def ptbxl2_dropdown_eventhandler(change):
    global change_flag
    change_flag = False
    
    generic_task_handler()


def model_dropdown_eventhandler(change):
    update_plot(task_dropdown.value, ptbxl1_dropdown.value, ptbxl2_dropdown.value, 
                [dropdown.value for dropdown in model_dropdowns], sample_dropdown.value, x_range_slider.value)

def sample_dropdown_eventhandler(change):
    update_plot(task_dropdown.value, ptbxl1_dropdown.value, ptbxl2_dropdown.value, 
                [dropdown.value for dropdown in model_dropdowns], change.new, x_range_slider.value)

def x_range_slider_eventhandler(change):
    update_plot(task_dropdown.value, ptbxl1_dropdown.value, ptbxl2_dropdown.value, 
                [dropdown.value for dropdown in model_dropdowns], sample_dropdown.value, change.new)

# Attach event handlers to widgets
task_dropdown.observe(task_dropdown_eventhandler, names='value')
ptbxl1_dropdown.observe(ptbxl1_dropdown_eventhandler, names='value')
ptbxl2_dropdown.observe(ptbxl2_dropdown_eventhandler, names='value')
for dropdown in model_dropdowns:
    dropdown.observe(model_dropdown_eventhandler, names='value')
sample_dropdown.observe(sample_dropdown_eventhandler, names='value')
x_range_slider.observe(x_range_slider_eventhandler, names='value')

# gap control
small_spacer = Output(layout=Layout(height='10px'))
separator_label = Label(value='----------' * 10, layout=Layout(margin='-5px 0 -5px 0'))
spacer = Output(layout=Layout(height='20px'))

# Display widgets
display(VBox([
    HBox([task_dropdown]),
    HBox([ptbxl1_dropdown, ptbxl2_dropdown]),
    HBox([sample_dropdown, previous_sample_button, next_sample_button]), 
    x_range_slider,
    separator_label,
    model_dropdowns_container,
    add_model_button,
    output]))

# Initial plot with default values
update_plot(current_task, current_ptbxl1, current_ptbxl2, current_models, current_sample_index, x_range_slider.value)