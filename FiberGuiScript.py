# %load_ext autoreload

# %autoreload 2

#Import everything
import io
import param
import panel as pn
import pandas as pd
import csv
import numpy as np
import os
import random
import ipywidgets as ipw
import time
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
from tornado.ioloop import IOLoop
import FiberClass as fc


'''
Command to run script:
    panel serve --show FiberGuiScript.py --websocket-max-message-size=104876000 --autoreload
'''


pn.extension('plotly', sizing_mode = "stretch_width", loading_color = '#00aa41')


#Dictionary of fiber objects
fiber_objs = {}
#Dataframe of object's info
fiber_data = pd.DataFrame(columns = ['Fiber #', 'Animal #', 'Exp. Date', 'Exp. Start Time', 'Filename'])

#Read fpho data
def run_init_fiberobj(event = None):
    # .value param to extract variables properly
    value = fpho_input.value
    file_name = fpho_input.filename
    obj_name = input_1.value
    fiber_num = input_2.value
    animal_num = input_3.value
    exp_date = input_4.value
    exp_time = input_5.value
    if value:
        try:
            #Add input params to list for initialization
            input_params = []
            input_params.extend([obj_name, fiber_num, animal_num, exp_date, exp_time, file_name])
            string_io = io.StringIO(value.decode("utf8"))
            df = pd.read_csv(string_io) #Read into dataframe
        except FileNotFoundError:
            print("Could not find file: " + fpho_input)
            sys.exit(2)
        except PermissionError:
            print("Could not access file: " + fpho_input)
            sys.exit(3)
    
    if df.empty:
        print("Dataframe is empty")
        sys.exit(4)
    else:
        #Add to dict if object name does not already exist
        if input_params[0] not in fiber_objs:
            #Creates new object
            new_obj = fc.fiberObj(df, input_params[0], input_params[1], input_params[2], input_params[3], input_params[4], input_params[5])
            #Adds to dict
            fiber_objs[input_params[0]] = new_obj
            #Adds to relevant info to dataframe
            fiber_data.loc[input_params[0]] = ([input_params[1], input_params[2], input_params[3], input_params[4], input_params[5]])
            info_table.value = fiber_data
            existing_objs = fiber_objs
            #Updates selectors with new objects
            obj_selecta.options = [*existing_objs]
            norm_selecta.options = [*existing_objs]
            behav_selecta.options = [*existing_objs]
            plot_beh_selecta.options = [*existing_objs]
            zscore_selecta.options = [*existing_objs]
            pearsons_selecta1.options = [*existing_objs]
            pearsons_selecta2.options = [*existing_objs]
            beh_corr_selecta1.options = [*existing_objs]
            beh_corr_selecta2.options = [*existing_objs]
            save_obj_selecta.options = [*existing_objs]
            info_selecta.options = [*existing_objs]
        else:
            print('This object name already exists')


# Upload pickled object files
def run_upload_fiberobj(event = None):
    upload = upload_pkl_selecta.filename
    for filename in upload:
        with io.open (filename, 'rb') as file:
            try:
                temp = pickle.load(file)
            except EOFError:
                break
    fiber_objs[temp.obj_name] = temp
    fiber_data.loc[temp.obj_name] = ([temp.fiber_num, temp.animal_num, temp.exp_date, temp.exp_start_time, temp.file_name])
    info_table.value = fiber_data
    existing_objs = fiber_objs
    # Updates all cards with new objects
    obj_selecta.options = [*existing_objs]
    norm_selecta.options = [*existing_objs]
    behav_selecta.options = [*existing_objs]
    plot_beh_selecta.options = [*existing_objs]
    zscore_selecta.options = [*existing_objs]
    pearsons_selecta1.options = [*existing_objs]
    pearsons_selecta2.options = [*existing_objs]
    beh_corr_selecta1.options = [*existing_objs]
    beh_corr_selecta2.options = [*existing_objs]
    save_obj_selecta.options = [*existing_objs]
    info_selecta.options = [*existing_objs]
        
        
# Saves selected object to pickle file
def run_save_fiberobj(event = None):
    obj = save_obj_selecta.value
    for obj in save_obj_selecta.value:
        temp = fiber_objs[obj]
        with open(obj + '.pickle', 'wb') as handle:
            pickle.dump(temp, handle)
        save_obj_box.append('# ' + temp.obj_name + ' pickled succesfully')
        
            

# Creates raw plot pane
def run_plot_raw_trace(event = None):
    # .value param to extract variables properly
    selected_objs = obj_selecta.value
    
    #For len of selected objs, create and plot raw signal graph
    for objs in selected_objs:
        temp = fiber_objs[objs]
        plot_pane = pn.pane.Plotly(height = 300, sizing_mode = "stretch_width") #Creates pane for plotting
        plot_pane.object = temp.raw_signal_trace() #Sets figure to plot variable
        # plot_pane.trigger('object')
        plot_raw_card.append(plot_pane) #Add figure to template

        
# Creates normalize signal pane
def run_normalize_a_signal(event = None):
    # .value param to extract variables properly
    selected_objs = norm_selecta.value
    #For len of selected objs, create and plot raw signal graph
    for objs in selected_objs:
        temp = fiber_objs[objs]
        plot_pane = pn.pane.Plotly(height = 900, sizing_mode = "stretch_width") #Creates pane for plotting
        plot_pane.object = temp.normalize_a_signal(pick_signal.value, pick_reference.value) #Sets figure to plot variable
        # plot_pane.trigger('object')
        norm_sig_card.append(plot_pane) #Add figure to template
        
        
#Read behavior data
def run_import_behavior_data(event = None):
    behav = behav_input.value
    path = io.StringIO(behav.decode("utf8"))
    selected_obj = behav_selecta.value
    obj = fiber_objs[selected_obj]
    # fpho = obj.fpho_data_df
    
    if behav:
        obj.import_behavior_data(path)
        upload_beh_card.append('# behavior added to ' + selected_obj + ' successfully')
    else:
        print("Error reading behavior data")
        sys.exit(5)
        
        
#Plot behavior on a full trace
def run_plot_behavior(event = None): 
    selected_objs = plot_beh_selecta.value
    #For len of selected objs, create and plot behavior data
    for objs in selected_objs:
        temp = fiber_objs[objs]
        plot_pane = pn.pane.Plotly(height = len(channel_selecta.value)*300, sizing_mode = "stretch_width") #Creates pane for plotting
        plot_pane.object = temp.plot_behavior(behavior_selecta.value, channel_selecta.value) #Sets figure to plot variable
        plot_beh_card.append(plot_pane) #Add figure to template
        
        
#Plot zscore of a point evnt
def run_plot_zscore(event = None): 
    selected_objs = zscore_selecta.value
    baseline_vals = np.array([baseline_start.value, baseline_end.value])
    # How user would like to apply the baseline window input
    baseline_option = baseline_selecta.value
    #For len of selected objs, create and plot zscores
    for objs in selected_objs:
        temp = fiber_objs[objs]
        for beh in zbehs_selecta.value:
            for channel in zchannel_selecta.value:
                plot_pane = pn.pane.Plotly(height = 500, sizing_mode = "stretch_width") #Creates pane for plotting
                plot_pane.object = temp.plot_zscore(channel, beh, time_before.value, time_after.value, baseline_vals, baseline_option) #Sets figure to plot variable
                zscore_card.append(plot_pane) #Add figure to template
                
                
# Runs the pearsons correlation coefficient
def run_trial_pearsons(event = None):
    for channels in pearsons_channel_selecta.value:
        name1 = pearsons_selecta1.value
        name2 = pearsons_selecta2.value
        obj1 = fiber_objs[name1]
        obj2 = fiber_objs[name2]
        plot_pane = pn.pane.Plotly(height = 300, sizing_mode = "stretch_width") #Creates pane for plot
        plot_pane.object = obj1.within_trial_pearsons(obj2, channels)
        pearsons_card.append(plot_pane) #Add figure to template

def run_beh_specific_pearsons(event = None):
    for channel in beh_corr_channel_selecta.value:
        for behavior in beh_corr_behavior_selecta.value:
            name1 = beh_corr_selecta1.value
            name2 = beh_corr_selecta2.value
            obj1 = fiber_objs[name1]
            obj2 = fiber_objs[name2]
            plot_pane = pn.pane.Plotly(height = 300, sizing_mode = "stretch_width") #Creates pane for plot
            plot_pane.object = obj1.behavior_specific_pearsons(obj2, channel, behavior)
            beh_corr_card.append(plot_pane) #Add figure to template 

            
#Updates available signal options based on selected object
def update_selecta_options(event = None): 
    # Normalize Card
    selected_norm = norm_selecta.value
    if selected_norm:
        #For len of selected objs, create and plot behavior data
        available_channels = fiber_objs[selected_norm[0]].channels
        for objs in selected_norm:
            temp = fiber_objs[objs]
            available_channels = temp.channels & available_channels
        pick_signal.options = list(available_channels)
        pick_reference.options = list(available_channels)
    
    
    # Plot Behav card
    selected_behav = plot_beh_selecta.value
    if selected_behav:
        #For len of selected objs, create and plot behavior data
        available_channels = fiber_objs[selected_behav[0]].channels
        available_behaviors = fiber_objs[selected_behav[0]].behaviors
        for objs in selected_behav:
            temp = fiber_objs[objs]
            available_channels = temp.channels & available_channels
            available_behaviors = temp.behaviors & available_behaviors
        channel_selecta.options = list(available_channels)
        behavior_selecta.options = list(available_behaviors)
    
    # Z-Score card
    selected_zscore = zscore_selecta.value
    if selected_zscore:
        #For len of selected objs, create and plot zscores
        available_behaviors = fiber_objs[selected_zscore[0]].behaviors
        available_channels = fiber_objs[selected_zscore[0]].channels
        for objs in selected_zscore:
            temp = fiber_objs[objs]
            available_behaviors = temp.behaviors & available_behaviors
            available_channels = temp.channels & available_channels
        zbehs_selecta.options = list(available_behaviors)
        zchannel_selecta.options = list(available_channels)
      
    #Full Pearsons card
    name1 = pearsons_selecta1.value
    name2 = pearsons_selecta2.value
    obj1 = fiber_objs[name1]
    obj2 = fiber_objs[name2]
    available_channels = obj1.channels & obj2.channels
    pearsons_channel_selecta.options = list(available_channels)
    
    #Correlation for a behavior
    name1 = beh_corr_selecta1.value
    name2 = beh_corr_selecta2.value
    obj1 = fiber_objs[name1]
    obj2 = fiber_objs[name2]
    available_channels = obj1.channels & obj2.channels
    available_behaviors = obj1.behaviors & obj2.behaviors
    beh_corr_channel_selecta.options = list(available_channels)
    beh_corr_behavior_selecta.options = list(available_behaviors)
    


# In[3]:
#Template and widget declarations
ACCENT_COLOR = "#0072B5"
template = pn.template.MaterialTemplate(site = 'Donaldson Lab: Fiber Photometry', title = 'FiberPho GUI',
                                       sidebar = ["**Upload CSV** and set **Input Parameters** for your fiber object here"],
                                        main = [], accent_base_color = ACCENT_COLOR, header_background = ACCENT_COLOR
                                       )

#Dict of objects
# ----------------------------------------------------- # 
#Init fiberobj Widget

#Input variables
input_1 = pn.widgets.TextInput(name = 'Object Name', width = 90, placeholder = 'String')
input_2 = pn.widgets.IntInput(name = 'Fiber Number', width = 90, placeholder = 'Int')
input_3 = pn.widgets.IntInput(name = 'Animal Number', width = 90, placeholder = 'Int')
input_4 = pn.widgets.TextInput(name = 'Exp Date', width = 90, placeholder = 'Date')
input_5 = pn.widgets.TextInput(name = 'Exp Time', width = 90, placeholder = 'Time')
input_col = pn.Column(input_1, input_2, input_3, input_4, input_5)
fpho_input = pn.widgets.FileInput(name = 'Upload FiberPho Data', accept = '.csv') #File input parameter

#Buttons
upload_button = pn.widgets.Button(name = 'Create Object', button_type = 'primary', width = 500, sizing_mode = 'stretch_width', align = 'end')
upload_button.on_click(run_init_fiberobj) #Button action

#Box
init_obj_box = pn.WidgetBox('# Input Params', fpho_input, input_col, upload_button)

# ----------------------------------------------------- # 
# ----------------------------------------------------- # 
#Load fiberobj Widget

#Input variables
upload_pkl_selecta = pn.widgets.FileInput(name = 'Upload Saved Fiber Objects', accept = '.pickle', multiple=True) #File input parameter

#Buttons
upload_pkl_btn = pn.widgets.Button(name = 'Upload Object', button_type = 'primary', width = 500, sizing_mode = 'stretch_width', align = 'end')
upload_pkl_btn.on_click(run_upload_fiberobj) #Button action

#Box
load_obj_box = pn.WidgetBox('# Reload saved Fiber Objects', upload_pkl_selecta, upload_pkl_btn)

# ----------------------------------------------------- #

# ----------------------------------------------------- # 
#Save fiberobj Widget

#Input variables
save_obj_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects', value = [], options = [], )

#Buttons
save_obj_btn = pn.widgets.Button(name = 'Save Object', button_type = 'primary', width = 500, sizing_mode = 'stretch_width', align = 'end')
save_obj_btn.on_click(run_save_fiberobj) #Button action

#Box
save_obj_box = pn.WidgetBox('# Save Fiber Objects for later', save_obj_selecta, save_obj_btn)

# ----------------------------------------------------- #

# ----------------------------------------------------- # 
#Plot raw signal Widget

#Input vairables
obj_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects', value = [], options = [], )

#Buttons
plot_raw_btn = pn.widgets.Button(name = 'Plot Raw Signal', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
plot_raw_btn.on_click(run_plot_raw_trace)

#Box
plot_options = pn.Column(obj_selecta, plot_raw_btn)
plot_raw_widget = pn.WidgetBox('# Options', plot_options)
plot_raw_card = pn.Card(plot_raw_widget, title = 'Plot Raw Signal', background = 'WhiteSmoke', width = 600, collapsed = True)
# ----------------------------------------------------- # 
#Normalize signal to reference Widget
#Input vairables

norm_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects', value = [], options = [], )
pick_signal = pn.widgets.Select(name = 'Signal', options = [])
pick_reference = pn.widgets.Select(name = 'Reference', options = [])
#Buttons
norm_sig_btn = pn.widgets.Button(name = 'Normalize Signal', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
norm_sig_btn.on_click(run_normalize_a_signal)
update_norm_options_btn = pn.widgets.Button(name = 'Update Signal/Reference Options', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
update_norm_options_btn.on_click(update_selecta_options)

#Box
norm_options = pn.Column(norm_selecta, update_norm_options_btn, pick_signal, pick_reference, norm_sig_btn)
norm_sig_widget = pn.WidgetBox(norm_options)
norm_sig_card = pn.Card(norm_sig_widget, title = 'Normalize to a reference', background = 'WhiteSmoke', width = 600, collapsed= True)


# ----------------------------------------------------- # 
#Add Behavior Widget

#Input variables
behav_input = pn.widgets.FileInput(name = 'Upload Behavior Data', accept = '.csv') #File input parameter
behav_selecta = pn.widgets.Select(name = 'Fiber Objects', value = [], options = [], )


#Buttons
upload_beh_btn = pn.widgets.Button(name = 'Read Behavior Data', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
upload_beh_btn.on_click(run_import_behavior_data) #Button action

#Box
behav_options = pn.Column(behav_selecta, behav_input, upload_beh_btn)
upload_beh_widget = pn.WidgetBox('# Import Behavior file', behav_options)
upload_beh_card = pn.Card(upload_beh_widget, title = 'Import Behavior', background = 'WhiteSmoke', width = 600, collapsed = True)

# ----------------------------------------------------- # 

# ----------------------------------------------------- # 
#Add Behavior plot Widget

#Input variables
plot_beh_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects', value = [], options = [], )
channel_selecta = pn.widgets.MultiSelect(name = 'Signal', value = [], options = [], )
behavior_selecta = pn.widgets.MultiSelect(name = 'Behavior', value = [], options = [], )

#Buttons
plot_beh_btn = pn.widgets.Button(name = 'Plot Behavior', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
plot_beh_btn.on_click(run_plot_behavior) #Button action
update_plot_options_btn = pn.widgets.Button(name = 'Update Options', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
update_plot_options_btn.on_click(update_selecta_options) #Button action

#Box
plot_beh_options = pn.Column(plot_beh_selecta, update_plot_options_btn, channel_selecta, behavior_selecta, plot_beh_btn)
plot_beh_widget = pn.WidgetBox('# Plot Behavior', plot_beh_options)
plot_beh_card = pn.Card(plot_beh_widget, title = 'Plot Behavior', background = 'WhiteSmoke', width = 600, collapsed = True)

# ----------------------------------------------------- # 
# ----------------------------------------------------- # 
#Plot Z-Score

#Input variables
zscore_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects', value = [], options = [], )
zbehs_selecta = pn.widgets.MultiSelect(name = 'Behavior', value = [], options = [], )
zchannel_selecta = pn.widgets.MultiSelect(name = 'Signal', value = [], options = [], )
time_before = pn.widgets.IntInput(name = 'Time before event(s)', width = 50, placeholder = 'Seconds', value = 2)
time_after = pn.widgets.IntInput(name = 'Time after initiation(s)', width = 50, placeholder = 'Seconds', value = 5)
baseline_start = pn.widgets.LiteralInput(name = 'Baseline Window Start Time (s)', width = 50, placeholder = 'Seconds', value = 0)
baseline_end = pn.widgets.LiteralInput(name = 'Baseline Window End Time (s)', width = 50, placeholder = 'Seconds', value = 0)
z_score_note = pn.pane.Markdown("""
                                   ***Note :***<br>
                                   - Baseline Window Parameters should be kept 0 unless you are using baseline<br> 
                                   z-score computation method. The parameters are in seconds. <br>
                                   - Please check where you would like your baseline window, **ONLY check one box**. <br>
                                   """, width = 200)

#Buttons
zscore_btn = pn.widgets.Button(name = 'Zscore of Behavior', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
zscore_btn.on_click(run_plot_zscore) #Button action
options_btn = pn.widgets.Button(name = 'Update Options', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
options_btn.on_click(update_selecta_options) #Button action
baseline_selecta = pn.widgets.CheckBoxGroup(
    name = 'Baseline Options', value = [], options = ['Start of Sample', 'Before Events', 'End of Sample'],
    inline = True)

#Box
zscore_options = pn.Column(zscore_selecta, options_btn, zchannel_selecta, zbehs_selecta, time_before, time_after, zscore_btn)
baseline_options = pn.Column(z_score_note, baseline_start, baseline_end, baseline_selecta)
tabs = pn.Tabs(('Z-Score', zscore_options), ('Options', baseline_options))
zscore_widget = pn.WidgetBox('# Zscore Plot', tabs)
zscore_card = pn.Card(zscore_widget, title = 'Zscore Plot', background = 'WhiteSmoke', width = 600, collapsed = True)

# ----------------------------------------------------- # 
# ----------------------------------------------------- # 
#Pearsons Trial widget

#Input variables
pearsons_selecta1 = pn.widgets.Select(name = 'Object 1', value = [], options = [], )
pearsons_selecta2 = pn.widgets.Select(name = 'Object 2', value = [], options = [], )
pearsons_channel_selecta = pn.widgets.MultiSelect(name = 'Signal', value = [], options = [], )

#Buttons
pearsons_btn = pn.widgets.Button(name = 'Calculate Pearsons Correlation', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
pearsons_btn.on_click(run_trial_pearsons) #Button action
pearson_options_btn = pn.widgets.Button(name = 'Update Options', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
pearson_options_btn.on_click(update_selecta_options) #Button action

#Box
pearson_options = pn.Column(pearsons_selecta1, pearsons_selecta2, pearson_options_btn, pearsons_channel_selecta, pearsons_btn)
pearson_widget = pn.WidgetBox('# Pearons Correlation Plot', pearson_options)
pearsons_card = pn.Card(pearson_widget, title = 'Pearsons Correlation Coefficient', background = 'WhiteSmoke', width = 600 , collapsed = True)


# ----------------------------------------------------- # 
# ----------------------------------------------------- # 
#Behavior specific pearsons widget

#Input variables
beh_corr_selecta1 = pn.widgets.Select(name = 'Object 1', value = [], options = [], )
beh_corr_selecta2 = pn.widgets.Select(name = 'Object 2', value = [], options = [], )
beh_corr_channel_selecta = pn.widgets.MultiSelect(name = 'Signal', value = [], options = [], )
beh_corr_behavior_selecta = pn.widgets.MultiSelect(name = 'Behavior', value = [], options = [], )

#Buttons
beh_corr_btn = pn.widgets.Button(name = 'Calculate Pearsons Correlation', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
beh_corr_btn.on_click(run_beh_specific_pearsons) #Button action
beh_corr_options_btn = pn.widgets.Button(name = 'Update Options', button_type = 'primary', width = 200, sizing_mode = 'stretch_width', align = 'start')
beh_corr_options_btn.on_click(update_selecta_options) #Button action

#Box
beh_corr_options = pn.Column(beh_corr_selecta1, beh_corr_selecta2, beh_corr_options_btn, beh_corr_channel_selecta, beh_corr_behavior_selecta, beh_corr_btn)
beh_corr_widget = pn.WidgetBox('# Behavior Specific Correlation Plot', beh_corr_options)
beh_corr_card = pn.Card(beh_corr_widget, title = 'Behavior Specific Pearsons Correlation', background = 'WhiteSmoke', width = 600, collapsed = True)


# ----------------------------------------------------- # 
#Object info widget

#Input variables
info_selecta = pn.widgets.MultiSelect(name = 'Objects', value = [], options = [], )
#Buttons
# obj_info_btn = pn.widgets.Button(name = "Read", button_type = 'primary', width = 200)
# obj_info_btn.on_click(get_obj_info)

#Table
info_table = pn.widgets.Tabulator(fiber_data, theme = "fast", height = 300, page_size = 10)
obj_info_card = pn.Card(info_table, title = "Display Object Attributes", background = 'WhiteSmoke', width = 200)

# ----------------------------------------------------- # 

#Append widgets to gui template
template.sidebar.append(init_obj_box)
template.sidebar.append(save_obj_box)
template.sidebar.append(load_obj_box)
template.sidebar.append(obj_info_card)
template.main.append(plot_raw_card)
template.main.append(norm_sig_card)
template.main.append(upload_beh_card)
template.main.append(plot_beh_card)
template.main.append(zscore_card)
template.main.append(pearsons_card)
template.main.append(beh_corr_card)
# template.main.append(visuals)

template.servable()
# In[4]:





