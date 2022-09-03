# %load_ext autoreload

# %autoreload 2
import io
import param
import panel as pn
import pandas as pd
import csv
import numpy as np
import os
import random
import sys
import ipywidgets as ipw
import time
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
import logging
import traceback
from playsound import playsound
import FiberClass as fc


'''
Command to run script:
    Script : panel serve --show FiberGuiScript.py --websocket-max-message-size=104876000 --autoreload
    Notebook : panel serve FiberGuiNotebook.ipynb --websocket-max-message-size=104876000 --show
'''

pn.extension('plotly')
pn.extension('terminal', notifications = True, sizing_mode = 'stretch_width')

#Dictionary of fiber objects
fiber_objs = {}
#Dataframe of object's info
fiber_data = pd.DataFrame(columns = ['Fiber #', 
                                     'Animal #', 
                                     'Exp. Date',
                                     'Exp. Start Time', 
                                     'Filename',
                                     'Behavior File'])

#Read fpho data
def run_init_fiberobj(event = None):
    # Use .value param to extract variables elements
    value = fpho_input.value
    file_name = fpho_input.filename
    obj_name = input_1.value_input
    fiber_num = input_2.value
    animal_num = input_3.value_input
    exp_date = input_4.value_input
    exp_time = input_5.value_input
    start_time = input_6.value #looking for better name
    stop_time = input_7.value #looking for better name
    
    #Add input params to list for initialization
    input_params = []
    input_params.extend([obj_name, fiber_num, animal_num, exp_date, exp_time,
                         start_time, stop_time, file_name])
    if value:
        try:
            string_io = io.StringIO(value.decode("utf8"))
            df = pd.read_csv(string_io) #Read into dataframe
        except FileNotFoundError:
            print("Could not find file: " + fpho_input)
            return
        except PermissionError:
            print("Could not access file: " + fpho_input)
            return

    try:       
        #Add to dict if object name does not already exist
        if (input_params[0] not in fiber_objs):
            new_obj = fc.fiberObj(df, input_params[0], input_params[1],
                                  input_params[2], input_params[3],
                                  input_params[4], input_params[5], 
                                  input_params[6], input_params[7])
            #Adds to dict
            fiber_objs[input_params[0]] = new_obj
            #Adds to relevant info to dataframe
            fiber_data.loc[input_params[0]] = ([input_params[1], 
                                                input_params[2],
                                                input_params[3], 
                                                input_params[4],
                                                input_params[7],
                                                'NaN'])
            info_table.value = fiber_data
            existing_objs = fiber_objs
            # Updates all cards with new objects
            update_obj_selectas(existing_objs)
            #Object created notification
            pn.state.notifications.success('Created ' + input_params[0]
                                           + ' object!', duration = 4000)
        else:
            pn.state.notifications.error(
                'Error: Please check logger for more info', duration = 4000)
            print('There is already an object with this name')
    except Exception as e:
        logger.error(traceback.format_exc())
        pn.state.notifications.error(
            'Error: Please check logger for more information',
            duration = 4000)
        return



# Upload pickled object files
def run_upload_fiberobj(event = None):
    upload = upload_pkl_selecta.filename
    try:
        for filename in upload:
            with io.open (filename, 'rb') as file:
                try:
                    temp = pickle.load(file)
                except EOFError:
                    break
                    
            fiber_objs[temp.obj_name] = temp
        
            fiber_data.loc[temp.obj_name] = ([temp.fiber_num,
                                              temp.animal_num,
                                              temp.exp_date,
                                              temp.exp_start_time,
                                              temp.file_name,
                                              temp.beh_filename])
            info_table.value = fiber_data

        existing_objs = fiber_objs
        # Updates all cards with new objects
        update_obj_selectas(existing_objs)
        
        #Object uploaded notification
        pn.state.notifications.success('Uploaded ' + temp.obj_name
                                       + ' object!', duration = 4000)
    except Exception as e:
        logger.error(traceback.format_exc())
        pn.state.notifications.error(
            'Error: Please check logger for more info', duration = 4000)
        print("Error uploading file. Ensure this is a valid .pkl file")
        
        
# Saves selected object to pickle file
def run_delete_fiberobj(event = None):
    # obj = delete_obj_selecta.value
    try:
        for obj in delete_obj_selecta.value:
            pn.state.notifications.warning('Deleting ' + obj + ' object!',
                                           duration = 4000)
            del fiber_objs[obj]
            fiber_data.drop([obj], axis = 0, inplace = True)
            
        info_table.value = fiber_data
        existing_objs = fiber_objs
        # Updates all cards with new objects
        update_obj_selectas(existing_objs)
    except Exception as e:
        logger.error(traceback.format_exc())
        pn.state.notifications.error(
            'Error: Please check logger for more info', duration = 4000)
        print("Error: Cannot delete object, please try again.")

# Saves selected object to pickle file
def run_save_fiberobj(event = None):
    # obj = save_obj_selecta.value
    try:
        for obj in save_obj_selecta.value:
            temp = fiber_objs[obj]
            with open(obj + '.pickle', 'wb') as handle:
                pickle.dump(temp, handle)
            pn.state.notifications.success('# ' + temp.obj_name
                                           + ' pickled successfully',
                                           duration = 4000)
    except Exception as e:
        pn.state.notifications.error(
            'Error: Please check logger for more info', duration = 4000)
        logger.error(traceback.format_exc())
        print("Error: Cannot save object, please try again.")
        
        
# Creates raw plot pane
def run_plot_raw_trace(event):
    # .value param to extract variables properly
    selected_objs = obj_selecta.value
    try:
        #For len of selected objs, create and plot raw signal graph
        for objs in selected_objs:
            temp = fiber_objs[objs]
            #Creates pane for plotting
            plot_pane = pn.pane.Plotly(height = 300,
                                       sizing_mode = "stretch_width") 
            #Sets figure to plot variable
            plot_pane.object = temp.raw_signal_trace() 
            plot_raw_card.append(plot_pane) #Add figure to template
    except Exception as e:
        logger.error(traceback.format_exc())
        pn.state.notifications.error(
            'Error: Please check logger for more info', duration = 4000)
        return
        
        
# Creates normalize signal pane
def run_normalize_a_signal(event = None):
    # .value param to extract variables properly
    selected_objs = norm_selecta.value
    try:
        #For len of selected objs, create and plot raw signal graph
        for objs in selected_objs:
            temp = fiber_objs[objs]
            #Creates pane for plotting
            plot_pane = pn.pane.Plotly(height = 900, sizing_mode = "stretch_width") 
            #Sets figure to plot variable
            plot_pane.object = temp.normalize_a_signal(pick_signal.value,
                                                       pick_reference.value) 
            norm_sig_card.append(plot_pane) #Add figure to template
    except Exception as e:
        logger.error(traceback.format_exc())
        pn.state.notifications.error(
            'Error: Please check logger for more info', duration = 4000)
        return
        
#Read behavior data
def run_import_behavior_data(event = None):
    selected_obj = behav_selecta.value
    obj = fiber_objs[selected_obj]
    try:
        behav = behav_input.value
        filename = behav_input.filename
        path = io.StringIO(behav.decode("utf8"))
        obj.import_behavior_data(path, filename)
        fiber_data.loc[obj.obj_name, 'Behavior File'] = obj.beh_filename
        info_table.value = fiber_data
        #upload_beh_card.append("Behavior for " + obj.obj_name + " uploaded")
        pn.state.notifications.success('Uploaded Behavior data for '
                                       + obj.obj_name, duration = 4000)
    except FileNotFoundError:
        print("No file was found")
        return
    except Exception as e:
        logger.error(traceback.format_exc())
        pn.state.notifications.error(
            'Error: Please check logger for more info', duration = 4000)
        return
                
#Plot behavior on a full trace
def run_plot_behavior(event = None): 
    selected_objs = plot_beh_selecta.value
    try:
        #For len of selected objs, create and plot behavior data
        for objs in selected_objs:
            temp = fiber_objs[objs]
            # if temp.beh_file is None: # Bug: Plot behavior still runs even without behavior file
            #Creates pane for plotting
            plot_pane = pn.pane.Plotly(height = 300,
                                       sizing_mode = "stretch_width")
            #Sets figure to plot variable
            plot_pane.object = temp.plot_behavior(behavior_selecta.value,
                                                  channel_selecta.value) 
            plot_beh_card.append(plot_pane) #Add figure to template
            #playsound(audio_chime)
    except Exception as e:
        logger.error(traceback.format_exc())
        pn.state.notifications.error(
            'Error: Please check logger for more info', duration = 4000)
        return

             
#Plot zscore of a point evnt
def run_plot_zscore(event = None): 
    selected_objs = zscore_selecta.value
    baseline_vals = np.array([baseline_start.value, baseline_end.value])
    # How user would like to apply the baseline window input
    baseline_option = baseline_selecta.value
    try:
        #For len of selected objs, create and plot zscores
        for objs in selected_objs:
            temp = fiber_objs[objs]
            for beh in zbehs_selecta.value:
                for channel in zchannel_selecta.value:
                    #Creates pane for plotting
                    plot_pane = pn.pane.Plotly(height = 500,
                                               sizing_mode = "stretch_width") 
                    #Sets figure to plot variable
                    plot_pane.object = temp.plot_zscore(channel, beh, 
                                                        time_before.value, 
                                                        time_after.value, 
                                                        baseline_vals, 
                                                        baseline_option,
                                                        first_trace.value,
                                                        last_trace.value,
                                                        show_every.value) 
                    zscore_card.append(plot_pane) #Add figure to template
                    #playsound(audio_chime)
    except Exception as e:
        logger.error(traceback.format_exc())
        pn.state.notifications.error(
            'Error: Please check logger for more info', duration = 4000)
        return
                
# Runs the pearsons correlation coefficient
def run_pearsons_correlation(event = None):
    try:
        name1 = pearsons_selecta1.value
        name2 = pearsons_selecta2.value
        obj1 = fiber_objs[name1]
        obj2 = fiber_objs[name2]
        channel1 = channel1_selecta.value
        channel2 = channel2_selecta.value
        start = pears_start_time.value
        end = pears_end_time.value
        #Creates pane for plot
        plot_pane = pn.pane.Plotly(height = 300,
                                   sizing_mode = "stretch_width") 
        plot_pane.object = obj1.pearsons_correlation(obj2,
                                                     channel1, channel2,
                                                     start, end)
        pearsons_card.append(plot_pane) #Add figure to template
    except Exception as e:
        logger.error(traceback.format_exc())
        pn.state.notifications.error(
            'Error: Please check logger for more info', duration = 4000)
        return

def run_beh_specific_pearsons(event = None):
    try:
        for channel in beh_corr_channel_selecta.value:
            for behavior in beh_corr_behavior_selecta.value:
                name1 = beh_corr_selecta1.value
                name2 = beh_corr_selecta2.value
                obj1 = fiber_objs[name1]
                obj2 = fiber_objs[name2]
                #Creates pane for plot
                plot_pane = pn.pane.Plotly(height = 300,
                                           sizing_mode = "stretch_width") 
                plot_pane.object = obj1.behavior_specific_pearsons(obj2,
                                                                   channel, 
                                                                   behavior)
                beh_corr_card.append(plot_pane) #Add figure to template 
    except Exception as e:
        logger.error(traceback.format_exc())
        pn.state.notifications.error(
            'Error: Please check logger for more info', duration = 4000)
        return
        

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
      
    #Pearsons card
    name1 = pearsons_selecta1.value
    name2 = pearsons_selecta2.value
    obj1 = fiber_objs[name1]
    obj2 = fiber_objs[name2]
    available_channels1 = obj1.channels
    available_channels2 = obj2.channels
    channel1_selecta.options = list(available_channels1)
    channel2_selecta.options = list(available_channels2)
    
    #Correlation for a behavior
    name1 = beh_corr_selecta1.value
    name2 = beh_corr_selecta2.value
    obj1 = fiber_objs[name1]
    obj2 = fiber_objs[name2]
    available_channels = obj1.channels & obj2.channels
    available_behaviors = obj1.behaviors & obj2.behaviors
    beh_corr_channel_selecta.options = list(available_channels)
    beh_corr_behavior_selecta.options = list(available_behaviors)
    
# Clear plots by card function
def clear_plots(event):
    if clear_raw.clicks:
        for i in range(len(plot_raw_card.objects)):
            if isinstance(plot_raw_card.objects[i], pn.pane.plotly.Plotly):
                plot_raw_card.remove(plot_raw_card.objects[i])
                return
    
    if clear_norm.clicks:
        for i in range(len(norm_sig_card.objects)):
            if isinstance(norm_sig_card.objects[i], pn.pane.plotly.Plotly):
                norm_sig_card.remove(norm_sig_card.objects[i])
                return
    
    if clear_beh.clicks:
        for i in range(len(plot_beh_card.objects)):
            if isinstance(plot_beh_card.objects[i], pn.pane.plotly.Plotly):
                plot_beh_card.remove(plot_beh_card.objects[i])
                return
    
    if clear_zscore.clicks:
        for i in range(len(zscore_card.objects)):
            if isinstance(zscore_card.objects[i], pn.pane.plotly.Plotly):
                zscore_card.remove(zscore_card.objects[i])
                return
    
    if clear_pears.clicks:
        for i in range(len(pearsons_card.objects)):
            if isinstance(pearsons_card.objects[i], pn.pane.plotly.Plotly):
                pearsons_card.remove(pearsons_card.objects[i])
                return
    
    if clear_beh_corr.clicks:
        for i in range(len(beh_corr_card.objects)):
            if isinstance(beh_corr_card.objects[i], pn.pane.plotly.Plotly):
                beh_corr_card.remove(beh_corr_card.objects[i])
                return
            
# Convert lickometer data to boris csv
def run_convert_lick(event):
    file = lick_input.value
    name = lick_input.filename
    if file:
        try:
            string_io = io.StringIO(file.decode("utf8"))
            #Read into dataframe
            lick_file = pd.read_csv(string_io, delimiter = '\s+',
                                    names = ['Time', 'Licks']) 
        except FileNotFoundError:
            print("Could not find file: " + lick_input.filename)
            return
        except PermissionError:
            print("Could not access file: " + lick_input.filename)
            return
    if not lick_file.empty:
        try:
            convert = fc.lick_to_boris(lick_file)
            outputname = lick_input.filename[0:-4] + '_reformatted' + '.csv'
            sio = io.StringIO()
            convert.to_csv(sio, index = False)
            sio.seek(0)
            out_file = pn.widgets.FileDownload(sio, embed = True,
                                               filename = outputname,
                                               button_type = 'success',
                                               label = 'Download Formatted Lickometer Data',
                                               width = 400,
                                               sizing_mode = 'fixed')
            beh_tabs[1].append(out_file)
        except Exception as e:
            logger.error(traceback.format_exc())
            pn.state.notifications.error(
                'Error: Please check logger for more info', duration = 4000)
    else:
        print('Error reading file')
        

def run_download_results(event):
    for types in result_type_selecta.value:
        results = pd.DataFrame()
        if types == 'Zscore Results':
            results = pd.concat([fiber_objs[name].z_score_results
                                for name in results_selecta.value],
                                ignore_index=True)
            results.to_csv(output_name.value + '_zscore_results.csv')
            pn.state.notifications.success(output_name.value +
                                           'Z-Score results downloaded',
                                           duration = 4000)
            print('Z-Score results saved locally to: ' +
                  output_name.value + '_zscore_results.csv')
        if types == 'Correlation Results':
            results = pd.concat([fiber_objs[name].correlation_results
                                for name in results_selecta.value],
                                ignore_index=True)
            results.to_csv(output_name.value + '_correlation_results.csv')
            pn.state.notifications.success(output_name.value +
                                           'Correlation results downloaded',
                                           duration = 4000)
            print('Correlation results saved locally to: ' +
                  output_name.value + '_correlation_results.csv')
        if types == 'Behavior Specific Correlation Reuslts':
            results = pd.concat([fiber_objs[name].beh_corr_results
                                for name in results_selecta.value],
                                ignore_index=True)
            results.to_csv(output_name.value +
                           '_behavior_correlation_results.csv')
            pn.state.notifications.success(output_name.value +
                                           'Behavior Correlation results downloaded',
                                           duration = 4000)
            print('Behavior Correlation results saved locally to: ' + 
                  output_name.value + '_behavior_correlation_results.csv')

def update_obj_selectas(existing_objs):
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
            delete_obj_selecta.options = [*existing_objs]
            results_selecta.options = [*existing_objs]


# ----------------------------------------------------- # 
# Error logger
terminal = pn.widgets.Terminal(options = {"cursorBlink": False}, height = 200,
                               sizing_mode = 'stretch_width')
sys.stdout = terminal
# Logger settings
logger = logging.getLogger("terminal")
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(terminal) # NOTE THIS
stream_handler.terminator = "  \n"
formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")

stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

#Buttons
clear_logs = pn.widgets.Button(name = 'Clear Logs', button_type = 'danger', 
                               height = 30, width = 40,
                               sizing_mode = 'fixed', align = 'end')
# Doesn't work rn for some reason
clear_logs.on_click(terminal.clear)

logger_info = pn.pane.Markdown(""" ##Logger
                            """, height = 40, width = 60)

log_card = pn.Card(pn.Row(logger_info, clear_logs), terminal, title = 'Logs', 
                   background = 'WhiteSmoke', width = 600,
                   collapsed = False, collapsible = False)

# ----------------------------------------------------- # 
# Init fiberobj Widget

#Input variables
input_1 = pn.widgets.TextInput(name = 'Object Name', width = 80,
                               placeholder = 'String')
input_2 = pn.widgets.IntInput(name = 'Fiber Number', start = 1,
                              end = 16, width = 80, placeholder = '1-16')
input_3 = pn.widgets.TextInput(name = 'Animal Number', width = 80,
                               placeholder = 'String')
input_4 = pn.widgets.TextInput(name = 'Exp Date', width = 80,
                               placeholder = 'Date')
input_5 = pn.widgets.TextInput(name = 'Exp Time', width = 80,
                               placeholder = 'Time')
input_6 = pn.widgets.IntInput(name = 'Exclude time from the beginning',
                               width = 90, placeholder = 'Seconds',
                              value = 0) #looking for better name
input_7 = pn.widgets.IntInput(name = 'Stop time from the beginning',
                               width = 90, placeholder = 'Seconds',
                              value = -1) #looking for better name
input_col = pn.Column(input_1, input_2, input_3, input_4,
                      input_5, input_6, input_7)
fpho_input = pn.widgets.FileInput(name = 'Upload FiberPho Data',
                                  accept = '.csv') #File input parameter

#Buttons
upload_button = pn.widgets.Button(name = 'Create Object',
                                  button_type = 'primary', width = 300,
                                  sizing_mode = 'stretch_width')
upload_button.on_click(run_init_fiberobj) #Button action

#Box
init_obj_box = pn.WidgetBox('# Create Object', fpho_input,
                            input_col, upload_button)

# ----------------------------------------------------- # 
# ----------------------------------------------------- # 
#Load fiberobj Widget

#Input variables
#File input parameter
upload_pkl_selecta = pn.widgets.FileInput(name = 'Upload Saved Fiber Objects',
                                          accept = '.pickle', multiple = True) 

#Buttons
upload_pkl_btn = pn.widgets.Button(name = 'Upload Object', 
                                   button_type = 'primary', width = 400, 
                                   sizing_mode = 'stretch_width',
                                   align = 'end')
upload_pkl_btn.on_click(run_upload_fiberobj) #Button action

#Box
load_obj_box = pn.WidgetBox('# Reload saved Fiber Objects',
                            upload_pkl_selecta, upload_pkl_btn)

# ----------------------------------------------------- #

# ----------------------------------------------------- # 
#Delete fiberobj widget

#Input variables
delete_obj_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects',
                                          value = [], options = [])

#Buttons
delete_obj_btn = pn.widgets.Button(name = 'Delete Object',
                                   button_type = 'danger', width = 500,
                                   sizing_mode = 'stretch_width', 
                                   align = 'end')
delete_obj_btn.on_click(run_delete_fiberobj) #Button action

#Box
delete_obj_box = pn.WidgetBox('# Delete Fiber Objects', 
                              delete_obj_selecta, delete_obj_btn)

# ----------------------------------------------------- #

# ----------------------------------------------------- # 
#Save fiberobj Widget

#Input variables
save_obj_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects',
                                          value = [], options = [], )

#Buttons
save_obj_btn = pn.widgets.Button(name = 'Save Object',
                                 button_type = 'primary', width = 400,
                                 sizing_mode = 'stretch_width',
                                 align = 'end')
save_obj_btn.on_click(run_save_fiberobj) #Button action

#Box
save_obj_box = pn.WidgetBox('## Save Fiber Objects for later',
                            save_obj_selecta, save_obj_btn)

# ----------------------------------------------------- #

# ----------------------------------------------------- # 
#Plot raw signal Widget

#Input variables
obj_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects', value = [],
                                     options = [], )

#Buttons
plot_raw_btn = pn.widgets.Button(name = 'Plot Raw Signal',
                                 button_type = 'primary',
                                 width = 200,
                                 sizing_mode = 'stretch_width',
                                 align = 'start')
plot_raw_btn.on_click(run_plot_raw_trace)
clear_raw = pn.widgets.Button(name = 'Clear Plots \u274c',
                              button_type = 'danger', width = 30, 
                              sizing_mode = 'fixed', align = 'start')
clear_raw.on_click(clear_plots)

raw_info = pn.pane.Markdown("""
                                - Plots the raw signal outputs of fiber objects.
                            """, width = 200)

#Box
plot_options = pn.Column(obj_selecta, plot_raw_btn)
plot_raw_widget = pn.WidgetBox(raw_info, plot_options)
plot_raw_card = pn.Card(plot_raw_widget, clear_raw,
                        title = 'Plot Raw Signal',
                        background = 'WhiteSmoke',
                        width = 600, collapsed = True)

# ----------------------------------------------------- # 
#Normalize signal to reference Widget
#Input vairables

norm_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects', value = [],
                                      options = [], )
pick_signal = pn.widgets.Select(name = 'Signal', options = [])
pick_reference = pn.widgets.Select(name = 'Reference', options = [])

#Buttons
norm_sig_btn = pn.widgets.Button(name = 'Normalize Signal',
                                 button_type = 'primary', width = 200,
                                 sizing_mode = 'stretch_width',
                                 align = 'start')
norm_sig_btn.on_click(run_normalize_a_signal)
update_norm_options_btn = pn.widgets.Button(name = 'Update Options',
                                            button_type = 'primary', 
                                            width = 200,
                                            sizing_mode = 'stretch_width',
                                            align = 'start')
update_norm_options_btn.on_click(update_selecta_options)
clear_norm = pn.widgets.Button(name = 'Clear Plots \u274c',
                               button_type = 'danger', width = 30,
                               sizing_mode = 'fixed', align = 'start')
clear_norm.on_click(clear_plots)

norm_info = pn.pane.Markdown("""
                                    - Normalizes the signal and reference trace 
                                    to a biexponential, linearly fits the normalized 
                                    reference to the normalized signal. <br>
                                    Stores all fitted traces in the dataframe and 
                                    plots them for examination.""",
                             width = 200)
#Box
norm_options = pn.Column(norm_selecta, update_norm_options_btn, pick_signal,
                         pick_reference, norm_sig_btn)
norm_sig_widget = pn.WidgetBox(norm_info, norm_options)
norm_sig_card = pn.Card(norm_sig_widget, clear_norm,
                        title = 'Normalize to a reference',
                        background = 'WhiteSmoke',
                        width = 600, collapsed = True)


# ----------------------------------------------------- # 
#Add Behavior Widget

#Input variables
behav_input = pn.widgets.FileInput(name = 'Upload Behavior Data',
                                   accept = '.csv') #File input parameter
behav_selecta = pn.widgets.Select(name = 'Fiber Objects', value = [],
                                  options = [], )
lick_input = pn.widgets.FileInput(name = 'Upload Lickometer Data',
                                  accept = '.csv')

#Buttons
upload_beh_btn = pn.widgets.Button(name = 'Read Behavior Data',
                                   button_type = 'primary', width = 200,
                                   sizing_mode = 'stretch_width',
                                   align = 'start')
upload_beh_btn.on_click(run_import_behavior_data) #Button action
upload_lick_btn = pn.widgets.Button(name = 'Upload', button_type = 'primary',
                                    width = 100, sizing_mode = 'fixed')
upload_lick_btn.on_click(run_convert_lick)

upload_beh_info = pn.pane.Markdown("""
                                        - Imports user uploaded behavior data and reads
                                        dataframe to update and include subject, behavior,
                                        and status columns to the dataframe.
                                    """, width = 200)

convert_info = pn.pane.Markdown(""" - Upload lickometer data to be converted 
                                to behavior file formatting <br> - Returns downloadable
                                csv after conversion has been completed""",
                                width = 200)


#Box
behav_options = pn.Column(upload_beh_info, behav_selecta,
                          behav_input, upload_beh_btn)
lick_options = pn.Column(convert_info, lick_input, upload_lick_btn)
beh_tabs = pn.Tabs(('Behavior Import', behav_options),
                   ('Lick to Boris', lick_options))

upload_beh_widget = pn.WidgetBox(beh_tabs, height = 270)
upload_beh_card = pn.Card(upload_beh_widget, title = 'Import Behavior', 
                          background = 'WhiteSmoke', collapsed = False)


# ----------------------------------------------------- # 

# ----------------------------------------------------- # 
#Add Behavior plot Widget

#Input variables
plot_beh_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects', value = [],
                                          options = [], )
channel_selecta = pn.widgets.MultiSelect(name = 'Signal', value = [],
                                         options = [], )
behavior_selecta = pn.widgets.MultiSelect(name = 'Behavior', value = [],
                                          options = [], )

#Buttons
plot_beh_btn = pn.widgets.Button(name = 'Plot Behavior',
                                 button_type = 'primary', width = 200,
                                 sizing_mode = 'stretch_width',
                                 align = 'start')
plot_beh_btn.on_click(run_plot_behavior) #Button action
update_plot_options_btn = pn.widgets.Button(name = 'Update Options',
                                            button_type = 'primary',
                                            width = 200,
                                            sizing_mode = 'stretch_width',
                                            align = 'start')
update_plot_options_btn.on_click(update_selecta_options) #Button action
clear_beh = pn.widgets.Button(name = 'Clear Plots \u274c',
                              button_type = 'danger',
                              width = 30, sizing_mode = 'fixed',
                              align = 'start')
clear_beh.on_click(clear_plots)

beh_info = pn.pane.Markdown("""
                                - Creates and displays the different channels from behavior data. <br>
                            """, width = 200)
#Box
plot_beh_options = pn.Column(plot_beh_selecta, update_plot_options_btn,
                             channel_selecta, behavior_selecta, plot_beh_btn)
plot_beh_widget = pn.WidgetBox(beh_info, plot_beh_options)
plot_beh_card = pn.Card(plot_beh_widget, clear_beh,
                        title = 'Plot Behavior', background = 'WhiteSmoke',
                        width = 600, collapsed = True)

# ----------------------------------------------------- # 
# ----------------------------------------------------- # 
#Plot Z-Score

#Input variables
zscore_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects', value = [],
                                        options = [], )
zbehs_selecta = pn.widgets.MultiSelect(name = 'Behavior', value = [],
                                       options = [], )
zchannel_selecta = pn.widgets.MultiSelect(name = 'Signal', value = [],
                                          options = [], )
time_before = pn.widgets.IntInput(name = 'Time before event(s)',  width = 50,
                                  placeholder = 'Seconds', value = 2)
time_after = pn.widgets.IntInput(name = 'Time after initiation(s)',
                                 width = 50, placeholder = 'Seconds',
                                 value = 5)
baseline_start = pn.widgets.IntInput(name = 'Baseline Start Time (s)', 
                                     width = 50, placeholder = 'Seconds',
                                     value = 0)
baseline_end = pn.widgets.IntInput(name = 'Baseline End Time (s)', 
                                   width = 50, placeholder = 'Seconds',
                                   value = 0)
first_trace = pn.widgets.IntInput(name = 'Show traces from event number __', 
                                  width = 50, placeholder = "start",
                                  value = -1)
last_trace = pn.widgets.IntInput(name = 'to event number __', 
                                 width = 50, placeholder = "end", value = 0)
show_every = pn.widgets.IntInput(name = 'Show one in every __ traces', 
                                 width = 50, placeholder = "1", value = 1)

z_score_note = pn.pane.Markdown("""
                                   ***Note :***<br>
                                   - Baseline Window Parameters should be kept 0 unless you are using baseline<br> 
                                   z-score computation method. <br>
                                   - The parameters are in seconds. <br>
                                   - Please check where you would like your baseline window <br>
                                   - **ONLY CHECK ONE BOX** <br>
                                   """, width = 200)
zscore_info = pn.pane.Markdown("""
                                    - Takes a dataframe and creates a plot of z-scores for
                                    each time a select behavior occurs with the average
                                    z-score and SEM.
                                """, width = 200)

#Buttons
zscore_btn = pn.widgets.Button(name = 'Zscore of Behavior', 
                               button_type = 'primary', width = 200,
                               sizing_mode = 'stretch_width',
                               align = 'start')
zscore_btn.on_click(run_plot_zscore) #Button action
options_btn = pn.widgets.Button(name = 'Update Options',
                                button_type = 'primary', width = 200,
                                sizing_mode = 'stretch_width',
                                align = 'start')
options_btn.on_click(update_selecta_options) #Button action

baseline_selecta = pn.widgets.CheckBoxGroup(name = 'Baseline Options',
                                            value = [],
                                            options = ['Start of Sample',
                                                       'Before Events',
                                                       'End of Sample'], 
                                            inline = True)
clear_zscore = pn.widgets.Button(name = 'Clear Plots \u274c',
                                 button_type = 'danger', width = 30,
                                 sizing_mode = 'fixed', align = 'start')
clear_zscore.on_click(clear_plots)

#Box
zscore_options = pn.Column(zscore_selecta, options_btn, zchannel_selecta, 
                           zbehs_selecta, time_before, time_after, zscore_btn)
baseline_options = pn.Column(z_score_note, baseline_start,
                             baseline_end, baseline_selecta)
trace_options = pn.Column(first_trace, last_trace, show_every)
tabs = pn.Tabs(('Z-Score', zscore_options),
               ('Baseline Options', baseline_options), 
               ('Reduce Displayed Traces', trace_options))
zscore_widget = pn.WidgetBox(zscore_info, tabs)
zscore_card = pn.Card(zscore_widget, clear_zscore,
                      title = 'Zscore Plot', background = 'WhiteSmoke',
                      width = 600, collapsed = True)

# ----------------------------------------------------- # 
# ----------------------------------------------------- # 
#Pearsons Correlation widget

#Input variables
pearsons_selecta1 = pn.widgets.Select(name = 'Object 1', value = [],
                                      options = [], )
pearsons_selecta2 = pn.widgets.Select(name = 'Object 2', value = [],
                                      options = [], )
channel1_selecta = pn.widgets.Select(name = 'Signal', value = [],
                                     options = [])
channel2_selecta = pn.widgets.Select(name = 'Signal', value = [],
                                     options = [])
pears_start_time = pn.widgets.IntInput(name = 'Start Time', width = 50, 
                                       placeholder = 'Seconds', value = 0)
pears_end_time = pn.widgets.IntInput(name = 'End Time', width = 50,
                                     placeholder = 'Seconds', value = -1)

#Buttons
pearsons_btn = pn.widgets.Button(name = 'Calculate Pearsons Correlation',
                                 button_type = 'primary', width = 200,
                                 sizing_mode = 'stretch_width',
                                 align = 'start')
pearsons_btn.on_click(run_pearsons_correlation) #Button action
pearson_options_btn = pn.widgets.Button(name = 'Update Options',
                                        button_type = 'primary', width = 200,
                                        sizing_mode = 'stretch_width',
                                        align = 'start')
pearson_options_btn.on_click(update_selecta_options) #Button action
clear_pears = pn.widgets.Button(name = 'Clear Plots \u274c',
                                button_type = 'danger', width = 30,
                                sizing_mode = 'fixed', align = 'start')
clear_pears.on_click(clear_plots)

pears_info = pn.pane.Markdown("""
                                    - Takes in user chosen objects and channels then returns the Pearson's correlation coefficient and   
                                    plots the signals. <br>
                                """, width = 200)

#Box
pearson_row1  = pn.Row(pearsons_selecta1, pearsons_selecta2)
pearson_row2  = pn.Row(channel1_selecta, channel2_selecta)
pearson_row3  = pn.Row(pears_start_time, pears_end_time)
pearson_widget = pn.WidgetBox('# Pearons Correlation Plot', pears_info,
                              pearson_row1, pearson_options_btn, pearson_row2,
                              pearson_row3, pearsons_btn)
pearsons_card = pn.Card(pearson_widget, clear_pears,
                        title = 'Pearsons Correlation Coefficient',
                        background = 'WhiteSmoke', width = 600,
                        collapsed = True)


# ----------------------------------------------------- # 
# ----------------------------------------------------- # 
#Behavior specific pearsons widget

#Input variables
beh_corr_selecta1 = pn.widgets.Select(name = 'Object 1', value = [],
                                      options = [], )
beh_corr_selecta2 = pn.widgets.Select(name = 'Object 2', value = [],
                                      options = [], )
beh_corr_channel_selecta = pn.widgets.MultiSelect(name = 'Signal',
                                                  value = [], options = [], )
beh_corr_behavior_selecta = pn.widgets.MultiSelect(name = 'Behavior',
                                                   value = [], options = [], )

#Buttons
beh_corr_btn = pn.widgets.Button(name = 'Calculate Pearsons Correlation',
                                 button_type = 'primary', width = 200,
                                 sizing_mode = 'stretch_width',
                                 align = 'start')
beh_corr_btn.on_click(run_beh_specific_pearsons) #Button action
beh_corr_options_btn = pn.widgets.Button(name = 'Update Options',
                                         button_type = 'primary', width = 200,
                                         sizing_mode = 'stretch_width',
                                         align = 'start')
beh_corr_options_btn.on_click(update_selecta_options) #Button action
clear_beh_corr = pn.widgets.Button(name = 'Clear Plots \u274c',
                                   button_type = 'danger', width = 30,
                                   sizing_mode = 'fixed', align = 'start')
clear_beh_corr.on_click(clear_plots)

beh_corr_info = pn.pane.Markdown("""
                                    - Takes in user chosen objects, channels and behaviors to calculate the behavior specific Pearson’s 
                                    correlation and plot the signals. <br>
                                """, width = 200)

#Box
beh_corr_options = pn.Column(beh_corr_selecta1, beh_corr_selecta2,
                             beh_corr_options_btn, beh_corr_channel_selecta,
                             beh_corr_behavior_selecta, beh_corr_btn)
beh_corr_widget = pn.WidgetBox(beh_corr_info, beh_corr_options)
beh_corr_card = pn.Card(beh_corr_widget, clear_beh_corr,
                        title = 'Behavior Specific Pearsons Correlation',
                        background = 'WhiteSmoke', width = 600,
                        collapsed = True)


# ----------------------------------------------------- # 
# ----------------------------------------------------- # 
#Download Results widget

#Input variables
output_name = pn.widgets.TextInput(name = 'Output filename', width = 90,
                                   placeholder = 'String')
results_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects', value = [],
                                         options = [])
result_type_selecta= pn.widgets.MultiSelect(name = 'Result Types', value = [],
                                            options = ['Zscore Results',
                                                       'Correlation Results',
                                                       'Behavior Specific Correlation Reuslts'])

#Buttons
download_results_btn = pn.widgets.Button(name = 'Download',
                                 button_type = 'primary', width = 200,
                                 sizing_mode = 'stretch_width',
                                 align = 'start')

download_results_btn.on_click(run_download_results) #Button action

#Box
download_results_widget = pn.WidgetBox('# Download Results', output_name,
                                       results_selecta, result_type_selecta,
                                       download_results_btn)
download_results_card = pn.Card(download_results_widget, clear_pears,
                                title = 'Download Results',
                                background = 'WhiteSmoke', width = 600,
                                collapsed = True)

# ----------------------------------------------------- # 
# ----------------------------------------------------- # 
#Object info widget

#Table
info_table = pn.widgets.Tabulator(fiber_data, height = 270, 
                                  page_size = 10, disabled = True)

obj_info_card = pn.Card(info_table, title = "Display Object Attributes", 
                        background = 'WhiteSmoke', collapsed = False)

# ----------------------------------------------------- # 
# Accent Colors
ACCENT_COLOR_HEAD = "#D9F3F3"
ACCENT_COLOR_BG = "#128CB6"

# Material Template
material = pn.template.MaterialTemplate(
    site = 'Donaldson Lab: Fiber Photometry', 
    title = 'FiberPho GUI',
    header_color = ACCENT_COLOR_HEAD,
    header_background = ACCENT_COLOR_BG)


# Append widgets to Material Template
material.sidebar.append(pn.pane.Markdown(
    "** Upload your photometry data *(.csv)* ** and set your fiber object's **attributes** here"))
material.sidebar.append(init_obj_box)
material.sidebar.append(load_obj_box)
material.sidebar.append(save_obj_box)
material.sidebar.append(delete_obj_box)

material.main.append(pn.Row(upload_beh_card, obj_info_card))
material.main.append(plot_raw_card)
material.main.append(norm_sig_card)
material.main.append(plot_beh_card)
material.main.append(zscore_card)
material.main.append(pearsons_card)
material.main.append(beh_corr_card)
material.main.append(download_results_card)
material.main.append(log_card)

material.servable()