from os import error
import sys
import argparse
import pandas as pd
import numpy as np
import csv
import pickle
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
from pathlib import Path
import panel as pn
from statistics import mean
import matplotlib.pyplot as plt
import scipy.stats as ss
import re

pn.extension('terminal')

def lick_to_boris(lick_file):
    """
    Converts lickometer data to a BORIS file that is readable by the GUI

    Parameters
    ----------
    lick_file : file
        uploaded lickometer file

    Returns
    ----------
    boris : Dataframe
        converted data for download
    """
    trimmed = lick_file[lick_file['Licks'] != 0]

    starts = [(trimmed.iloc[0]['Time'] - lick_file.iloc[0]['Time']) / 1000]
    stops = []
    diffs = np.diff(trimmed.index)

    for i, v in enumerate(diffs):
        if v > 500:
            stops.append((trimmed.iloc[i]['Time'] 
                          - lick_file.iloc[0]['Time']) / 1000)
            if i+1 < len(diffs):
                starts.append((trimmed.iloc[i+1]['Time'] 
                               - lick_file.iloc[0]['Time']) / 1000)
    stops.append((trimmed.iloc[-1]['Time'] 
                  - lick_file.iloc[0]['Time']) / 1000)

    time = starts + stops
    time.sort()

    status = ['START'] * len(time)
    half = len(time) / 2
    status[1::2] = ['STOP'] * int(half)
    behavior = ['Lick'] * len(time)


    time = [0]*14 + ['Time'] + time
    media = ['n/a'] * len(time)
    total = ['n/a'] * len(time)
    FPS = ['n/a'] * len(time)
    subject = ['n/a'] * len(time)
    behavior = [0]*14 + ['Behavior'] + behavior
    beh_cat = ['n/a'] * len(time)
    comment = ['n/a'] * len(time)
    status = [0]*14 + ['Status'] + status

    boris = pd.DataFrame([time, media, total, FPS, subject, behavior, beh_cat,
                          comment, status])
    boris = boris.transpose()
    return boris


class fiberObj:
    
    """
    A class to represent a fiber object for fiber photometry and behavior analysis.

    Attributes
    ----------
    obj_name : str
        Name of the fiber object
    
    fiber_num : int
        Fiber number used in photometry data (ranges from 0 - 2 as of now)
        
    animal_num : int
        The animal number used in the experiment
        
    exp_date : Date-string (MM/DD)
        Date of the captured photometry recording
    
    exp_time : Time (Hr/Min)
        Time of the captured photometry recording
        
    start_time : int
        Time to exclude from beginning of recording
    
    stop_time : int
        Time to stop at from start of recording 
        
    file_name : str
        File name of the uploaded photometry data
    
    beh_file : Dataframe
        Stores a pandas dataframe of the behavior recording
        
    beh_filename : str
        Name of the behavior dataset
        
    behaviors : set
        Stores unique behaviors of fiber object
    
    channels : set
        Stores the signals used in photometry data
        
    fpho_data_dict : dict
        Stores photometry data into a dictionary
        
    fpho_data_df : Dataframe
        Uses fpho_data_dict to convert photometry data into a pandas dataframe for use
        
    color_dict : dict
        stores translated channel labels
    
    z_score_results : Dataframe
        stores results of Z-Score computations
    
    correlation_results : Dataframe
        stores results of Pearsons computations
    
    beh_corr_results : Dataframe
        stores results of behavior specific Pearsons computations
        
    frame_rate : List ???
        calculates frame rate of captured data
        
    """
    def __init__(self, file, obj, fiber_num, animal, exp_date,
                 exp_start_time, start_time, stop_time, filename):
        """
        Constructs all the necessary attributes for the FiberPho object. Holds all 
        the data from a fiber experiment as well as some results from analysis. 
        Takes in a fiber photometry file (.csv) and parses it into a dataframe 
        (fpho_data_df) with 9 columns: time_iso, time_green, time_red, 
        green_iso, green_green, green_red, red_iso, red_green, red_red.

        Parameters
        ----------
        obj_name : str
            name of the fiber object

        fiber_num : int
            fiber number to analyze (range: 0-2)

        animal_num : int
            the animal number used in the experiment

        exp_date : Date-string (MM/DD), optional
            date of the captured photometry recording

        exp_time : Time (Hr/Min), optional
            time of the captured photometry recording

        start_time : int
            time to exclude from beginning of recording

        stop_time : int
            time to stop at from start of recording 

        file_name : str
            file name of the uploaded photometry file

        Returns
        ----------
        class object : fiberObj
            Initialized object of type fiberObj
        """
        
        self.obj_name = obj
        self.fiber_num = fiber_num
        self.animal_num = animal
        self.exp_date = exp_date
        self.exp_start_time = exp_start_time
        self.start_time = start_time #looking for better names
        self.stop_time = stop_time #looking for better names
        self.file_name = filename
        self.beh_file = None
        self.beh_filename = 'NaN'
        self.behaviors = set()
        self.channels = set()
        self.color_dict = {'Raw_Green' : 'LawnGreen', 'Raw_Red': 'Red', 'Raw_Isosbestic': 'Cyan',
                           'Green_Normalized': 'MediumSeaGreen', 'Red_Normalized': 'Dark_Red',
                           'Isosbestic_Normalized':'DeepSkyBlue'}
        self.z_score_results = pd.DataFrame(columns = ['Object Name', 'Behavior', 'Channel', 'delta Z_score',
                                                       'Max Z_score', 'Max Z_score Time',
                                                       'Min Z_score', 'Min Z_score Time',
                                                       'Average Z_score Before', 'Average Z_score After',
                                                       'Time Before', 'Time After', 'Number of events',
                                                       'Z_score Baseline'])
        self.correlation_results = pd.DataFrame(columns = ['Object Name', 'Channel', 'Obj2', 'Obj2 Channel',
                                                           'start_time', 'end_time', 
                                                           'R Score', 'p score'])
        self.beh_corr_results = pd.DataFrame(columns = ['Object Name', 'Channel', 'Obj2', 'Obj2 Channel',
                                                        'Behavior', 'Number of Events' 
                                                        'R Score', 'p score'])
        file['Timestamp'] = (file['Timestamp'] - file['Timestamp'][0])
        
        self.frame_rate = (file['Timestamp'].iloc[-1]
                            - file['Timestamp'][0])/len(file['Timestamp'])
        
        if start_time == 0:
            self.start_idx = 0
        else:
            self.start_idx = np.searchsorted(file['Timestamp'], start_time)
        
        if stop_time == -1:
            self.stop_idx = len(file['Timestamp'])
        else:
            self.stop_idx = np.searchsorted(file['Timestamp'], stop_time) 
        
        time_slice = file.iloc[self.start_idx : self.stop_idx]
        
        data_dict = {}
        
        #Check for green ROI
        try: 
            test_green = file.columns.str.endswith('G')
        except:
            green_ROI = False
            print('no green ROI found')
        else:
            green_ROI = True
            green_col = np.where(test_green)[0][self.fiber_num - 1]
            
        #Check for red ROI   
        try: 
            test_red = file.columns.str.endswith('R')
        except:
            red_ROI = False
            print('no green ROI found')
        else:
            red_ROI = True
            red_col = np.where(test_red)[0][self.fiber_num - 1]
        
        led_states = file['LedState'][2:8].unique()
        npm_dict = {2: 'Green', 1: 'Isosbestic', 4: 'Red'}
        
        for color in led_states:
            if color == 1 or color == 2 or color == 4:
                data_dict['time_' + npm_dict[color]] =  time_slice[
                    time_slice['LedState'] == color]['Timestamp'].values.tolist()
                
            if green_ROI: 
                if color == 1 or color == 2:
                    data_dict['Raw_' + npm_dict[color]] = time_slice[
                        time_slice['LedState'] == color].iloc[:, green_col].values.tolist()
                    self.channels.add('Raw_' + npm_dict[color])
            
            if red_ROI: 
                if color == 4:
                    data_dict['Raw_' + npm_dict[color]] = time_slice[
                        time_slice['LedState'] == color].iloc[:, red_col].values.tolist() 
                    self.channels.add('Raw_' + npm_dict[color])
            
        shortest_list = min([len(data_dict[ls]) for ls in data_dict])
        
        for ls in data_dict:
            data_dict[ls] = data_dict[ls][:shortest_list-1]
        
        self.fpho_data_df = pd.DataFrame.from_dict(data_dict)
        
    
#### Helper Functions ####
    def fit_exp(self, values, a, b, c, d, e):
        """
        Transforms data into an exponential function
        of the form 
        ..math:: 
            y = A * exp(-B * X) + C * exp(-D * x) + E

        Parameters
        ----------
        values : list
            data

        a, b, c, d, e : int/float
            estimates for the parameter values of A, B, C, D and E
        """
        values = np.array(values)

        return a * np.exp(-b * values) + c * np.exp(-d * values) + e

    def lin_fit(self, values, a, b):
        """
        Linearly transforms data to fit ...

        Parameters
        ----------
        values : list
            data

        a, b : integers or floats
            estimates for the parameter values of A, B

        """
        values = np.array(values)

        return a * values + b

#### End Helper Functions #### 
    

    
##### Class Functions #####

    #Signal Trace function
    def raw_signal_trace(self):
        """
        Creates and displays graphs of a fiber object's signals.

        Parameters
        ----------
        None

        Returns
        -------
        fig : plotly.graph_objects.Scatter
            Plot of raw signal traces
        """

        fig = make_subplots(rows = 1, cols = 1, shared_xaxes = True,
                            vertical_spacing = 0.02, x_title = "Time (s)",
                            y_title = "Fluorescence")
        for channel in self.channels:
            fig.add_trace(
                go.Scatter(
                    x = self.fpho_data_df['time' + channel[3:]],
                    y = self.fpho_data_df[channel],
                    mode = "lines",
                    line = go.scatter.Line(color = self.color_dict[channel]),
                    name = channel,
                    text = channel,
                    showlegend = True
                ), row = 1, col = 1
            )
        fig.update_layout(
            title = self.obj_name + ' Raw Data'
        )
        return fig
        # fig.write_html(self.obj_name+'raw_sig.html', auto_open = True)

    #Plot fitted exp function
    def normalize_a_signal(self, signal, reference):
        print('Does this work')
        """
        Creates a plot normalizing 1 fiber data to an
        exponential of the form y=A*exp(-B*X)+C*exp(-D*x)

        Parameters
        ----------
        signal : string
                user channel selection
        reference : string
                user reference trace selection
        Returns
        --------
        output_filename_f1GreenNormExp.png
        & output_filename_f1RedNormExp.png: png files
        containing the normalized plot for each fluorophore
        """
        # Get coefficients for normalized fit using first guesses
        # for the coefficients - B and D (the second and fourth
        # inputs for p0) must be negative, while A and C (the
        # first and third inputs for p0) must be positive
        
        # Channels={'Green':'Raw_Green', 'Red':'Raw_Red', 'Isosbestic':'Raw_Isosbestic'}
        #times = {'Green':'time_green', 'Red':'time_red', 'Isosbestic':'time_iso'}   
        # time = self.fpho_data_df[times[signal]]
        time = self.fpho_data_df['time_Green']
        sig = self.fpho_data_df[signal]
        ref = self.fpho_data_df[reference]
        popt, pcov = curve_fit(self.fit_exp, time, sig, p0 = (1.0, 0, 1.0, 0, 0),
                               bounds = (0, np.inf))

        AS = popt[0]  # A value
        BS = popt[1]  # B value
        CS = popt[2]  # C value
        DS = popt[3]  # D value
        ES = popt[4]  # E value

        popt, pcov = curve_fit(self.fit_exp, time, ref, p0=(1.0, 0, 1.0, 0, 0),
                               bounds = (0,np.inf))

        AR = popt[0]  # A value
        BR = popt[1]  # B value
        CR = popt[2]  # C value
        DR = popt[3]  # D value
        ER = popt[4]  # E value     

        # Generate fit line using calculated coefficients
        fitSig = self.fit_exp(time, AS, BS, CS, DS, ES)
        fitRef = self.fit_exp(time, AR, BR, CR, DR, ER)

        sigRsquare = np.corrcoef(sig, fitSig)[0,1] ** 2
        refRsquare = np.corrcoef(ref, fitRef)[0,1] ** 2
        print('sig r^2 =', sigRsquare ,'ref r^2 =', refRsquare)

        if sigRsquare < .01:
            print('sig r^2 =', sigRsquare)
            print('No exponential decay was detected in ', signal)
            print(signal + ' expfit is now the median of ', signal)
            AS = 0
            BS = 0
            CS = 0
            DS = 0
            ES = np.median(sig)
            fitSig = self.fit_exp(time, AS, BS, CS, DS, ES)

        if refRsquare < .001:
            print('ref r^2 =', refRsquare)
            print('No exponential decay was detected in ', reference)
            print(reference + ' expfit is now the median  ', reference)
            AR = 0
            BR = 0
            CR = 0
            DR = 0
            ER = np.median(ref)
            fitRef = self.fit_exp(time, AR, BR, CR, DR, ER)

        normed_sig = [(k / j) for k,j in zip(sig, fitSig)]
        normed_ref = [(k / j) for k,j in zip(ref, fitRef)]      

        popt, pcov = curve_fit(self.lin_fit, normed_sig, normed_ref,
                               bounds = ([0, -5], [np.inf, 5]))
        AL = popt[0]
        BL = popt[1]

        adjusted_ref=[AL * j + BL for j in normed_ref]
        normed_to_ref=[(k / j) for k,j in zip(normed_sig, adjusted_ref)]

        # below saves all the variables we generated to the df #
        #  data frame inside the obj ex. self 
        # and assign all the long stuff to that
        # assign the AS, BS,.. etc and AR, BR, etc to lists called self.sig_fit_coefficients, self.ref_fit_coefficients and self.sig_to_ref_coefficients
        self.fpho_data_df.loc[:, signal + ' expfit'] = fitSig
        self.sig_fit_coefficients = ['A= ' + str(AS), 'B= ' + str(BS), 'C= ' 
                                     + str(CS), 'D= ' + str(DS), 'E= ' + str(ES)]
        self.fpho_data_df.loc[:, signal + ' normed to exp']=normed_sig
        self.fpho_data_df.loc[:, reference + ' expfit']=fitRef
        self.ref_fit_coefficients = ['A= ' + str(AR), 'B= ' + str(BR), 'C= ' 
                                     + str(CR), 'D= ' + str(DR), 'E= ' + str(ER)]
        self.fpho_data_df.loc[:, reference + ' normed to exp']=normed_ref
        self.fpho_data_df.loc[:,reference + ' fitted to ' + signal]=adjusted_ref
        self.sig_to_ref_coefficients = ['A= ' + str(AL), 'B= ' + str(BL)]
        self.fpho_data_df.loc[:,signal[4:] + '_Normalized'] = normed_to_ref
        self.channels.add(signal[4:] + '_Normalized')

        fig = make_subplots(rows = 3, cols = 2, x_title = 'Time(s)',
                    subplot_titles=("Biexponential Fitted to Signal",
                                    "Signal Normalized to Biexponential",
                                    "Biexponential Fitted to Ref", 
                                    "Reference Normalized to Biexponential",
                                    "Reference Linearly Fitted to Signal",
                                    "Final Normalized Signal"),
                    shared_xaxes = True, vertical_spacing = 0.1)
        fig.add_trace(
            go.Scatter(
            x = time,
            y = sig,
            mode = "lines",
            line = go.scatter.Line(color="rgba(0, 255, 0, 1)"),
            name ='Signal:' + signal,
            text = 'Signal',
            showlegend = True),
            row = 1, col = 1
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[signal + ' expfit'],
            mode = "lines",
            line = go.scatter.Line(color="Purple"),
            name = 'Biexponential fitted to Signal',
            text = 'Biexponential fitted to Signal',
            showlegend = True),
            row = 1, col = 1
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[signal + ' normed to exp'],
            mode = "lines",
            line = go.scatter.Line(color="rgba(0, 255, 0, 1)"),
            name = 'Signal Normalized to Biexponential',
            text = 'Signal Normalized to Biexponential',
            showlegend = True),
            row = 1, col = 2
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = ref,
            mode = "lines",
            line = go.scatter.Line(color="Cyan"),
            name = 'Reference:' + reference,
            text = 'Reference',
            showlegend = True),
            row = 2, col = 1
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[reference + ' expfit'],
            mode = "lines",
            line = go.scatter.Line(color="Purple"),
            name = 'Biexponential fit to Reference',
            text = 'Biexponential fit to Reference',
            showlegend = True),
            row = 2, col = 1
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[reference + ' normed to exp'],
            mode = "lines",
            line = go.scatter.Line(color="Cyan"),
            name = 'Reference Normalized to Biexponential',
            text = 'Reference Normalized to Biexponential',
            showlegend = True),
            row = 2, col = 2
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[reference + ' fitted to ' + signal],
            mode = "lines",
            line = go.scatter.Line(color="Cyan"),
            name = 'Reference linearly scaled to signal',
            text = 'Reference linearly scaled to signal',
            showlegend = True),
            row = 3, col = 1  
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[signal + ' normed to exp'],
            mode = "lines",
            line = go.scatter.Line(color="rgba(0, 255, 0, 0.5)"),
            name = 'Signal Normalized to Biexponential',
            text = 'Signal Normalized to Biexponential',
            showlegend = True),
            row = 3, col = 1
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[signal[4:] + '_Normalized'],
            mode="lines",
            line = go.scatter.Line(color = "Hot Pink"), 
            name = 'Final Normalized Signal',
            text = 'Final Normalized Signal',
            showlegend = True), 
            row = 3, col = 2
            )
        fig.update_layout(
            title = "Normalizing " + signal + ' for ' + self.obj_name
            )
        return fig
    

    # ----------------------------------------------------- # 
    # Behavior Functions
    # ----------------------------------------------------- # 

    def import_behavior_data(self, BORIS_filename, filename):
        """
        Takes a file name and returns a dataframe of parsed data

        Parameters
        ----------
        BORIS_filename : string
            The path to the CSV file

        Returns
        --------
        behaviorData : pandas dataframe
            contains - Time(total msec), Time(sec), 
            Subject, Behavior, Status
        """
        
        # Open file, catch errors
        try:
            BORIS_data = pd.read_csv(BORIS_filename, header=15)  # starts at data
        except FileNotFoundError:
            print("Could not find file: " + BORIS_filename)
            sys.exit(1)
        except PermissionError:
            print("Could not access file: " + BORIS_filename)
            sys.exit(2)

        unique_behaviors = BORIS_data['Behavior'].unique()
        for beh in unique_behaviors:
            self.behaviors.add(beh)
            idx_of_beh = [i for i in range(len(BORIS_data['Behavior']
                         )) if BORIS_data.loc[i, 'Behavior'] == beh]             
            j = 0
            self.fpho_data_df[beh] = ' '
            while j < len(idx_of_beh):
                if BORIS_data.loc[(idx_of_beh[j]), 'Status']=='POINT': 
                    point_idx=self.fpho_data_df['time_Green'].searchsorted(
                        BORIS_data.loc[idx_of_beh[j],'Time'])
                    self.fpho_data_df.loc[point_idx, beh]='S'
                    j = j + 1
                elif (BORIS_data.loc[(idx_of_beh[j]), 'Status']=='START' and 
                      BORIS_data.loc[(idx_of_beh[j + 1]), 'Status']=='STOP'):
                    startIdx = self.fpho_data_df['time_Green'].searchsorted(
                        BORIS_data.loc[idx_of_beh[j],'Time'])
                    endIdx = self.fpho_data_df['time_Green'].searchsorted(
                        BORIS_data.loc[idx_of_beh[j + 1],'Time'])
                    if endIdx < len(self.fpho_data_df['time_Green']) and startIdx > 0:
                        self.fpho_data_df.loc[startIdx, beh] = 'S'
                        self.fpho_data_df.loc[startIdx+1 : endIdx-1, beh] = 'O'
                        self.fpho_data_df.loc[endIdx, beh] = 'E'
                    j = j + 2
                else: 
                    print("\nStart and stops for state behavior:" 
                          + beh + " are not paired correctly.\n")
                    sys.exit()
        self.beh_file = BORIS_filename
        self.beh_filename = filename
        return

    def plot_behavior(self, behaviors, channels):
        """
        Plots behavior specific signals
        
        Parameters
        ----------
        behaviors : string
            user selected behaviors
        
        channels : string
            user selected channels
            
        Returns
        ----------
        fig : plotly.graph_objects.Scatter
            Scatter plot of select behavior signals
        """
        
        fig = make_subplots(rows = len(channels), cols = 1,
                            subplot_titles = [channel for channel in channels],
                            shared_xaxes = True)
        for i, channel in enumerate(channels):
            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_Green'],
                y = self.fpho_data_df[channel],
                mode = "lines",
                line = go.scatter.Line(color = "Green"),
                name = channel,
                showlegend = False), row = i + 1, col = 1
                )
            colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                      '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
            j = 0
            behaviorname = ""
            for j, beh in enumerate(behaviors):
                behaviorname = behaviorname + " " + beh
                temp_beh_string = ''.join([key for key in self.fpho_data_df[beh]])
                pattern = re.compile(r'S[O]+E')
                bouts = pattern.finditer(temp_beh_string)
                for bout in bouts:
                    start_time = self.fpho_data_df.at[bout.start(), 'time_Green']
                    end_time = self.fpho_data_df.at[bout.end(), 'time_Green']
                    fig.add_vrect(x0 = start_time, x1 = end_time, 
                                opacity = 0.75,
                                layer = "below",
                                line_width = 1, 
                                fillcolor = colors[j % 10],
                                row = i + 1, col = 1,
                                name = beh
                                )
                S = re.compile(r'S')
                starts = S.finditer(temp_beh_string)
                for start in starts:
                    start_time = self.fpho_data_df.at[start.start(), 'time_Green']
                    fig.add_vline(x = start_time, 
                                layer = "below",
                                line_width = 3, 
                                line_color = colors[j % 10],
                                row = i + 1, col = 1,
                                name = beh
                                )
                
                fig.add_annotation(xref = "x domain", yref = "y domain",
                    x = 1, 
                    y = (j + 1) / len(self.behaviors),
                    text = beh,
                    bgcolor = colors[j % 10],
                    showarrow = False,
                    row = i + 1, col = 1
                    )
                fig.update_layout(title = behaviorname + ' for ' + self.obj_name)
        return fig
        
    
    def plot_zscore(self, channel, beh, time_before, time_after,
                    baseline = 0, base_option = 0, show_first = -1,
                    show_last = 0, show_every = 1):
        
        """
        Takes a dataframe and creates plot of z-scores for
        each time a select behavior occurs with the avg
        z-score and SEM. Stores results in dataframe
    
        Parameters
        ----------
        channel : string
            user selected channels
        
        beh : string
            user selected behaviors
        
        time_before : int
            timestamps to include before event
            
        time_after : int
            timestamps to include after start of event
        
        baseline : list, optional
            baseline window start and end times [0, 1] respectively
        
        base_option : int, optional
            baseline parameter options - start of sample, before events, end of sample
        
        show_first : int, optional
            show traces from event number [int]
        
        show_last : int, optional
            show traces up to event number [int]
            
        show_every : int, optional
            show one in every [int] traces
        
        Returns
        ----------
        fig : plotly.graph_objects.Scatter
            Plot of z-scores for select behaviors
        """
        
        # Finds all times where behavior starts, turns into list
        beh_times = list(self.fpho_data_df[(
            self.fpho_data_df[beh]=='S')]['time_Green'])
        # Initialize figure
        fig = make_subplots(rows = 1, cols = 2,
                            subplot_titles = ('Full trace with events',
                                              'Average'
                                             )
                           )
        # Adds trace
        fig.add_trace(
            # Scatter plot
            go.Scatter(
            # X = all times
            # Y = all values at that channel
            x = self.fpho_data_df['time_Green'],
            y = self.fpho_data_df[channel],
            mode = "lines",
            line = go.scatter.Line(color="Green"),
            name = channel,
            showlegend = False), 
            row = 1, col =1
            )

        # Initialize array of zscore sums
        Zscore_sum = []
        # Initialize events counter to 0
        n_events = 0
        
        if not base_option:
            base_mean = None
            base_std = None
            zscore_baseline = 'Each event'
        
        elif base_option[0] == 'Start of Sample':
            # idx = np.where((start_event_time > baseline[0]) & (start_event_time < baseline[1]))
            # Find baseline start/end index
            # Start event time is the first occurrence of event, this option will be for a baseline at the beginning of the trace
            base_start_idx = self.fpho_data_df['time_Green'].searchsorted(
                baseline[0])
            base_end_idx = self.fpho_data_df['time_Green'].searchsorted(
                baseline[1])
            # Calc mean and std for values within window
            base_mean = np.nanmean(self.fpho_data_df.loc[
                base_start_idx:base_end_idx, channel]) 
            base_std = np.nanstd(self.fpho_data_df.loc[
                base_start_idx:base_end_idx, channel])
            zscore_baseline = 'From ' + str(baseline[0]) + ' to ' + str(baseline[1])
            
        
        elif base_option[0] == 'End of Sample':
            # Indexes for finding baseline at end of sample
            start = max(baseline)
            end = min(baseline)
            end_time = self.fpho_data_df['time_Green'].iloc[-1]
            base_start_idx = self.fpho_data_df['time_Green'].searchsorted(
                end_time - start)
            base_end_idx = self.fpho_data_df['time_Green'].searchsorted(
                end_time - end)
            # Calculates mean and standard deviation
            base_mean = np.nanmean(self.fpho_data_df.loc[
                base_start_idx:base_end_idx, channel])
            base_std = np.nanstd(self.fpho_data_df.loc[
                base_start_idx:base_end_idx, channel])
            zscore_baseline = 'From ' + str(end_time - start) + ' to ' + str(end_time - end)


        # Loops over all start times for this behavior
        # i = index, time = actual time
        for i, time in enumerate(beh_times):
            # Calculates indices for baseline window before each event
            if base_option and base_option[0] == 'Before Events':
                start = max(baseline)
                end = min(baseline)
                base_start_idx = self.fpho_data_df['time_Green'].searchsorted(
                    time - start)
                base_end_idx = self.fpho_data_df['time_Green'].searchsorted(
                    time - end)
                base_mean = np.nanmean(self.fpho_data_df.loc[
                    base_start_idx:base_end_idx, channel])
                base_std = np.nanstd(self.fpho_data_df.loc[
                    base_start_idx:base_end_idx, channel])

            # time - time_Before = start_time for this event trace, time is the actual event start, time before is secs input before event start
            # Finds time in our data that is closest to time - time_before
            # start_idx = index of that time
            start_idx = self.fpho_data_df['time_Green'].searchsorted(
                time - time_before)
            # time + time_after = end_time for this event trace, time is the actual event start, time after is secs input after event start
            # end_idx = index of that time
            end_idx = self.fpho_data_df['time_Green'].searchsorted(
                time + time_after)
            
            # Edge case: If indexes are within bounds
            if (start_idx > 0 and 
                end_idx < len(self.fpho_data_df['time_Green']) - 1):
                # Finds usable events
                n_events = n_events + 1
                # Tempy stores channel values for this event trace
                trace = self.fpho_data_df.loc[
                    start_idx : end_idx, channel].values.tolist()
                this_Zscore=self.zscore(trace, base_mean, base_std)
                if len(Zscore_sum)>1:
                    # Sums values at each index
                    Zscore_sum = [Zscore_sum[i] + this_Zscore[i] 
                                 for i in range(len(trace))]
                else:
                    # First value
                    Zscore_sum = this_Zscore

                if show_first == -1 or i in np.arange(show_first, show_last, show_every):
                    # Times for this event trace
                    x = self.fpho_data_df.loc[start_idx : end_idx, 'time_Green']
                    # Trace color (First event blue, last event red)
                    trace_color = 'rgb(' + str(
                        int((i+1) * 255/(len(beh_times)))) + ', 0, 255)'
                    # Adds a vertical line for each event time
                    fig.add_vline(x = time, line_dash = "dot", row = 1, col = 1)
                    # Adds trace for each event
                    fig.add_trace(
                        # Scatter plot
                        go.Scatter( 
                        # Times starting at user input start time, ending at user input end time
                        x = x - time,
                        # Y = Zscore of event trace
                        # y = ss.zscore(self.fpho_data_df.loc[start_idx:end_idx,channel]),
                        y = this_Zscore, 
                        mode = "lines",
                        line = dict(color = trace_color, width = 2),
                        name = 'Event:' + str(i),
                        text = 'Event:' + str(i),
                        showlegend=True), 
                        row = 1, col = 2
                        )
        avg_Zscore = [i / n_events for i in Zscore_sum]  
        graph_time = np.linspace(-time_before, time_after, num = len(x))
        zero_idx = np.searchsorted(graph_time, 0)
        fig.add_vline(x = 0, line_dash = "dot", row = 1, col = 2)
        # Adds trace
        fig.add_trace(
            # Scatter plot
            go.Scatter( 
            # Times for baseline window
            x = graph_time,
            # Y = Zscore average of all event traces
            y = avg_Zscore,
            mode = "lines",
            line = dict(color = "Black", width = 5),
            name = 'average',
            text = 'average',
            showlegend = True),
            row = 1, col = 2
            )
        fig.update_layout(
            title = 'Z-score of ' + beh + ' for ' 
                    + self.obj_name + ' in channel ' + channel
            )      
        results = {'Object Name': self.obj_name, 'Behavior': beh, 'Channel' : channel,
                   'Max Z_score' : max(avg_Zscore),
                   'Max Z_score Time' : graph_time[np.argmax(avg_Zscore)], 
                   'Min Z_score': min(avg_Zscore), 
                   'Min Z_score Time' : graph_time[np.argmin(avg_Zscore)],
                   'delta Z_score' : max(avg_Zscore) - min(avg_Zscore), 
                   'Average Z_score Before' : np.mean(avg_Zscore[:zero_idx]),
                   'Average Z_score After' : np.mean(avg_Zscore[zero_idx:]),
                   'Time Before':time_before, 'Time After':time_after,
                   'Number of events' : n_events, 'Z_score Baseline' : zscore_baseline}
        print(results)
        self.z_score_results = self.z_score_results.append(results, ignore_index = True)
        return fig
        
        
    # Zscore calc helper
    def zscore(self, ls, mean = None, std = None):
        """
        Helper function to calculate z-scores

        Parameters
        ----------
        ls : list
            list of trace signals

        mean : int/float, optional
            baseline mean value

        std : int/float, optional
            baseline standard deviation value

        Returns
        ----------
        new_ls : list
            list of calculated z-scores per event
        """
        # Default Params, no arguments passed
        if mean is None and std is None:
            mean = np.nanmean(ls)
            std = np.nanstd(ls)
        # Calculates zscore per event in list  
        new_ls = [(i - mean) / std for i in ls]
        return new_ls
        
        
        
        
        
         #return the pearsons correlation coefficient and r value between 2 full channels and plots the signals overlaid and their scatter plot
    def pearsons_correlation(self, obj2, channel1, channel2, start_time, end_time):
        """
        Takes in user chosen objects and channels then returns the 
        Pearson’s correlation coefficient and plots the signals. 

        Parameters
        ----------
        obj2 : fiber object
            second object for correlation analysis
        
        channel1 : string
            first object's selected signal for analysis
        
        channel2 : string
            second object's selected signal for analysis
        
        start_time : int
            starting timestamp of data
        
        end_time : int
            ending timestamp of data
            
        Returns
        ----------
        fig : plotly.graph_objects.Scatter
            Plot of signals based on correlation results
        """
        
        #find start
        if np.round(self.frame_rate) != np.round(self.frame_rate):
            print('These traces have different frame rates\n')
            
        start_idx1 = np.searchsorted(self.fpho_data_df['time_Green'], start_time)
        start_idx2 = np.searchsorted(obj2.fpho_data_df['time_Green'], start_time)

        #find end
        if end_time == -1:
            end_idx1 = len(self.fpho_data_df['time_Green'])-1
            end_idx2 = len(obj2 .fpho_data_df['time_Green'])-1
        else:
            end_idx1 = np.searchsorted(self.fpho_data_df['time_Green'], end_time)-1
            end_idx2 = np.searchsorted(obj2.fpho_data_df['time_Green'], end_time) -1 

        if start_time - self.fpho_data_df['time_Green'][start_idx1] < -1/self.frame_rate:
            print('Trace 1 starts at ' + str(np.round(self.fpho_data_df['time_Green'][start_idx1])) + 's after your start time ' + str(start_time) + '\n')
        if end_time - self.fpho_data_df['time_Green'][end_idx1] > 1/self.frame_rate:
            print('Trace 1 ends at ' + str(np.round(self.fpho_data_df['time_Green'][end_idx1])) + 's before your end time ' + str(end_time) + 's\n')

        
        if start_time - obj2.fpho_data_df['time_Green'][start_idx2] < -1/obj2.frame_rate:
            print('Trace 2 starts at ' + str(np.round(obj2.fpho_data_df['time_Green'][start_idx2])) + 's after your start time ' + str(start_time) + 's\n')
        if end_time - obj2.fpho_data_df['time_Green'][end_idx2] > 1/obj2.frame_rate:
            print('Trace 2 ends at ' + str(np.round(obj2.fpho_data_df['time_Green'][end_idx2])) + 's before your end time ' + str(end_time) + 's\n')

        
        sig1 = self.fpho_data_df.loc[start_idx1:end_idx1, channel1]
        sig2 = obj2.fpho_data_df.loc[start_idx2:end_idx2, channel2]
        time1 = self.fpho_data_df.loc[start_idx1:end_idx1, 'time_Green']
        time2 = obj2.fpho_data_df.loc[start_idx2:end_idx2, 'time_Green']

        fig = make_subplots(rows = 1, cols = 2)
        #creates a scatter plot
        fig.add_trace(
            go.Scattergl(
            x = sig1,
            y = sig2,
            mode = "markers",
            name ='correlation',
            showlegend = False),
            row = 1, col = 2
            )
        #plots sig1
        fig.add_trace(
            go.Scattergl(
            x = time1,
            y = sig1,
            mode = "lines",
            name = 'sig1',
            showlegend = False),
            row = 1, col = 1
            )
        #plots sig2
        fig.add_trace(
            go.Scattergl(
            x = time2,
            y = sig2,
            mode = "lines",
            name = "sig2",
            showlegend = False),
            row = 1, col = 1
            )
    
        #make traces the same length for correlation 
        shorter_len = min(len(sig1), len(sig2)) 
        #calculates the pearsons R  
        [r, p] = ss.pearsonr(sig1, sig2)

        results = {'Object Name': self.obj_name, 'Channel' : channel1, 'Obj2': obj2.obj_name, 'Obj2 Channel': channel2, 'start_time' : start_time,
                   'end_time' : end_time, 'R Score' : str(r), 'p score': str(p)}
        self.correlation_results = self.correlation_results.append(results, ignore_index = True)
        fig.update_layout(
            title = 'Correlation between ' + self.obj_name + ' and ' 
                  + obj2.obj_name + ' is, ' + str(r) + ' p = ' + str(p)
            )
        return fig
        

    def behavior_specific_pearsons(self, obj2, channel, beh):
        """
        Takes in user chosen objects, channels and behaviors to calculate 
        the behavior specific Pearson’s correlation and plot the signals. 

        Parameters
        ----------
        obj2 : fiber object
            second object for correlation analysis
        
        channel : string
            user selected signals for analysis
        
        beh : string
            user selected behaviors for analysis 
            
        Returns
        --------
        fig : plotly.graph_objects.Scatter
            Plot of signals based on behavior specific correlations
        """
        
        # behaviorSlice=df.loc[:,beh]
        behaviorSlice1 = self.fpho_data_df[self.fpho_data_df[beh] != ' ']
        behaviorSlice2 = obj2.fpho_data_df[self.fpho_data_df[beh] != ' ']

        time = behaviorSlice1['time_Green']
        sig1 = behaviorSlice1[channel]
        sig2 = behaviorSlice2[channel]
        fig = make_subplots(rows = 1, cols = 2)
        fig.add_trace(
            go.Scattergl(
            x = sig1,
            y = sig2,
            mode = "markers",
            name = beh,
            showlegend = False), 
            row = 1, col = 2
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = sig1,
            mode = "lines",
            line = go.scatter.Line(color = 'rgb(255,100,150)'),
            name = channel,
            showlegend = False),
            row = 1, col = 1
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = sig2,
            mode = "lines",
            line = go.scatter.Line(color = 'rgba(100,0,200, .6)'),
            name = channel[1],
            showlegend = False),
            row = 1, col = 1
            )

        [r, p] = ss.pearsonr(sig1, sig2) # 'r' and 'p' were both being referenced before assignment

        fig.update_layout(
                title = 'Correlation between ' + self.obj_name + ' and ' 
                  + obj2.obj_name + ' during ' + beh + ' is, ' + str(r) + ' p = ' + str(p)
                )
        fig.update_xaxes(title_text = channel + ' Zscore', col = 2, row = 1)
        fig.update_yaxes(title_text = channel + ' Zscore', col = 2, row = 1)
        fig.update_xaxes(title_text = 'Time (s)', col = 1, row = 1)
        fig.update_yaxes(title_text = 'Zscore', col = 1, row = 1)

        results = {'Object Name': self.obj_name, 'Channel': channel, 'Object2 Name': obj2.obj_name, 'rObj2 Channel': channel,
                   'Behavior' : beh, 'Number of Events': 'unknown',  'R Score' : r, 'p score': p}
        self.beh_corr_results = self.beh_corr_results.append(results, ignore_index = True)       
        
        return fig
    
##### End Class Functions #####