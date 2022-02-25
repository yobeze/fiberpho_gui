from os import error
import sys
import argparse
# import fpho_config_copy
# import fpho_setup_copys
# import correlation_setup_copy
# import behavior_setup_copy
import pandas as pd
import numpy as np
import csv
import pickle
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import butter, filtfilt
from pathlib import Path
import panel as pn
from statistics import mean
import matplotlib.pyplot as plt
import scipy.stats as ss
import re

pn.extension()

class fiberObj:
#figure out if list/dict is better than dataframe
    start_idx = 301
    colIdx = 0
    
    
    def __init__(self, file, obj, fiber_num, animal, exp_date, exp_start_time):
        self.obj_name = obj
        self.fiber_num = fiber_num
        self.animal_num = animal
        self.exp_date = exp_date
        self.exp_start_time = exp_start_time
        self.behaviors = set()
        self.channels = set()
        
        #modify to accept dictionary w values
        
        start_is_not_green = True
        while(start_is_not_green):
            if file["Flags"][self.start_idx] == 18:
                start_is_not_green = False
            if file["Flags"][self.start_idx] == 20:
                start_is_not_green = True
                self.start_idx=self.start_idx+1
            if file["Flags"][self.start_idx] == 17:
                start_is_not_green = True
                self.start_idx=self.start_idx+1
    
        # Find min data length
        file['Timestamp'] = (file['Timestamp'] - file['Timestamp'][0])
        length = len(file['Flags']) - 1
        values = length - self.start_idx
        extras = values % 3
        min = int(length - extras)
        
        #Setting column values
        #Finds columns by str and sets to assigns index to numpy array
        time_col = 1
        
        test_green = file.columns.str.endswith('G')
        green_col = np.where(test_green)[0][self.fiber_num-1]
        self.channels.add('Raw_Green')
        self.channels.add('Raw_Isosbestic')
        
        test_red = file.columns.str.endswith('R')
        red_col = np.where(test_red)[0][self.fiber_num-1]
        self.channels.add('Raw_Red')  
        
        data_dict = {
            'time_iso': file.iloc[self.start_idx + 2:min:3, time_col].values.tolist(),
            'time_green': file.iloc[self.start_idx:min:3, time_col].values.tolist(),
            'time_red': file.iloc[self.start_idx + 1:min:3, time_col].values.tolist(),
            'Raw_Isosbestic': file.iloc[self.start_idx + 2:min:3, green_col].values.tolist(),
            'Raw_Green': file.iloc[self.start_idx:min:3, green_col].values.tolist(),
            'green_red': file.iloc[self.start_idx + 1:min:3, green_col].values.tolist(),
            'red_iso': file.iloc[self.start_idx + 2:min:3, red_col].values.tolist(),
            'red_green': file.iloc[self.start_idx:min:3, red_col].values.tolist(),
            'Raw_Red': file.iloc[self.start_idx + 1:min:3, red_col].values.tolist()
        }

        dict_items = data_dict.items()
#         head = list(dict_items)[:1][:]
        self.fpho_data_dict = data_dict
        self.fpho_data_df = pd.DataFrame.from_dict(data_dict)
        
        
     
    ##Helper Functions   
    def fit_exp(self, values, a, b, c, d, e):
        """Transforms data into an exponential function
            of the form y=A*exp(-B*X)+C*exp(-D*x) + E

            Parameters
            ----------
            values: list
                    data
            a, b, c, d, e: integers or floats
                    estimates for the parameter values of
                    A, B, C, D and E
        """
        values = np.array(values)

        return a * np.exp(-b * values) + c * np.exp(-d * values) + e

    def lin_fit(self, values, a, b):
        values = np.array(values)

        return a * values + b
    ##Helper Functions
    
    #Validates the instance properly created
    def validate(self):
        has_attribute_1 = hasattr(test_1, "fpho_data_df")
#         has_attribute_2 = hasattr(test_2, "fpho_data_df")

        if has_attribute_1:
            print("Instance and dataframe created")
            print(self.fpho_data_df.head(5))
#         elif has_attribute_2:
#             print("Second instance created")
#             print(fpho_data_df.head(5))
        else:
            raise error("No instance created")
            
    
##### Class Functions ####

    #Signal Trace function
    def raw_signal_trace(self):
        fig = make_subplots(rows = 1, cols = 2, shared_xaxes = True, vertical_spacing = 0.02, x_title = "Time (s)", y_title = "Fluorescence")
        fig.add_trace(
            go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df['Raw_Green'],
                mode = "lines",
                line = go.scatter.Line(color = "Green"),
                name = 'f1Green',
                text = 'f1Green',
                showlegend = False), row = 1, col = 1
        )
        fig.add_trace(
            go.Scatter(
                x = self.fpho_data_df['time_iso'],
                y = self.fpho_data_df['Raw_Isosbestic'],
                mode = "lines",
                line = go.scatter.Line(color = "Cyan"),
                name = 'Raw_Isosbestic',
                text = 'Raw_Isosbestic',
                showlegend = False), row = 1, col = 1
        )
        fig.add_trace(
            go.Scatter(
                x = self.fpho_data_df['time_red'],
                y = self.fpho_data_df['Raw_Red'],
                mode = "lines",
                line = go.scatter.Line(color="Red"),
                name = 'f1Red',
                text = 'f1Red',
                showlegend = False), row = 1, col = 2
        )
        fig.add_trace(
            go.Scatter(
                x = self.fpho_data_df['time_iso'],
                y = self.fpho_data_df['red_iso'],
                mode = "lines",
                line = go.scatter.Line(color="Violet"),
                name = 'red_iso',
                text = 'red_iso',
                showlegend = False), row = 1, col = 2
        )
        
        fig.update_layout(
            title = "Raw Traces from all channels for animal number " + str(self.animal_num) + " on " + self.exp_date + " at " + self.exp_start_time
        )
        # fig.show()
        return fig
        # fig.write_html(self.obj_name+'raw_sig.html', auto_open = True)

    
    #Plot fitted exp function
    def normalize_a_signal(self, signal, reference):

        """Creates a plot normalizing 1 fiber data to an
            exponential of the form y=A*exp(-B*X)+C*exp(-D*x)

            Parameters
            ----------
            fpho_dataframe: string
                    pandas dataframe
            output_filename: string
                    name for output csv
            Returns:
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
        Times={'Green':'time_green', 'Red':'time_red', 'Isosbestic':'time_iso'}
        
     
        fig = make_subplots(rows = 3, cols = 2, x_title = 'Time(s)', subplot_titles=("Biexponential Fitted to Signal", "Signal Normalized to Biexponential", "Biexponential Fitted to Ref", "Reference Normalized to Biexponential", "Reference Linearly Fitted to Signal", "Final Normalized Signal"), shared_xaxes=True, vertical_spacing=0.1)
        
        # time = self.fpho_data_df[Times[signal]]
        time = self.fpho_data_df['time_green']
        sig = self.fpho_data_df[signal]
        ref = self.fpho_data_df[reference]
        popt, pcov = curve_fit(self.fit_exp, time, sig, p0 = (1.0, 0, 1.0, 0, 0), bounds = (0, np.inf))

        AS = popt[0]  # A value
        BS = popt[1]  # B value
        CS = popt[2]  # C value
        DS = popt[3]  # D value
        ES = popt[4]  # E value

        popt, pcov = curve_fit(self.fit_exp, time, ref, p0=(1.0, 0, 1.0, 0, 0), bounds=(0,np.inf))

        AR = popt[0]  # A value
        BR = popt[1]  # B value
        CR = popt[2]  # C value
        DR = popt[3]  # D value
        ER = popt[4]       

        # Generate fit line using calculated coefficients
        fitSig = self.fit_exp(time, AS, BS, CS, DS, ES)
        fitRef = self.fit_exp(time, AR, BR, CR, DR, ER)

        sigRsquare = np.corrcoef(sig, fitSig)[0,1]**2
        refRsquare = np.corrcoef(ref, fitRef)[0,1]**2
        print('sig r^2 =', sigRsquare ,'ref r^2 =', refRsquare )

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


        normedSig = [(k/j) for k,j in zip(sig, fitSig)]
        normedRef = [(k/j) for k,j in zip(ref, fitRef)]      

        popt, pcov = curve_fit(self.lin_fit, normedSig, normedRef, bounds = ([0, -5],[np.inf, 5]))

        AL = popt[0]
        BL = popt[1]

        AdjustedRef=[AL* j + BL for j in normedRef]
        normedToReference=[(k/j) for k,j in zip(normedSig, AdjustedRef)]

        # below saves all the variables we generated to the df #
        #  data frame inside the obj ex. self 
        # and assign all the long stuff to that
        # assign the AS, BS,.. etc and AR, BR, etc to lists called self.sig_fit_coefficients, self.ref_fit_coefficients and self.sig_to_ref_coefficients
        self.fpho_data_df.loc[:, signal + ' expfit'] = fitSig
        # self.fpho_data_df.loc[:,signals[i] + ' expfit parameters']=['na']


        #self.fpho_data_df.at[0:4, signals[i] + ' expfit parameters']=['A= ' + str(AS), 'B= ' + str(BS), 'C= ' + str(CS), 'D= ' + str(DS), 'E= ' + str(ES)]
        self.sig_fit_coefficients = ['A= ' + str(AS), 'B= ' + str(BS), 'C= ' + str(CS), 'D= ' + str(DS), 'E= ' + str(ES)]

        self.fpho_data_df.loc[:, signal + ' normed to exp']=normedSig
        self.fpho_data_df.loc[:, reference + ' expfit']=fitRef

        self.ref_fit_coefficients = ['A= ' + str(AR), 'B= ' + str(BR), 'C= ' + str(CR), 'D= ' + str(DR), 'E= ' + str(ER)]

        self.fpho_data_df.loc[:, reference + ' normed to exp']=normedRef
        self.fpho_data_df.loc[:,reference + ' fitted to ' + signal]=AdjustedRef

        self.sig_to_ref_coefficients = ['A= ' + str(AL), 'B= ' + str(BL)]
        self.fpho_data_df.loc[:,signal[4:] + '_Normalized'] = normedToReference
        
        self.channels.add(signal[4:] + '_Normalized')
        # fig = make_subplots(rows = 3, cols = 2, x_title = 'Time(s)', subplot_titles=("Biexponential Fitted to Signal", "Signal Normalized to Biexponential", "Biexponential Fitted to Ref", "Reference Normalized to Biexponential", "Reference Linearly Fitted to Signal", "Final Normalized Signal"), shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(
            go.Scatter(
            x = time,
            y = sig,
            mode = "lines",
            line = go.scatter.Line(color="Green"),
            name ='Signal:' + signal,
            text = 'Signal',
            showlegend = False), row = 1, col = 1
        )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[signal + ' expfit'],
            mode = "lines",
            line = go.scatter.Line(color="Purple"),
            text = 'Biexponential fitted to Signal',
            showlegend = False), row = 1, col = 1
        )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[signal + ' normed to exp'],
            mode = "lines",
            line = go.scatter.Line(color="Green"),
            text = 'Signal Normalized to Biexponential',
            showlegend = False), row = 1, col = 2
        )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = ref,
            mode = "lines",
            line = go.scatter.Line(color="Cyan"),
            name = 'Reference:' + reference,
            text = 'Reference',
            showlegend = False), row = 2, col = 1
        )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[reference + ' expfit'],
            mode = "lines",
            line = go.scatter.Line(color="Purple"),
            text = 'Biexponential fit to Reference',
            showlegend = False), row = 2, col = 1
        )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[reference + ' normed to exp'],
            mode = "lines",
            line = go.scatter.Line(color="Cyan"),
            text = 'Reference Normalized to Biexponential',
            showlegend = False), row = 2, col = 2
        )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[signal + ' normed to exp'],
            mode = "lines",
            line = go.scatter.Line(color="Green"),
            text = 'Signal Normalized to Biexponential',
            showlegend = False), row = 3, col = 1
        )

        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[reference + ' fitted to ' + signal],
            mode = "lines",
            line = go.scatter.Line(color="Cyan"),
            text = 'Reference linearly scaled to signal',
            showlegend = False), row = 3, col = 1  
        )

        fig.add_trace(
            go.Scatter(
            x = time,
            y = self.fpho_data_df[signal[4:] + '_Normalized'],
            mode="lines",
            line = go.scatter.Line(color = "Pink"), 
            text = 'Final Normalized Signal',
            showlegend = False), row = 3, col = 2

        )
        # fig.update_layout(
        #     title = "Normalizing " + signals[i] + ' for ' + self.file
        # )
        # fig.show()
        # return fig
            
        return fig
    

    # ----------------------------------------------------- # 
    # Behavior Functions
    # ----------------------------------------------------- # 

    def import_behavior_data(self, BORIS_filename):
        """Takes a file name, returns a dataframe of parsed data

            Parameters
            ----------
            BORIS_filename: string
                            The path to the CSV file

            Returns:
            --------
            behaviorData: pandas dataframe
                    contains:
                         Time(total msec), Time(sec), Subject,
                         Behavior, Status
            """

        # Open file, catch errors
        try:
            BORISData = pd.read_csv(BORIS_filename, header=15)  # starts at data
        except FileNotFoundError:
            print("Could not find file: " + BORIS_filename)
            sys.exit(1)
        except PermissionError:
            print("Could not access file: " + BORIS_filename)
            sys.exit(2)

        UniqueBehaviors=BORISData['Behavior'].unique()
        for beh in UniqueBehaviors:
            self.behaviors.add(beh)
            IdxOfBeh = [i for i in range(len(BORISData['Behavior'])) if BORISData.loc[i, 'Behavior'] == beh]                    
            j=0
            self.fpho_data_df[beh]=' '
            while j < len(IdxOfBeh):
                if BORISData.loc[(IdxOfBeh[j]), 'Status']=='POINT': 
                    pointIdx=self.fpho_data_df['time_green'].searchsorted(BORISData.loc[IdxOfBeh[j],'Time'])
                    self.fpho_data_df.loc[pointIdx, beh]='S'
                    j=j+1
                elif BORISData.loc[(IdxOfBeh[j]), 'Status']=='START' and BORISData.loc[(IdxOfBeh[j+1]), 'Status']=='STOP':
                    startIdx=self.fpho_data_df['time_green'].searchsorted(BORISData.loc[IdxOfBeh[j],'Time'])
                    endIdx=self.fpho_data_df['time_green'].searchsorted(BORISData.loc[IdxOfBeh[j+1],'Time'])
                    self.fpho_data_df.loc[startIdx, beh]='S'
                    self.fpho_data_df.loc[startIdx+1:endIdx-1, beh]='O'
                    self.fpho_data_df.loc[endIdx, beh]='E'
                    j=j+2
                else: 
                    print("\nStart and stops for state behavior:" + beh + " are not paired correctly.\n")
                    sys.exit()

        return
    
    
        
    def plot_behavior(self, channels):
        # channels =[channel_dict[i] for i in channels_inputed]
        fig = make_subplots(rows=len(channels), cols = 1, subplot_titles = [channel for channel in channels], shared_xaxes = True)
        for i, channel in enumerate(channels):
            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df[channel],
                mode = "lines",
                line = go.scatter.Line(color = "Green"),
                name = channel,
                showlegend = True), row = i + 1, col = 1
                )
            
            colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
            for j, beh in enumerate(self.behaviors):
                temp_beh_string = ''.join([key for key in self.fpho_data_df[beh]])
                pattern = re.compile(r'S[O]+E')
                bouts = pattern.finditer(temp_beh_string)
                for bout in bouts:
                    start_time = self.fpho_data_df.at[bout.start(), 'time_green']
                    end_time = self.fpho_data_df.at[bout.end(), 'time_green']
                    fig.add_vrect( x0=start_time, x1 = end_time, 
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
                    start_time = self.fpho_data_df.at[start.start(), 'time_green']
                    fig.add_vline( x = start_time, 
                                layer = "below",
                                line_width = 3, 
                                line_color = colors[j % 10],
                                row = i + 1, col = 1,
                                name = beh
                                )
                
                # flag = False
                # for k in range(len(self.fpho_data_df[channel])):
                #     if self.fpho_data_df.at[k, beh] == True:
                #         if flag == False:
                #             start = self.fpho_data_df.at[k, 'time_green'] 
                #             flag = True
                #     else:
                #         if flag == True:
                #             end = self.fpho_data_df.at[k, 'time_green'] 
                #             fig.add_vrect(
                #                 x0=start, x1=end, opacity=0.75,
                #                 line_width=1, 
                #                 layer="below",
                #                 fillcolor=colors[j%10],
                #                 row=i+1, col=1,
                #                 name=beh
                #                 )
                #             flag = False
                # if flag == True:
                #     end = self.fpho_data_df.at[k, 'time_green'] 
                
                fig.add_annotation(xref = "x domain", yref = "y domain",
                    x = 1, 
                    y = (j + 1)/len(self.behaviors),
                    text = beh,
                    bgcolor = colors[j % 10],
                    showarrow = False,
                    row = i + 1, col = 1
                    )
        # fig.show()
        return fig
        
    
    def plot_zscore(self, channel, beh):
        """Takes a dataframe and creates plot of z-scores for
        each time a select behavior occurs with the avg
    z-score and SEM"""
        BehTimes=list(self.fpho_data_df[(self.fpho_data_df[beh]=='S')]['time_green'])
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Full trace with events', 'average'))
        fig.add_trace(
            go.Scatter(
            x=self.fpho_data_df['time_green'],
            y=self.fpho_data_df[channel],
            mode="lines",
            line=go.scatter.Line(color="Green"),
            name =channel,
            showlegend=True), row=1, col=1
        )
        sum=[]
        zscoresum=[]
        for time in BehTimes:
            tempy=self.fpho_data_df.loc[self.fpho_data_df['time_green'].searchsorted(time-1):self.fpho_data_df['time_green'].searchsorted(time+5),channel].values.tolist()
            if len(sum)>1:
                sum = [sum[i] + tempy[i] for i in range(len(tempy))]
            else:
                sum=tempy
            x=self.fpho_data_df.loc[self.fpho_data_df['time_green'].searchsorted(time-1):self.fpho_data_df['time_green'].searchsorted(time+5),'time_green']
            fig.add_vline(x=time, line_dash="dot", row=1, col=1)
            fig.add_trace(
                go.Scatter( 
                x=x-time,
                y=ss.zscore(self.fpho_data_df.loc[self.fpho_data_df['time_green'].searchsorted(time-1):self.fpho_data_df['time_green'].searchsorted(time+5),channel]),
                mode="lines",
                line=dict(color="Black", width=0.5, dash = 'dot'),
                name =channel,
                showlegend=False), row=1, col=2
            )
        fig.add_vline(x=0, line_dash="dot", row=1, col=2)
        fig.add_trace(
            go.Scatter( 
            x=x-time,
            y=ss.zscore([i/len(BehTimes) for i in sum]),
            mode="lines",
            line=dict(color="Red", width=3),
            name =channel,
            showlegend=False), row=1, col=2
            )
        fig.update_layout(
        #title= beh + ' overlaid on ' + channel + ' for animal ' +str(self.fpho_data_df['animalID'][0]) + ' on ' + str(self.fpho_data_df['date'][0]),
        xaxis_title='Time')
        return fig
    
    
    
    
    
    
    #return the pearsons correlation coefficient and r value between 2 full channels and plots the signals overlaid and thier scatter plot
    def within_trial_pearsons(self, key, channels):
        results={}
        for channel in channels:
            our_channels = [col for col in self.fpho_data_df.columns if channel + ' final normalized' in col]
            if len(our_channels) < 1:
                print('data must be normalized before running a correlation')
                sys.exit()
            sig1 = self.fpho_data_df[our_channels[0]]
            sig2 = self.fpho_data_df[our_channels[1]]

            time = self.fpho_data_df['time_green']
            for i in range(50, 100, 10):
                #sig1smooth = ss.zscore(uniform_filter1d(sig1, size=i))
                #sig2smooth = ss.zscore(uniform_filter1d(sig2, size=i))
                fig = make_subplots(rows=1, cols=2)
                #creates a scatter plot
                fig.add_trace(
                    go.Scatter(
                    x=sig1smooth,
                    y=sig2smooth,
                    mode="markers",
                    name ='correlation',
                    showlegend=False), row=1, col=2
                )
                #plots sig1
                fig.add_trace(
                    go.Scatter(
                    x=time,
                    y=sig1,
                    mode="lines",
                    name='sig1',
                    showlegend=False), row=1, col=1
                )
                #plots sig2
                fig.add_trace(
                    go.Scatter(
                    x=time,
                    y=sig2,
                    mode="lines",
                    name = "sig2",
                    showlegend=False), row=1, col=1
                )

            #calculates the pearsons R  
            [r, p] = ss.pearsonr(sig1, sig2)
            results[channel]=[r, p]
        #returns the pearsons R
        print('Pearsons')
        return results
    
    
    
    #return the pearsons 
    def behavior_specific_pearsons(self, file, channels, behs):
        results = {}
        for channel in channels:
            our_channels = [chan + ' final normalized' for chan in channel ]
            channel_names=channel[0]+ ' vs ' +channel[1]
            if len(our_channels) < 1:
                print('data must be normalized before running a correlation')
                sys.exit()
            results[channel_names] = {} 
            sig1 = []
            sig2 = []
            for beh in behs:
                behaviorname=''
                flag=False
                for name in beh:
                    behaviorname= behaviorname + ' ,' + name
                    # if name in df.columns:
                    if name in self.fpho_data_df.columns
                        flag = True
                if flag:
                    # behaviorSlice=df.loc[:,beh]
                    behaviorSlice = self.fpho_data_df.loc[:,beh]
                    TrueTimes = behaviorSlice.any(axis=1);
                    # corDf = pd.concat([df['fTimeGreen'], df[our_channels[0]], df[our_channels[1]], TrueTimes], axis=1)
                    corDf = pd.concat([self.fpho_data_df['time_green'], self.fpho_data_df[our_channels[0]], df[our_channels[1]], TrueTimes],                       axis=1)
                    corDf.columns = ['time_green', our_channels[0], our_channels[1], 'TrueTimes']

                    sig1 = corDf[corDf.TrueTimes == True][our_channels[0]].tolist()
                    sig2 = corDf[corDf.TrueTimes == True][our_channels[1]].tolist()


                    #difsig1=[sig1.iloc[i+1]-sig1.iloc[i] for i in range(len(sig1)-1)]
                    #difsig2=[sig2.iloc[i+1]-sig2.iloc[i] for i in range(len(sig2)-1)]
                    #sig1 = ss.zscore(uniform_filter1d(sig1, size=50))
                    #sig2 = ss.zscore(uniform_filter1d(sig2, size=50))

                    time = corDf[corDf.TrueTimes == True]['time_green']
                    fig = make_subplots(rows = 1, cols = 2)
                    fig.add_trace(
                        go.Scatter(
                        x = sig1,
                        y = sig2,
                        mode = "markers",
                        name = behaviorname,
                        showlegend = False), row = 1, col = 2
                    )
                    fig.add_trace(
                        go.Scatter(
                        x = time,
                        y = sig1,
                        mode = "lines",
                        line = go.scatter.Line(color = 'rgb(255,100,150)'),
                        name = channel[0],
                        showlegend = False), row = 1, col = 1
                    )
                    fig.add_trace(
                        go.Scatter(
                        x = time,
                        y = sig2,
                        mode = "lines",
                        line = go.scatter.Line(color = 'rgba(100,0,200, .6)'),
                        name = channel[1],
                        showlegend = False), row = 1, col = 1
                    )
                    fig.update_yaxes(
                        #title_text = "Interbrain Correlation (PCC)",
                        showgrid = True,
                        #showline=True, linewidth=2, linecolor='black',
                    )
                    fig.update_layout(
                            title = channel_names + ' while' + behaviorname + ' for ' + file
                            )
                    fig.update_xaxes(title_text = channel[0]+ ' zscore')
                    fig.update_yaxes(title_text = channel[1] + ' zscore')

                    fig.update_xaxes(title_text = 'Time (s)', col = 1, row = 1)
                    fig.update_yaxes(title_text = 'Zscore', col = 1, row = 1)

                    fig.show()
                    #fig.write_image('together_seperate1.pdf')
                    [r, p] = ss.pearsonr(sig1, sig2)
                    #print(sig1.iloc[0:len(sig1)*(1/3)])
                    #print(sig1.type())
                    beg = ss.pearsonr(sig1[0:int(len(sig1)*(1/3))], sig2[0:int(len(sig1)*(1/3))])
                    mid = ss.pearsonr(sig1[int(len(sig1)*(1/3)):int(len(sig1)*(2/3))], sig2[int(len(sig1)*(1/3)):int(len(sig1)*(2/3))])
                    end = ss.pearsonr(sig1[int(len(sig1)*(2/3)):], sig2[int(len(sig1)*(2/3)):])
                else:
                    [r, p] = ['na', 'na']
                    print(behaviorname + ' not found in this trial')
                results[channel_names][behaviorname]={'full':r, 'start':beg, 'middle':mid, 'end':end}  

        # Change this to append to gui
        # perhaps return file/results then append to gui
        print(file)
        print(results)
        return 