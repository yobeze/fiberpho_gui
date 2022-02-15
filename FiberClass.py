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
from pathlib import Path
import panel as pn

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
          
        test_red = file.columns.str.endswith('R')
        red_col = np.where(test_red)[0][self.fiber_num-1]

        data_dict = {
            'time_iso': file.iloc[self.start_idx + 2:min:3, time_col].values.tolist(),
            'time_green': file.iloc[self.start_idx:min:3, time_col].values.tolist(),
            'time_red': file.iloc[self.start_idx + 1:min:3, time_col].values.tolist(),
            'green_iso': file.iloc[self.start_idx + 2:min:3, green_col].values.tolist(),
            'green_green': file.iloc[self.start_idx:min:3, green_col].values.tolist(),
            'green_red': file.iloc[self.start_idx + 1:min:3, green_col].values.tolist(),
            'red_iso': file.iloc[self.start_idx + 2:min:3, red_col].values.tolist(),
            'red_green': file.iloc[self.start_idx:min:3, red_col].values.tolist(),
            'red_red': file.iloc[self.start_idx + 1:min:3, red_col].values.tolist()
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
                y = self.fpho_data_df['green_green'],
                mode = "lines",
                line = go.scatter.Line(color = "Green"),
                name = 'f1Green',
                text = 'f1Green',
                showlegend = False), row = 1, col = 1
        )
        fig.add_trace(
            go.Scatter(
                x = self.fpho_data_df['time_iso'],
                y = self.fpho_data_df['green_iso'],
                mode = "lines",
                line = go.scatter.Line(color = "Cyan"),
                name = 'green_iso',
                text = 'green_iso',
                showlegend = False), row = 1, col = 1
        )
        fig.add_trace(
            go.Scatter(
                x = self.fpho_data_df['time_red'],
                y = self.fpho_data_df['red_red'],
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
    def plot_fitted_exp(self):

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

        signals = ['green_green']
        references = ['green_iso']
        
        fig = make_subplots(rows = 3, cols = 2, x_title = 'Time(s)', subplot_titles=("Biexponential Fitted to Signal", "Signal Normalized to Biexponential", "Biexponential Fitted to Ref", "Reference Normalized to Biexponential", "Reference Linearly Fitted to Signal", "Final Normalized Signal"), shared_xaxes=True, vertical_spacing=0.1)
        
        for i in range(len(signals)):
            time = self.fpho_data_df['time_green']
            sig = self.fpho_data_df['green_green']
            popt, pcov = curve_fit(self.fit_exp, time, sig, p0 = (1.0, 0, 1.0, 0, 0), bounds = (0, np.inf))

            AS = popt[0]  # A value
            BS = popt[1]  # B value
            CS = popt[2]  # C value
            DS = popt[3]  # D value
            ES = popt[4]  # E value

            popt, pcov = curve_fit(self.fit_exp, self.fpho_data_df['time_green'], self.fpho_data_df[references[i]], p0=(1.0, 0, 1.0, 0, 0), bounds=(0,np.inf))

            AR = popt[0]  # A value
            BR = popt[1]  # B value
            CR = popt[2]  # C value
            DR = popt[3]  # D value
            ER = popt[4]       

            # Generate fit line using calculated coefficients
            fitSig = self.fit_exp(self.fpho_data_df['time_green'], AS, BS, CS, DS, ES)
            fitRef = self.fit_exp(self.fpho_data_df['time_green'], AR, BR, CR, DR, ER)

            sigRsquare = np.corrcoef(self.fpho_data_df[signals[i]], fitSig)[0,1]**2
            refRsquare = np.corrcoef(self.fpho_data_df[references[i]], fitRef)[0,1]**2
            print('sig r^2 =', sigRsquare ,'ref r^2 =', refRsquare )

            if sigRsquare < .01:
                print('sig r^2 =', sigRsquare)
                print('No exponential decay was detected in ', signals[i])
                print(signals[i] + ' expfit is now the median of ', signals[i])
                AS = 0
                BS = 0
                CS = 0
                DS = 0
                ES = np.median(self.fpho_data_df[signals[i]])
                fitSig = self.fit_exp(self.fpho_data_df['time_green'], AS, BS, CS, DS, ES)


            if refRsquare < .001:
                print('ref r^2 =', refRsquare)
                print('No exponential decay was detected in ', references[i])
                print(references[i] + ' expfit is now the median  ', references[i])
                AR = 0
                BR = 0
                CR = 0
                DR = 0
                ER = np.median(self.fpho_data_df[references[i]])
                fitRef = self.fit_exp(self.fpho_data_df['time_green'], AR, BR, CR, DR, ER)


            normedSig = [(k/j) for k,j in zip(self.fpho_data_df[signals[i]], fitSig)]
            normedRef = [(k/j) for k,j in zip(self.fpho_data_df[references[i]], fitRef)]      

            popt, pcov = curve_fit(self.lin_fit, normedSig, normedRef, bounds = ([0, -5],[np.inf, 5]))

            AL = popt[0]
            BL = popt[1]

            AdjustedRef=[AL* j + BL for j in normedRef]
            normedToReference=[(k/j) for k,j in zip(normedSig, AdjustedRef)]

            # below saves all the variables we generated to the df #
            #  data frame inside the obj ex. self 
            # and assign all the long stuff to that
            # assign the AS, BS,.. etc and AR, BR, etc to lists called self.sig_fit_coefficients, self.ref_fit_coefficients and self.sig_to_ref_coefficients
            self.fpho_data_df.loc[:,signals[i] + ' expfit'] = fitSig
            # self.fpho_data_df.loc[:,signals[i] + ' expfit parameters']=['na']
            
            
            #self.fpho_data_df.at[0:4, signals[i] + ' expfit parameters']=['A= ' + str(AS), 'B= ' + str(BS), 'C= ' + str(CS), 'D= ' + str(DS), 'E= ' + str(ES)]
            self.sig_fit_coefficients = ['A= ' + str(AS), 'B= ' + str(BS), 'C= ' + str(CS), 'D= ' + str(DS), 'E= ' + str(ES)]
            
            self.fpho_data_df.loc[:,signals[i] + ' normed to exp']=normedSig
            self.fpho_data_df.loc[:,references[i] + ' expfit']=fitRef

            self.ref_fit_coefficients = ['A= ' + str(AR), 'B= ' + str(BR), 'C= ' + str(CR), 'D= ' + str(DR), 'E= ' + str(ER)]
            
            self.fpho_data_df.loc[:,references[i] + ' normed to exp']=normedRef
            self.fpho_data_df.loc[:,references[i] + ' fitted to ' + signals[i]]=AdjustedRef

            self.sig_to_ref_coefficients = ['A= ' + str(AL), 'B= ' + str(BL)]
            self.fpho_data_df.loc[:,signals[i] + ' final normalized'] = normedToReference

            # fig = make_subplots(rows = 3, cols = 2, x_title = 'Time(s)', subplot_titles=("Biexponential Fitted to Signal", "Signal Normalized to Biexponential", "Biexponential Fitted to Ref", "Reference Normalized to Biexponential", "Reference Linearly Fitted to Signal", "Final Normalized Signal"), shared_xaxes=True, vertical_spacing=0.1)
            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df[signals[i]],
                mode = "lines",
                line = go.scatter.Line(color="Green"),
                name ='Signal:' + signals[i],
                text = 'Signal',
                showlegend = False), row = 1, col = 1
            )
            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df[signals[i] + ' expfit'],
                mode = "lines",
                line = go.scatter.Line(color="Purple"),
                text = 'Biexponential fitted to Signal',
                showlegend = False), row = 1, col = 1
            )
            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df[signals[i] + ' normed to exp'],
                mode = "lines",
                line = go.scatter.Line(color="Green"),
                text = 'Signal Normalized to Biexponential',
                showlegend = False), row = 1, col = 2
            )
            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df[references[i]],
                mode = "lines",
                line = go.scatter.Line(color="Cyan"),
                name = 'Reference:' + references[i],
                text = 'Reference',
                showlegend = False), row = 2, col = 1
            )
            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df[references[i] + ' expfit'],
                mode = "lines",
                line = go.scatter.Line(color="Purple"),
                text = 'Biexponential fit to Reference',
                showlegend = False), row = 2, col = 1
            )
            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df[references[i] + ' normed to exp'],
                mode = "lines",
                line = go.scatter.Line(color="Cyan"),
                text = 'Reference Normalized to Biexponential',
                showlegend = False), row = 2, col = 2
            )
            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df[signals[i] + ' normed to exp'],
                mode = "lines",
                line = go.scatter.Line(color="Green"),
                text = 'Signal Normalized to Biexponential',
                showlegend = False), row = 3, col = 1
            )

            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df[references[i] + ' fitted to ' + signals[i]],
                mode = "lines",
                line = go.scatter.Line(color="Cyan"),
                text = 'Reference linearly scaled to signal',
                showlegend = False), row = 3, col = 1  
            )

            fig.add_trace(
                go.Scatter(
                x = self.fpho_data_df['time_green'],
                y = self.fpho_data_df[signals[i] + ' final normalized'],
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
    
    def display(self, num):
        #Prints/displays relevant info according to function number passed in
        if num == 1:
            #Only works if plot_fit function runs first to initialise lists
            print(self.sig_fit_coefficients)
            print(self.ref_fit_coefficients)
            print(self.sig_to_ref_coefficients)
            
        elif num == 2:
            print(self.fpho_data_df.head(5))
            # head = list(self.fpho_data_dict)[:1]
            # print(head)
        
        elif num == 3:
            print("Yurrrr")
        
        else:
            raise error("Invalid input")
            
            
    # ----------------------------------------------------- # 
    # Behavior Functions
    # ----------------------------------------------------- # 

    def import_behavior_data(BORIS_filename, fdata):
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
        IdxOfBeh = [i for i in range(len(BORISData['Behavior'])) if BORISData.loc[i, 'Behavior'] == beh]                    
        j=0
        fdata[beh]=False
        while j < len(IdxOfBeh):
            if BORISData.loc[(IdxOfBeh[j]), 'Status']=='POINT': 
                pointIdx=fdata['fTimeGreen'].searchsorted(BORISData.loc[IdxOfBeh[j],'Time'])
                fdata.loc[pointIdx, beh]=True
                j=j+1
            elif BORISData.loc[(IdxOfBeh[j]), 'Status']=='START' and BORISData.loc[(IdxOfBeh[j+1]), 'Status']=='STOP':
                startIdx=fdata['fTimeGreen'].searchsorted(BORISData.loc[IdxOfBeh[j],'Time'])
                endIdx=fdata['fTimeGreen'].searchsorted(BORISData.loc[IdxOfBeh[j+1],'Time'])
                fdata.loc[startIdx:endIdx, beh]=True
                j=j+2
            else: 
                print("\nStart and stops for state behavior:" + beh + " are not paired correctly.\n")
                sys.exit()
    return(fdata)
    
    
        

# def new_exp(self):
#         entry_fields = 'Fiber Num', 'Animal Num', 'Exp. Date', 'Exp. Time'
#         file_path = "/Users/yobae/Desktop/CS Stuff/fiberphotometry-master/Python/Yobe_test",
#         # filetypes = (("CSV files", "*.csv"),))
#         files = 
#         # Open file, catch errors
#         if file is not None: 
#             try:
#                 self.df_test = pd.read_csv(file)
#             except FileNotFoundError:
#                 print("Could not find file: " + file)
#                 sys.exit(1)
#             except PermissionError:
#                 print("Could not access file: " + file)
#                 sys.exit(2)
#         if not self.df_test.empty:
#             ents = self.makeform(entry_fields)
            
#             self.new_exp_button.pack_forget()
#             # self.new_exp_button.grid_forget()
#             # self.open_pickle_button.grid_forget()
#             self.open_pickle_button.pack_forget()
#             print(ents)
#             self.controller.bind("<Return>", (lambda event, e = ents: self.user_input(e)))
#             # root.bind('<Return>', (lambda event, e = ents: fetch(e)))
