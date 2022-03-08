"""Library of functions for behavior analysis
    * import_behavior_data - inputs data from BORIS csv
    * plot_zscore - plots z-score for each behavior occurance
"""
import sys
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def plot_behavior(fdata, key, channels):
    fig = make_subplots(rows=len(channels), cols=1, subplot_titles=[channel for channel in channels], shared_xaxes=True)
    for i, channel in enumerate(channels):
        fig.add_trace(
            go.Scatter(
            x=fdata['fTimeGreen'],
            y=fdata[channel],
            mode="lines",
            line=go.scatter.Line(color="Green"),
            name =channel,
            showlegend=True), row=i+1, col=1
            )
        behaviors = fdata.select_dtypes(include=['bool']).columns
        colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
        for j, beh in enumerate(behaviors):
            flag = False
            for k in range(len(fdata[channel])):
                if fdata.at[k, beh] == True:
                    if flag == False:
                        start = fdata.at[k, 'fTimeGreen'] 
                        flag = True
                else:
                    if flag == True:
                        end = fdata.at[k, 'fTimeGreen'] 
                        fig.add_vrect(
                            x0=start, x1=end, opacity=0.75,
                            line_width=1, 
                            layer="below",
                            fillcolor=colors[j%10],
                            row=i+1, col=1,
                            name=beh
                            )
                        flag = False
            if flag == True:
                end = fdata.at[k, 'fTimeGreen'] 
                fig.add_vrect( x0=start, x1=end, 
                        opacity=0.75,
                        layer="below",
                        line_width=1, 
                        fillcolor=colors[j%10],
                        row=i+1, col=1,
                        name=beh
                        )
            fig.add_annotation(xref="x domain", yref="y domain",
                x=1, 
                y=(j+1)/len(behaviors),
                text=beh,
                bgcolor = colors[j%10],
                showarrow=False,
                row=i+1, col=1
                )
    fig.show()
                
            

def plot_zscore(fdata, key, channels, behs):   
    """Takes a dataframe and creates plot of z-scores for
        each time a select behavior occurs with the avg
        z-score and SEM
    """
    for channel in channels:
        for beh in behs:
            BehTimes=list(fdata[(fdata[beh]==True)]['fTimeGreen'])
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Full trace with events', 'average'))
            fig.add_trace(
                go.Scatter(
                x=fdata['fTimeGreen'],
                y=fdata[channel],
                mode="lines",
                line=go.scatter.Line(color="Green"),
                name =channel,
                showlegend=True), row=1, col=1
            )
            sum=[]
            zscoresum=[]
            for time in BehTimes:
                tempy=fdata.loc[fdata['fTimeGreen'].searchsorted(time-1):fdata['fTimeGreen'].searchsorted(time+5),channel].values.tolist()
                if len(sum)>1:
                    sum = [sum[i] + tempy[i] for i in range(len(tempy))]
                else:
                    sum=tempy
                x=fdata.loc[fdata['fTimeGreen'].searchsorted(time-1):fdata['fTimeGreen'].searchsorted(time+5),'fTimeGreen']
                fig.add_vline(x=time, line_dash="dot", row=1, col=1)
                fig.add_trace(
                    go.Scatter( 
                    x=x-time,
                    y=ss.zscore(fdata.loc[fdata['fTimeGreen'].searchsorted(time-1):fdata['fTimeGreen'].searchsorted(time+5),channel]),
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
            title= beh + ' overlaid on ' + channel + ' for animal ' +str(fdata['animalID'][0]) + ' on ' + str(fdata['date'][0]),
            xaxis_title='Time')
            fig.show()
    return
    # built in zscore function?
    # don't pick a baseline, instead use the mean (of the clip) as the baseline

def plot_FFT(df, channels):
    for channel in channels:
        our_channels = [col for col in df.columns if channel + ' final normalized' in col]
    sig1 = df[our_channels[0]]
    sig2 = df[our_channels[1]]
    time = df['fTimeGreen']
    # Number of sample points
    N = len(sig1)
    #sampling rate
    T=N/(time.iloc[-1]-time.iloc[0])
    y1 = np.square(np.abs(np.fft.rfft(sig1.tolist())))
    y2 = np.square(np.abs(np.fft.rfft(sig2.tolist())))
    xf = np.linspace(0, T/2, len(y1))
    
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
        x=xf,
        y=y1,
        mode="lines",
        name ='animal 2 at' + str(T),
        showlegend=True), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
        x=xf,
        y=y2,
        mode="lines",
        name ='animal 1',
        showlegend=True), row=1, col=1
    )   
    fig.show()
    
def behavior_on(df, beh):
    behaviorname=''
    flag=False
    for name in beh:
        behaviorname= behaviorname + ' ,' + name
        if name in df.columns:
            flag=True
    if flag:
        behaviorSlice=df.loc[:,beh]
        TrueTimes = behaviorSlice.any(axis=1)
    else:
        print(behaviorname + ' not found in this trial')
    
    return(TrueTimes, flag)
    
