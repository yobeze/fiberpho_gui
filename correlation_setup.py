"""Library of functions for synchrony analysis
    * within_trial_pearsons
"""

import sys
import scipy.stats as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import butter, filtfilt

#return the pearsons correlation coefficient and r value between 2 full channels and plots the signals overlaid and thier scatter plot
def within_trial_pearsons(df, key, channels):
    results={}
    for channel in channels:
        our_channels = [col for col in df.columns if channel + ' final normalized' in col]
        if len(our_channels) < 1:
            print('data must be normalized before running a correlation')
            sys.exit()
        sig1 = df[our_channels[0]]
        sig2 = df[our_channels[1]]
        
        time = df['fTimeGreen']
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
def behavior_specific_pearsons(df, file, channels, behs):
    results={}
    for channel in channels:
        our_channels = [chan + ' final normalized' for chan in channel ]
        channel_names=channel[0]+ ' vs ' +channel[1]
        if len(our_channels) < 1:
            print('data must be normalized before running a correlation')
            sys.exit()
        results[channel_names]={} 
        sig1=[]
        sig2=[]
        for beh in behs:
            behaviorname=''
            flag=False
            for name in beh:
                behaviorname= behaviorname + ' ,' + name
                if name in df.columns:
                    flag=True
            if flag:
                behaviorSlice=df.loc[:,beh]
                TrueTimes = behaviorSlice.any(axis=1);
                corDf=pd.concat([df['fTimeGreen'], df[our_channels[0]], df[our_channels[1]], TrueTimes], axis=1)
                corDf.columns = ['fTimeGreen', our_channels[0], our_channels[1], 'TrueTimes']
                
                sig1=corDf[corDf.TrueTimes == True][our_channels[0]].tolist()
                sig2=corDf[corDf.TrueTimes == True][our_channels[1]].tolist()

                
                #difsig1=[sig1.iloc[i+1]-sig1.iloc[i] for i in range(len(sig1)-1)]
                #difsig2=[sig2.iloc[i+1]-sig2.iloc[i] for i in range(len(sig2)-1)]
                #sig1 = ss.zscore(uniform_filter1d(sig1, size=50))
                #sig2 = ss.zscore(uniform_filter1d(sig2, size=50))

                time=corDf[corDf.TrueTimes == True]['fTimeGreen']
                fig = make_subplots(rows=1, cols=2)
                fig.add_trace(
                    go.Scatter(
                    x=sig1,
                    y=sig2,
                    mode="markers",
                    name =behaviorname,
                    showlegend=False), row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(
                    x=time,
                    y=sig1,
                    mode="lines",
                    line=go.scatter.Line(color='rgb(255,100,150)'),
                    name =channel[0],
                    showlegend=False), row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                    x=time,
                    y=sig2,
                    mode="lines",
                    line=go.scatter.Line(color='rgba(100,0,200, .6)'),
                    name = channel[1],
                    showlegend=False), row=1, col=1
                )
                fig.update_yaxes(
                    #title_text = "Interbrain Correlation (PCC)",
                    showgrid=True,
                    #showline=True, linewidth=2, linecolor='black',
                )
                fig.update_layout(
                        title=channel_names + ' while' + behaviorname + ' for ' + file
                        )
                fig.update_xaxes(title_text=channel[0]+ ' zscore')
                fig.update_yaxes(title_text=channel[1] + ' zscore')
                
                fig.update_xaxes(title_text='Time (s)', col=1, row=1)
                fig.update_yaxes(title_text='Zscore', col=1, row=1)

                fig.show()
                #fig.write_image('together_seperate1.pdf')
                [r, p] = ss.pearsonr(sig1, sig2)
                #print(sig1.iloc[0:len(sig1)*(1/3)])
                #print(sig1.type())
                beg = ss.pearsonr(sig1[0:int(len(sig1)*(1/3))], sig2[0:int(len(sig1)*(1/3))])
                mid = ss.pearsonr(sig1[int(len(sig1)*(1/3)):int(len(sig1)*(2/3))], sig2[int(len(sig1)*(1/3)):int(len(sig1)*(2/3))])
                end = ss.pearsonr(sig1[int(len(sig1)*(2/3)):], sig2[int(len(sig1)*(2/3)):])
            else:
                [r, p]=['na', 'na']
                print(behaviorname + ' not found in this trial')
            results[channel_names][behaviorname]={'full':r, 'start':beg, 'middle':mid, 'end':end}  
    print(file)
    print(results)
    return 
    
def plot_FFT(df, key, channels):
    from scipy.fft import fft, fftfreq
    fig = make_subplots(rows=1, cols=1)
    for channel in channels:
        sig1 = df[channel + ' final normalized']
        time = df['fTimeGreen']
        # Number of sample points
        N = len(sig1)
        # sample spacing
        T = time.iloc[-1]/(N*300)
        y1 = fft(sig1.tolist())
        y1=2.0/N * np.abs(y1[0:N//2]),
        xf = fftfreq(N, T)[:N//2]

        y1binned=np.arange(0, np.floor(xf[-1]))
        xbinned=np.arange(0, np.floor(xf[-1]), 1)
        counter = 0
        idx=-1
        y1tot=0
        for i in xf:
            idx=idx+1
            if i < counter+1:
                y1tot=y1tot+y1[0][idx]
            else:
                y1binned[counter]=y1tot
                counter=counter+1
                y1tot=0

        fig.add_trace(
            go.Scatter(
            x=xf,
            y=y1[0],
            mode="lines",
            name = channel,
            showlegend=True), row=1, col=1
        )
        fig.add_trace(
             go.Scatter(
             x=xbinned,
             y=y1binned,
             mode="lines",
             name = channel+ 'binned',
             showlegend=True), row=1, col=1
         )
    fig.show()
    

def get_freq(freq, sig, time): 
    N= len(time)
    start=time[0]
    end=time[N-1]
    length=end-start
    fs= N/length #sampling rate
    lowcut=freq-0.5
    highcut=freq+0.5
    nyq=fs/2
    low= lowcut/nyq
    high = highcut/nyq
    order= 2
    b, a = butter(order, [low, high], 'bandpass', analog=False)
    filtered_sig=filtfilt(b, a, sig, axis=0)
    return filtered_sig

def all_syncs(df, key, channels):
    time = df[df.together == True]['fTimeGreen']
    sig1 = df[df.together == True]['f1GreenGreen final normalized']
    sig2 = df[df.together == True]['f2GreenGreen final normalized']
    time = np.array(time)
    sig1 = np.array(sig1)
    sig2 = np.array(sig2)
    correlations={}
    for freq in range(19):
        freq=freq+1
        fsig1=get_freq(freq, sig1, time)
        fsig2=get_freq(freq, sig2, time)
        
        correlations[freq] = ss.pearsonr(fsig1, fsig2)
        fig = make_subplots(rows=1, cols=1)
        #plots sig1
        fig.add_trace(
            go.Scatter(
            x=time,
            y=fsig1,
            mode="lines",
            name='sig1',
            showlegend=False), row=1, col=1
        )
        #plots sig2
        fig.add_trace(
            go.Scatter(
            x=time,
            y=fsig2,
            mode="lines",
            name = "sig2",
            showlegend=False), row=1, col=1
        )
        fig.update_layout(title='frequency=' + str(freq))
        fig.show()
    correlations=pd.DataFrame.from_dict(correlations, orient='index')
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
        y=correlations[0],
        mode="lines",
        name = "sig2",
        showlegend=False), row=1, col=1
    )
    r,p=ss.pearsonr(sig1, sig2)
    fig.update_layout(title=key+'pearsons = '+ str(r))
    fig.show()
    print()
    return(correlations)

def sliding_window_corr_pause(df, key, channels):
    results={}
    times={}
    our_channels = [chan + ' final normalized' for chan in channels[0]]
    channel=channels[0][0]+ ' vs ' +channels[0][1]
    if len(our_channels) < 1:
        print('data must be normalized before running a correlation')
        sys.exit()
    #get the signal from each channel
    sig1 = df[our_channels[0]]
    sig2 = df[our_channels[1]]
    #get the time for the channels
    time = df['fTimeGreen']
    #initialize the results dict
    results[channel] = {}
    #initializes the length of the first list of corrs for the results
    firstlen=0
    # loop over window sizes starting to ending with a certain step size
    for win in range(1000, 4000, 500):
        print('starting window size', time[win+1]-time[1], "seconds")
        #initializes the times and results 
        results[channel][win]=[]
        if firstlen < 1:
            times[channel]=[]
        i=0
        nancount=0
        #loops until the end of our window is at the end of the signal
        while (i + win) < len(sig1):
            #calculates the pearsons R
            if firstlen < 1:
                times[channel].append(time[i+int(np.ceil(win/2))])
            [r, p] = ss.pearsonr(sig1[i: i + win], sig2[i: i + win])
            if p > 0.05:
                nancount=nancount+1
                results[channel][win].append(float("nan"))
                #print('p=', p, 'for window size=', win, 'at time=', time[i+np.floor(win/2)])
            else:
                results[channel][win].append(r)
            i=i+10
        print(nancount/len(results[channel][win]))
        if firstlen > 0:
            short=firstlen-len(results[channel][win])
            zeros=[float("nan") for i in range(int((short)/2))]
            results[channel][win]=zeros + results[channel][win] + zeros
            print(len(results[channel][win]))                            
        else:
            firstlen=len(results[channel][win])
    #print(times[channel])
    fig = go.Figure(data=go.Heatmap(x=list(times[channel]), y=list(results[channel].keys()), z=list(results[channel].values()), zmin=-1, zmax=1))
    fig.show()
    #returns the pearsons R
    
    np.save('winCorResults.npy', results)
    np.save('winCorTimes.npy', times)
    return results, times

def sliding_window_corr(df, key, channels):
    win=100 #window size can be moved to config
    step=3 #step size can be moved to config
    results={}
    times={}
    nanresults={}
    nantimes={}
    our_channels = [chan + ' final normalized' for chan in channels[0]]
    channel=channels[0][0]+ ' vs ' +channels[0][1]
    if len(our_channels) < 1:
        print('data must be normalized before running a correlation')
        sys.exit()
    #get the signal from each channel
    sig1 = df[our_channels[0]]
    sig2 = df[our_channels[1]]
    #get the time for the channels
    time = df['fTimeGreen']
    #initialize the results dict
    
    #initializes the length of the first list of corrs for the results
    # loop over window sizes starting to ending with a certain step size
    print('starting window size', time[win+1]-time[1], "seconds")
    #initializes the times and results 
    results[channel] = []
    times[channel]=[]
    nanresults[channel]=[]
    nantimes[channel]=[]
    i=0
    nancount=0
    #loops until the end of our window is at the end of the signal
    while (i + win) < len(sig1):
        #calculates the pearsons R
        [r, p] = ss.pearsonr(sig1[i: i + win], sig2[i: i + win])
        if p > 0.05:
            nancount=nancount+1
            nanresults[channel].append(r)
            nantimes[channel].append(time[i+int(np.ceil(win/2))])
            #print('p=', p, 'for window size=', win, 'at time=', time[i+np.floor(win/2)])
        results[channel].append(r)
        times[channel].append(time[i+int(np.ceil(win/2))])
        i=i+step
    print(nancount/len(results[channel]))
    #print(times[channel])
    fig = make_subplots(rows=1, cols=1, subplot_titles=[key]) #subplot_titles=[channel for channel in channels], shared_xaxes=True)
    #plot each signal 
    fig.add_trace(
        go.Scatter(          
        x=times[channel],
        y=results[channel],
        mode="lines",
        line=go.scatter.Line(color='Black', width=3),
        name =str(time[win+1]-time[1]),
        showlegend=True), row=1, col=1
        )
    fig.add_trace(
        go.Scatter(          
        x=nantimes[channel],
        y=nanresults[channel],
        mode="markers",
        marker=dict(color='Grey',size=3),
        name =str(time[win+1]-time[1]),
        showlegend=True), row=1, col=1
        )
    #find all coloumns in the dataframe for behaviors
    behaviors = df.select_dtypes(include=['bool']).columns
    behaviors = ["active interaction", "one sided interaction", "passive interaction"]
    #a list of colors to use for each behavior
    colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    #for each behavior
    for j, beh in enumerate(behaviors):
        flag = False
        #iterate over the len of the data frame
        for k in range(len(sig1)):
            #if the current value is true and the previous was false you've found a behavior start
            if df.at[k, beh] == True:
                if flag == False:
                    start = df.at[k, 'fTimeGreen'] 
                    flag = True
            else:
                #if the current value is false and the previous is true you've found an end
                if flag == True:
                    end = df.at[k, 'fTimeGreen'] 
                    #adds a color cdoded rectangle over this bout of behavior
                    fig.add_vrect(
                        x0=start, x1=end, opacity=0.75,
                        line_width=0, 
                        layer="below",
                        fillcolor=colors[j%10],
                        row=1, col=1,
                        name=beh
                        )
                    flag = False
        #if the last value is true then end the last bout on the last frame
        if flag == True:
            end = df.at[k, 'fTimeGreen'] 
            #adds a color coded rectangle over a bout of behavior that end on the last frame
            fig.add_vrect( x0=start, x1=end, 
                    opacity=0.75,
                    layer="below",
                    line_width=0, 
                    fillcolor=colors[j%10],
                    row=1, col=1,
                    name=beh
                    )
        # adds the name of the behavior to the graph in the color of the rectangles
        fig.add_annotation(xref="x domain", yref="y domain",
            x=1, 
            y=(j+1)/len(behaviors),
            text=beh,
            bgcolor = colors[j%10],
            showarrow=False,
            row=1, col=1
            )
    #shows the plots
    fig.show()
    #returns the pearsons R
    
    #np.save('winCorResults.npy', results)
    #np.save('winCorTimes.npy', times)
    return results, times
