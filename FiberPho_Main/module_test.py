import io
import param
import panel as pn
import pandas as pd
import csv
import numpy as np
import os
import sys
import ipywidgets as ipw
import time
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
from tornado.ioloop import IOLoop
import logging
import traceback
from playsound import playsound # pip install playsound==1.2.2 not the newest version
import FiberClass as fc
# Saves selected object to pickle file

def run_delete_fiberobj(event = None):
    for obj in delete_obj_selecta.value:
        try:
            del fiber_objs[obj]
        except Exception as e:
            logger.error(traceback.format_exc())
            pn.state.notifications.error(
                'Error: Please check logger for more info', duration = 4000)
            print("Error: Cannot delete " + obj + ", please try again.")
            continue
        fiber_data.drop([obj], axis = 0, inplace = True)
    info_table.value = fiber_data
    existing_objs = fiber_objs
    # Updates all cards with new objects
    update_obj_selectas(existing_objs)
    return

def make_delete_box():
    delete_obj_selecta = pn.widgets.MultiSelect(name = 'Fiber Objects',
                                          value = [], options = [])

    #Buttons
    delete_obj_btn = pn.widgets.Button(name = 'Delete Object',
                                   button_type = 'danger', width = 500,
                                   sizing_mode = 'stretch_width',
                                   align = 'end')
    delete_obj_btn.on_click(run_delete_fiberobj) #Button action

    #Box
    delete_obj_box = pn.WidgetBox('# Delete unwanted Fiber Objects', 
                              delete_obj_selecta, delete_obj_btn)
    return delete_obj_box

def update_selecta(existing_objs):
    delete_obj_selecta.options = [*existing_objs]