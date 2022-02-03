import sys
import os
import io
import csv
import pandas as pd
import pickle
from FiberClass import fiberObj

def read_csv(file_in):
    # print(file_in.filename)
    # print(file_in.value)
    value = file_in.value
    if value is not None:
        try:
            string_io = io.StringIO(value.decode("utf8"))
            df = pd.read_csv(string_io)
            # print(file_in.filename)
        except FileNotFoundError:
            print("Could not find file: " + file_in)
            sys.exit(2)
        except PermissionError:
            print("Could not access file: " + file_in)
            sys.exit(3)
        
    if df.empty:
        print("Dataframe is empty")
        sys.exit(4)
    else:
        return df

def createObj(file, obj, fiber, animal, date, time):
    testObj = fiberObj(file, obj, fiber, animal, date, time)
    return testObj

def createPkl(pkl_obj):
    if pkl_file is None:
        print("No object passed for pickling")
        return 7
    else:
        #Create a pickle file
        picklefile = open('pkl_obj', 'wb')
        #Pickle the dictionary and write to file
        pickle.dump(pkl_obj, picklefile)
        #Close file
        picklefile.close()