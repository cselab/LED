#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import pickle
import glob, os
import numpy as np
import argparse
from matplotlib.pyplot import cm

# ADDING PARENT DIRECTORY TO PATH
import os, sys, inspect

import pickle

def formatTime(time):
    time = "{:.2f}".format(time)
    # assert time[-3:] == ".00"
    # if time[-3:] == ".00": time = time[:-3]
    while time[-1:]=="0":
        time = time[:-1]
    if time[-1:] == ".": time = time[:-1]
    return time
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
        
def saveDataPickle(data, data_path, add_file_format=False):
    if add_file_format: data_path += ".pickle"
    with open(data_path, "wb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        del data
    return 0

def loadDataPickle(data_path, add_file_format=False):
    if add_file_format: data_path += ".pickle"
    try:
        with open(data_path, "rb") as file:
            data = pickle.load(file)
    except Exception as inst:
        print("[utils] Datafile\n {:s}\nNOT FOUND.".format(data_path))
        raise ValueError(inst)
    return data



def getSavingPath(Experiment_Name, system_name, global_params):
    if Experiment_Name is None or Experiment_Name == "None" or global_params.cluster == "daint":
        saving_path = global_params.saving_path.format(system_name)
    else:
        saving_path = global_params.saving_path.format(Experiment_Name + "/" + system_name)
    return saving_path


def getValueFromName(model_name, abrev, type_):
    for hyper_value in model_name.split("-"):
        idx = hyper_value.find(abrev)
        len_ = len(abrev)
        # print(idx)
        if idx == 0:
            value = hyper_value[idx + len_:]
            value = type_(value)
    return value


def sortModelList(modellist, COLUMN_TO_SORT, descenting=True):
    modellist_array = np.array(modellist)
    # print(np.shape(modellist_array))
    list_ = modellist_array[:, COLUMN_TO_SORT].copy()
    list_ = np.array([float(x) for x in list_])
    idx = list_.argsort()
    if descenting: idx = idx[::-1]
    modellist_sorted = modellist_array[idx]
    return modellist_sorted


def parseModelFields(logfile_path, FIELDS, PYTHON_TYPES, filename):
    # print(logfile_path)
    # print(FIELDS)
    # print(PYTHON_TYPES)
    # print(filename)
    assert (FIELDS[0] == "model_name")
    # Adding the .txt ending
    if not (filename[-4:] == ".txt"): filename = filename + ".txt"
    model_list = []
    model_dict = {}
    MAX_FILES = 1e10
    for root, dirs, files in os.walk(logfile_path, topdown=False):
        for model_ in dirs:
            model_entry = len(FIELDS) * [None]
            model_path = os.path.join(root, model_)
            file_path = model_path + "/" + filename
            # print(file_path)
            # print(os.path.exists(file_path))
            if (os.path.exists(file_path)):
                with open(file_path, 'r') as file_object:
                    lines = file_object.readlines()

                """ Get last line (last reported model) """
                line = lines[-1]

                elements = line[:-1].split(":")
                dict_ = {}
                element_names = elements[::2]
                element_values = elements[1::2]
                
                for i in range(len(element_names)):
                    # assert(element_names[i] not in dict_)
                    if element_names[i] in dict_:
                        raise ValueError(
                            "Duplicate entry found in logfile.")
                    dict_[element_names[i]] = element_values[i]
                i_ = 0

                valid_model = True
                for i in range(len(FIELDS)):
                    field_name = FIELDS[i]
                    if field_name not in element_names:
                        print("[SOFT ERROR] field {:} not found in logfile of model {:}. Ignoring model.".format(field_name, element_values[0]))
                        valid_model = False

                if valid_model:

                    for i in range(len(FIELDS)):
                        field_name = FIELDS[i]
                        assert field_name in element_names, "field_name={:} not found in {:}".format(field_name, element_names)
                        model_processed = True
                        idx_ = element_names.index(field_name)
                        if element_values[idx_] is None or element_values[idx_]=="None":
                            model_entry[i_] = element_values[idx_]
                        else:
                            model_entry[i_] = PYTHON_TYPES[i_](element_values[idx_])
                        i_ = i_ + 1
                        
                    model_list.append(model_entry)
                    model_name = model_entry[0]
                    model_dict[model_name] = model_entry[1:]
                    if len(model_list) > MAX_FILES:
                        break
    # print(len(model_list))
    # print(len(model_dict))
    # print(model_list)
    return model_list, model_dict


def getColors():
    # name = "tab10"
    # cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    # colors = cmap.colors  # type: list
    # # ax.set_prop_cycle(color=colors)
    colors = [
    'tab:red', 
    'tab:blue',
    'tab:green', 
    'tab:purple', 
    'tab:brown', 
    'tab:orange',
    'tab:pink', 
    'tab:gray', 
    'tab:olive', 
    'tab:cyan',
    ]
    # colors = [
    #     "blue",
    #     "green",
    #     "red",
    #     "orange",
    #     "blueviolet",
    #     "black",
    #     "cornflowerblue",
    #     "yellow",
    # ]
    markers = [
        "s",
        "x",
        "o",
        "d",
        "*",
        "<",
        ">",
        ">",
    ]
    return colors, markers
