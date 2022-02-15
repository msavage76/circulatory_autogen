'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

import pandas as pd
import numpy as np
import os
import csv
import re

class CSVFileParser(object):
    '''
    Parses CSV files
    '''

    def __init__(self):
        '''
        Constructor
        '''
        
    def get_data_as_dataframe_multistrings(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe where entries in the data array that have two
        entries are put in a list in the entry for the dataframe
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''
        if( has_header ):
            csv_dataframe = pd.read_csv(filename, dtype=str)
        else:
            csv_dataframe = pd.read_csv(filename, dtype=str, header=None)

        csv_dataframe = csv_dataframe.rename(columns=lambda x: x.strip())
        for II in range(csv_dataframe.shape[0]):
            for column_name in csv_dataframe.columns:
                entry = csv_dataframe[column_name][II]
                sub_entries = entry.split()
                if column_name in ['vessel_name', 'inp_vessels', 'out_vessels']:
                    new_entry = [sub_entry.strip() for sub_entry in sub_entries]
                else:
                    new_entry = sub_entries[0].strip()

                csv_dataframe[column_name][II] = new_entry

        # for column_name in csv_dataframe.columns:
        #     if column_name == 'vessel_name':
        #         continue
        #     csv_dataframe[column_name] = csv_dataframe[column_name].str.strip()
    
        return csv_dataframe

    def get_data_as_dataframe(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''
        if (has_header):
            csv_dataframe = pd.read_csv(filename, dtype=str)
        else:
            csv_dataframe = pd.read_csv(filename, dtype=str, header=None)

        for column_name in csv_dataframe.columns:
            csv_dataframe[column_name] = csv_dataframe[column_name].str.strip()

        return csv_dataframe

    def get_data_as_nparray(self,filename,has_header=True):
        '''
        Returns the data in the CSV file as a numpy array
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''

        csv_dataframe = self.get_data_as_dataframe(filename, has_header)
    
        csv_np_array = csv_dataframe.to_numpy()
        dtypes = []
        for column in list(csv_dataframe.columns):
            dtypes.append((column,'<U64'))
            
        csv_np_array = np.array(list(zip(*csv_np_array.T)), dtype=dtypes)
    
        return csv_np_array

    def get_data_as_dictionary(self,filename):
        '''
        Returns the data in the CSV file as a Python dictionary
        :param filename: filename of CSV file
        '''

        csv_dataframe = self.get_data_as_dataframe(filename).T
        csv_dictionary = csv_dataframe.to_dict()
        
        return list(csv_dictionary.values())


    def get_param_id_params_as_lists_of_tuples(self, param_id_dir):

        param_state_names = []
        param_const_names = []

        # param names that were identified in param_id
        with open(os.path.join(os.path.join(param_id_dir, 'param_state_names_for_gen.csv')), 'r') as f:
            rd = csv.reader(f)
            for row in rd:
                param_state_names.append(row)
        with open(os.path.join(os.path.join(param_id_dir, 'param_const_names_for_gen.csv')), 'r') as f:
            rd = csv.reader(f)
            for row in rd:
                param_const_names.append(row)

        param_names = param_state_names + param_const_names
        # get date identifier of the parameter id
        date_id = np.load(os.path.join(os.path.join(param_id_dir, 'date.npy'))).item()

        param_vals = np.load(os.path.join(param_id_dir, 'best_param_vals.npy'))
        state_param_name_and_val = []
        const_param_name_and_val = []
        # this only looks at the first param_vals relating to param_state_names, not to constants
        for name_or_list, val in zip(param_names, param_vals):
            if name_or_list in param_state_names:
                if isinstance(name_or_list, list):
                    for name in name_or_list:
                        state_param_name_and_val.append((name, val))
                else:
                    state_param_name_and_val.append((name, val))
            elif name_or_list in param_const_names:
                if isinstance(name_or_list, list):
                    for name in name_or_list:
                        const_param_name_and_val.append((name, val))
                else:
                    const_param_name_and_val.append((name, val))

            else:
                print('error, exiting')
                exit()

        return state_param_name_and_val, const_param_name_and_val, date_id

