'''
Created on 13/12/2021

@author: Finbar J. Argus

Modified by M Savage for sensitivity and identifiability analysis
'''

import sys
import os
from mpi4py import MPI
import numpy as np

root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'src/param_id')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

from param_id.paramID import CVS0DParamID
import traceback

if __name__ == '__main__':

    try:
        mpi_debug = False

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        print(f'starting script for rank = {rank}')

        # FOR MPI DEBUG WITH PYCHARM
        # You have to change the configurations to "python debug server/mpi" and
        # click the debug button as many times as processes you want. You
        # must but the ports for each process in port_mapping.
        if mpi_debug:
            import pydevd_pycharm

            port_mapping = [36939, 44271, 33017, 46467]
            pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

        if len(sys.argv) != 4:
            print(f'incorrect number of inputs to sensitivity_run_script.py')
            exit()

        param_id_method = sys.argv[1]
        file_name_prefix = sys.argv[2]
        model_path = os.path.join(generated_models_dir_path, f'{file_name_prefix}.cellml')
        param_id_model_type = 'CVS0D' # TODO make this an input variable eventually

        input_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_id.csv')
        sensitivity_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_sensitivity.csv')
        if not os.path.exists(sensitivity_params_path):
            sensitivity_params_path = input_params_path
        sensitivity_output_paths = os.path.join(resources_dir_path, f'{file_name_prefix}_sensitivity_output_paths.csv')

        param_id_obs_path = sys.argv[3]
        if not os.path.exists(param_id_obs_path):
            print(f'param_id_obs_path={param_id_obs_path} does not exist')
            exit()


        # set the simulation time where the cost is calculated (sim_time) and the amount of 
        # simulation time it takes to get to an oscilating steady state before that (pre_time)
        if file_name_prefix == '3compartment' or 'FTU_wCVS':
          pre_time = 6.0
        else: 
          pre_time = 20.0
        sim_time = 2.0

        param_id = CVS0DParamID(model_path, param_id_model_type, param_id_method, file_name_prefix,
                                input_params_path=input_params_path, sensitivity_params_path=sensitivity_params_path,
                                param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time, maximumStep=0.0004, DEBUG=True)

        param_id.run_sensitivity(sensitivity_output_paths)
        param_id.close_simulation()

    except:
        print(traceback.format_exc())
        print("Usage: param_id_method file_name_prefix path_to_obs_file.json")
        print("e.g. genetic_algorithm simple_physiological path/to/circulatory_autogen/resources/simple_physiological_obs_data.json")
        comm.Abort()
        exit
