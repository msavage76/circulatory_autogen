'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI

root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'src/param_id')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

from param_id.paramID import CVS0DParamID
from utilities import obj_to_string
import traceback

if __name__ == '__main__':

    try:

        if len(sys.argv) != 5:
            print(f'incorrect number of inputs to param_id_run_script.py')
            exit()
        param_id_method = sys.argv[1]
        file_name_prefix = sys.argv[2]
        model_path = os.path.join(generated_models_dir_path, f'{file_name_prefix}.cellml')
        param_id_model_type = 'CVS0D' # TODO make this an input variable eventually

        input_params_to_id = sys.argv[3]
        if input_params_to_id:
            input_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_id.csv')
        else:
            input_params_path = False
        param_id_obs_path = os.path.join(resources_dir_path, sys.argv[4])

        param_id = CVS0DParamID(model_path, param_id_model_type, param_id_method, file_name_prefix,
                                input_params_path=input_params_path, param_id_obs_path=param_id_obs_path)

        # print(obj_to_string(param_id))
        param_id.simulate_with_best_param_vals()
        param_id.plot_outputs()
        param_id.close_simulation()

    except:
        print(traceback.format_exc())
        print("Usage: param_id_method file_name_prefix input_params_to_id")
        print("e.g. bayesian simple_physiological True")
        exit()
