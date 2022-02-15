## general inputs
file_prefix=FTU_wCVS
# file_prefix=3compartment
# file_prefix=simple_physiological
# file_prefix=physiological
input_param_file=FTU_wCVS.csv # this must be stored in resources.
# input_param_file=3compartment_parameters.csv # this must be stored in resources.
# input_param_file=simple_physiological_parameters.csv # this must be stored in resources.
# input_param_file=physiological_parameters.csv # this must be stored in resources.
                                     # If first creating a model
                                     # set this to parameters_orig.csv and a 
                                     # parameters file will be generated with 
                                     # spaces for the required parameters

## parameter identification inputs
## param_id_method can be any of [genetic_algorithm, bayesian]
param_id_method=genetic_algorithm
# num_procs=31
num_procs=3
num_calls_to_function=1000
num_param_id_runs=3

# This for 3compartment
param_id_obs_path=/home/finbar/Documents/data/cardiohance_data/cardiohance_observables.json 
# param_id_obs_path=/people/farg967/Documents/data/cardiohance_data/cardiohance_observables.json

# This for simple_physiological (doesn't use experimental data)
# param_id_obs_path=/home/finbar/Documents/git_projects/circulatory_autogen/resources/simple_physiological_obs_data.json

# This for physiological
# param_id_obs_path=/home/finbar/Documents/data/cardiohance_data/cardiohance_observables_with_ADAN_flows.json 
# param_id_obs_path=/people/farg967/Documents/data/cardiohance_data/cardiohance_observables_with_ADAN_flows.json
input_params_to_id=True # Whether an input file for params to identify is given in resources
                        # This should be named {file_prefix}_params_to_id.csv
                        # If false the params to identify will be chosen automatically.


## paths
## the below for my hpc
# opencor_pythonshell_path=/hpc/farg967/OpenCOR-2021-07-04-Linux/pythonshell
## the below for my local
opencor_pythonshell_path=/opt/OpenCOR-2021-10-05-Linux/pythonshell
## Users should modify opencor_pythonshell_path to the path of their own opencor pythonshell

