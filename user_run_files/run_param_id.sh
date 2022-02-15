# user inputs are defined in user_inputs.sh
source user_inputs.sh
mpiexec -n ${num_procs} ${opencor_pythonshell_path} ../src/scripts/param_id_run_script.py ${param_id_method} ${file_prefix} ${num_calls_to_function} ${input_params_to_id} ${param_id_obs_path} ${num_param_id_runs} > ../param_id_output/log.txt &
echo process running, output is in param_id_output/log.txt

