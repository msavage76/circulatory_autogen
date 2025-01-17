# user inputs are defined in user_inputs.sh
source user_inputs.sh

# run code generations without param id parameters
${opencor_pythonshell_path} ../src/scripts/script_generate_with_new_architecture.py ${file_prefix}_vessel_array.csv ${input_param_file} ${param_id_output_dir} ${file_prefix}

# run parameter identification, don't output to log
mpiexec -n ${num_procs} ${opencor_pythonshell_path} ../src/scripts/param_id_run_script.py ${param_id_method} ${file_prefix} ${num_calls_to_function}

# run code generation with new param_id parameters 
${opencor_pythonshell_path} ../src/scripts/script_generate_with_new_architecture.py ${file_prefix}_vessel_array.csv ${input_param_file} ${param_id_output_dir} ${file_prefix} ${file_prefix}_${param_id_method}
