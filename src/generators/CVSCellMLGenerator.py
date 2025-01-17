'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

import numpy as np
import re
import pandas as pd
import os
from sys import exit
generators_dir_path = os.path.dirname(__file__)


class CVS0DCellMLGenerator(object):
    '''
    Generates CellML files for the 0D model represented in @
    '''


    def __init__(self, model, output_path, filename_prefix):
        '''
        Constructor
        '''
        self.model = model
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.filename_prefix = filename_prefix
        self.user_resources_path = os.path.join(generators_dir_path, '../../resources')
        self.base_script = os.path.join(generators_dir_path, 'resources/base_script.cellml')
        self.modules_script = os.path.join(generators_dir_path, 'resources/BG_modules.cellml')
        self.units_script = os.path.join(generators_dir_path, 'resources/units.cellml')

    def generate_files(self):
        if(type(self.model).__name__ != "CVS0DModel"):
            print("Error: The model should be a CVS0DModel representation")
            return
        
        print("Generating model files at {}".format(self.output_path))
        
        #    Code to generate model files
        self.__generate_CellML_file()
        if self.model.param_id_consts:
            self.__modify_parameters_array_from_param_id()
        self.__generate_parameters_csv()
        self.__generate_parameters_file()
        self.__generate_modules_file()
        self.__generate_units_file()

        # TODO check that model generation is succesful, possibly by calling to opencor
        print('Model generation complete.')
        print('Testing to see if model opens in OpenCOR')
        opencor_available = True
        try:
            import opencor as oc
        except:
            opencor_available = False
            pass
        if opencor_available:
            sim = oc.open_simulation(os.path.join(self.output_path, f'{self.filename_prefix}.cellml'))
            if sim.valid():
                print('Model generation has been successful.')
            else:
                if self.model.all_parameters_defined:
                    print('The OpenCOR model is not yet working, The reason for this is unknown.\n')
                else:
                    print('The OpenCOR model is not yet working because all parameters have not been given values, \n'
                          f'Enter the values in '
                          f'{os.path.join(self.user_resources_path, f"{self.filename_prefix}_parameters_unfinished.csv")}')

        else:
            print('Model generation is complete but OpenCOR could not be opened to test the model. \n'
                  'If you want this check to happen make sure you use the python that is shipped with OpenCOR')


    def __generate_CellML_file(self):
        print("Generating CellML file {}.cellml".format(self.filename_prefix))
        if self.model.param_id_states:
            state_modified = [False]*len(self.model.param_id_states) #  whether this state has been found and modified
        with open(self.base_script, 'r') as rf:
            with open(os.path.join(self.output_path,f'{self.filename_prefix}.cellml'), 'w') as wf:
                for line in rf:
                    # TODO when heart and pulmonary are modules the state modification
                    #  will be done in __generate_modules()
                    if self.model.param_id_states:
                        for idx, (state_name, val) in enumerate(self.model.param_id_states):
                            if state_name in line and 'initial_value' in line:
                                inp_string = f'initial_value="{val:.4e}"'
                                line = re.sub('initial_value=\"\d*\.?\d*e?-?\d*\"', inp_string, line)
                                state_modified[idx] = True

                    if 'import xlink:href="units.cellml"' in line:
                        line = re.sub('units', f'{self.filename_prefix}_units', line)
                    elif 'import xlink:href="parameters_autogen.cellml"' in line:
                        line = re.sub('parameters_autogen', f'{self.filename_prefix}_parameters', line)
                    # copy the start of the basescript until line that says #STARTGENBELOW
                    wf.write(line)
                    if '#STARTGENBELOW' in line:
                        break

                # check if each state was modified
                if self.model.param_id_states:
                    if any(state_modified) == False:
                        false_states = [self.model.param_id_states[JJ][0] for JJ in range(len(state_modified)) if state_modified[JJ] == False]
                        print(f'The parameter id states {false_states} \n'
                              f'were not found in the cellml script, check the parameter id state names and the '
                              f'base_script.cellml file')
                        exit()
    
                ###### now start generating own code ######

                # import vessels
                print('writing imports')
                self.__write_section_break(wf, 'imports')
                self.__write_imports(wf, self.model.vessels)
    
                # define mapping between vessels
                print('writing vessel mappings')
                self.__write_section_break(wf, 'vessel mappings')
                self.__write_vessel_mappings(wf, self.model.vessels)
    
                # create computation environment to sum flows from terminals
                # to have a total flow input into each first venous component.
                print('writing environment to sum venous input flows')
                self.__write_section_break(wf, 'terminal venous connection')
                self.__write_terminal_venous_connection_comp(wf, self.model.vessels)
    
                # define variables so they can be accessed
                print('writing variable access')
                self.__write_section_break(wf, 'access_variables')
                self.__write_access_variables(wf, self.model.vessels)
    
                # map between computational environment and module so they can be accessed
                print('writing mappings between computational environment and modules')
                self.__write_section_break(wf, 'vessel mappings')
                self.__write_comp_to_module_mappings(wf, self.model.vessels)
    
                # map constants to different modules
                print('writing mappings between constant params')
                self.__write_section_break(wf, 'parameters mapping to modules')
                self.__write_param_mappings(wf, self.model.vessels, params_array=self.model.parameters)
    
                # map environment time to module times
                print('writing writing time mappings between environment and modules')
                self.__write_section_break(wf, 'time mapping')
                self.__write_time_mappings(wf, self.model.vessels)
    
                # Finalise the file
                wf.write('</model>\n')
            
    
    def __generate_parameters_file(self):
        print("Generating CellML file {}_parameters.cellml".format(self.filename_prefix))
        """
        Takes in a data frame of the params and generates the parameter_cellml file
        """

        with open(os.path.join(self.output_path, f'{self.filename_prefix}_parameters.cellml'), 'w') as wf:

            wf.write('<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n')
            wf.write('<model name="Parameters" xmlns="http://www.cellml.org/cellml/1.1#'
                     '" xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n')
    
            heart_params_array = self.model.parameters[np.where(self.model.parameters["comp_env"]=='heart')]
            pulmonary_params_array = self.model.parameters[np.where(self.model.parameters["comp_env"]=='pulmonary')]
            systemic_params_array = self.model.parameters[np.where(self.model.parameters["comp_env"]=='systemic')]


            wf.write('<component name="parameters_pulmonary">\n')
            self.__write_constant_declarations(wf, pulmonary_params_array["variable_name"],
                                        pulmonary_params_array["units"],
                                        pulmonary_params_array["value"])
            wf.write('</component>\n')
            wf.write('<component name="parameters_heart">\n')
            self.__write_constant_declarations(wf, heart_params_array["variable_name"],
                                        heart_params_array["units"],
                                        heart_params_array["value"])
            wf.write('</component>\n')
            wf.write('<component name="parameters_systemic">\n')
            self.__write_constant_declarations(wf, systemic_params_array["variable_name"],
                                        systemic_params_array["units"],
                                        systemic_params_array["value"])
            wf.write('</component>\n')
            wf.write('</model>\n')

    def __modify_parameters_array_from_param_id(self):
        # first modify param_const names easily by modifying them in the array
        print('modifying constants to values identified from parameter id')
        for const_name, val in self.model.param_id_consts:
            self.model.parameters[np.where(self.model.parameters['variable_name'] ==
                                           const_name)[0][0]]['value'] = f'{val:.4e}'
            self.model.parameters[np.where(self.model.parameters['variable_name'] ==
                                           const_name)[0][0]]['data_reference'] = \
                f'{self.model.param_id_date}_identified'

    def __generate_parameters_csv(self):

        # check if all the required parameters have been defined, if not we make an "unfinished"
        # csv file which makes it easy for the user to include the required parameters
        if self.model.all_parameters_defined:
            file_to_create = os.path.join(self.output_path, f'{self.filename_prefix}_parameters.csv')
        else:
            file_to_create = os.path.join(self.user_resources_path,
                                          f'{self.filename_prefix}_parameters_unfinished.csv')
            print(f'\n WARNING \nRequired parameters are missing. \nCreating a file {file_to_create},\n'
                  f'which has EMPTY_MUST_BE_FILLED tags where parameters\n'
                  f'need to be included. The user should include these parameters then remove \n'
                  f'the "_unfinished" ending of the file name, then rerun the model generation \n'
                  f'with the new parameters file as input.\n')
        df = pd.DataFrame(self.model.parameters)
        df.to_csv(file_to_create, index=None, header=True)
    
    def __generate_units_file(self):
        # TODO allow a specific units file to be generated
        #  This function simply copies the units file
        print(f'Generating CellML file {self.filename_prefix}_units.cellml')
        with open(self.units_script, 'r') as rf:
            with open(os.path.join(self.output_path, f'{self.filename_prefix}_units.cellml'), 'w') as wf:
                for line in rf:
                    wf.write(line)

    def __generate_modules_file(self):
        print(f'Generating modules file {self.filename_prefix}_modules.cellml')
        with open(self.modules_script, 'r') as rf:
            # with open(self.output_path+f'{self.filename_prefix}_modules.cellml', 'w') as wf:
            with open(os.path.join(self.output_path, f'{self.filename_prefix}_modules.cellml'), 'w') as wf:
                for line in rf:
                    wf.write(line)

    def __write_section_break(self, wf, text):
        wf.write('<!--&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;' +
                text + '&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;&#45;//-->\n')

    def __write_imports(self, wf, vessel_array):
        for vessel_vec in vessel_array:
            if vessel_vec["vessel_type"] in ['heart', 'pulmonary']:
                continue
            self.__write_import(wf, vessel_vec)
        # add a zero mapping to heart ivc or svd flow input if only one input is specified
        if not vessel_array[np.where(vessel_array["name"] == 'heart')][0]["inp_vessel_2"]:
            wf.writelines([f'<import xlink:href="{self.filename_prefix}_modules.cellml">\n',
                           f'    <component component_ref="zero_flow" name="zero_flow_module"/>\n',
                           '</import>\n'])

    def __write_vessel_mappings(self, wf, vessel_array):
        for vessel_vec in vessel_array:
            # input and output vessels
            main_vessel = vessel_vec["name"]
            main_vessel_BC_type = vessel_vec["BC_type"]
            main_vessel_type = vessel_vec["vessel_type"]
            out_vessel = vessel_vec["out_vessel_1"]
            if out_vessel not in vessel_array["name"]:
                print(f'the output vessel of {out_vessel} is not defined')
                exit()
            out_vessel_vec = vessel_array[np.where(vessel_array["name"] == out_vessel)][0]
            out_vessel_BC_type = out_vessel_vec["BC_type"]
            out_vessel_type = out_vessel_vec["vessel_type"]
    
            # check that input and output vessels are defined as connection variables
            # for that vessel and they have corresponding BCs
            self.__check_input_output_vessels(vessel_array, main_vessel, out_vessel,
                                       main_vessel_BC_type, out_vessel_BC_type,
                                       main_vessel_type, out_vessel_type)
    
            # determine BC variables from vessel_type and BC_type
            if main_vessel_type == 'heart':
                v_1 = 'v_lv'
                p_1 = 'u_root'
            elif main_vessel_type == 'last_venous':
                v_1 = 'v'
                p_1 = 'u_out'
            elif main_vessel_type == 'terminal':
                # flow output is to the terminal_venous_connection, not
                # to the venous module
                v_1 = ''
                p_1 = 'u_out'
            elif main_vessel_type == 'split_junction':
                if main_vessel_BC_type.endswith('v'):
                    v_1 = 'v_out_1'
                    p_1 = 'u'
                elif main_vessel_BC_type == 'vp':
                    v_1 = 'v_out_1'
                    p_1 = 'u'
                elif main_vessel_BC_type == 'pp':
                    print('Currently we have not implemented junctions'
                          'with output pressure boundary conditions, '
                          f'change {main_vessel}')
                    exit()
            elif main_vessel_type == '2in2out_junction':
                if main_vessel_BC_type == 'vv':
                    v_1 = 'v_out_1'
                    p_1 = 'u_d'
                else:
                    print('2in2out vessels only have vv type BC, '
                          f'change "{main_vessel}" or create new BC module '
                          f'in BG_modules.cellml')
                    exit()
            elif main_vessel_type == 'merge_junction':
                if main_vessel_BC_type == 'vp':
                    v_1 = 'v'
                    p_1 = 'u_out'
                elif main_vessel_BC_type == 'vv':
                    v_1 = 'v_out'
                    p_1 = 'u_d'
                else:
                    print('Merge boundary condiditons only have vp or vv type BC, '
                          f'change "{main_vessel}" or create new BC module in '
                          f'BG_modules.cellml')
                    exit()
            else:
                if main_vessel_BC_type.endswith('v'):
                    v_1 = 'v_out'
                    p_1 = 'u'
                elif main_vessel_BC_type == 'vp':
                    v_1 = 'v'
                    p_1 = 'u_out'
                elif main_vessel_BC_type == 'pp':
                    v_1 = 'v_d'
                    p_1 = 'u_out'
    
            if out_vessel_type == 'heart':
                if main_vessel == 'venous_ivc':
                    v_2 = 'v_ivc'
                elif main_vessel == 'venous_svc':
                    v_2 = 'v_svc'
                else:
                    print('venous input to heart can only be venous_ivc or venous_svc')
                    exit()
                p_2 = 'u_ra'
            elif out_vessel_type in ['merge_junction', '2in2out_junction']:
                if out_vessel_vec["inp_vessel_1"] == main_vessel:
                    v_2 = 'v_in_1'
                elif out_vessel_vec["inp_vessel_2"] == main_vessel:
                    v_2 = 'v_in_2'
                else:
                    print('error, exiting')
                    exit()
                p_2 = 'u'
            elif main_vessel_type == 'terminal':
                # For terminal output we link to a terminal_venous connection
                # to sum up the output terminal flows
                if out_vessel_BC_type == 'vp':
                    v_2 = ''
                    p_2 = f'u'
                else:
                    print('venous section connected to terminal only works'
                          'for vp BC currently')
                    exit()
            else:
                if out_vessel_BC_type == 'vp':
                    v_2 = 'v_in'
                    p_2 = 'u'
                elif out_vessel_BC_type == 'vv':
                    v_2 = 'v_in'
                    p_2 = 'u_C'
                elif out_vessel_BC_type.startswith('p'):
                    v_2 = 'v'
                    p_2 = 'u_in'
    
            # TODO make heart and pulmonary BG modules so everything can be a module
            if main_vessel in ['heart', 'pulmonary']:
                main_vessel_module = main_vessel
            else:
                main_vessel_module = main_vessel + '_module'
            if out_vessel in ['heart', 'pulmonary']:
                out_vessel_module = out_vessel
            else:
                out_vessel_module = out_vessel + '_module'
    
            self.__write_mapping(wf, main_vessel_module, out_vessel_module, [v_1, p_1], [v_2, p_2])
    
            if vessel_vec["vessel_type"].endswith('junction'):
                if vessel_vec["vessel_type"] in ['split_junction', '2in2out_junction']:
                    out_vessel = vessel_vec["out_vessel_2"]
                    if out_vessel in ['heart', 'pulmonary']:
                        out_vessel_module = out_vessel
                    else:
                        out_vessel_module = out_vessel + '_module'
                    if vessel_vec["BC_type"].endswith('v'):
                        v_1 = 'v_out_2'
                    else:
                        pass
                if out_vessel not in vessel_array["name"]:
                    print(f'the output vessel of {out_vessel} is not defined')
                    exit()
                out_vessel_vec = vessel_array[np.where(vessel_array["name"] == out_vessel)][0]
                out_vessel_BC_type = out_vessel_vec["BC_type"]
                out_vessel_type = out_vessel_vec["vessel_type"]
                self.__check_input_output_vessels(vessel_array, main_vessel, out_vessel,
                                           main_vessel_BC_type, out_vessel_BC_type,
                                           main_vessel_type, out_vessel_type)
                self.__write_mapping(wf, main_vessel_module, out_vessel_module, [v_1, p_1], [v_2, p_2])
        # check if heart has a second input vessel
        # if it doesn't add a zero flow mapping to that input
        if not vessel_array[np.where(vessel_array["name"] == 'heart')][0]["inp_vessel_2"]:
            if vessel_array[np.where(vessel_array["name"] == 'heart')][0]["inp_vessel_1"] == 'venous_ivc':
                self.__write_mapping(wf, 'zero_flow_module', 'heart', ['v_zero'], ['v_svc'])
            elif vessel_array[np.where(vessel_array["name"] == 'heart')][0]["inp_vessel_1"] == 'venous_svc':
                self.__write_mapping(wf, 'zero_flow_module', 'heart', ['v_zero'], ['v_ivc'])
            else:
                print('for now, the final venous compartment must be named venous_ivc or venous_svc')
                exit()


    def __write_terminal_venous_connection_comp(self, wf, vessel_array):
        first_venous_names = [] # stores name of venous compartments that take flow from terminals
        for vessel_vec in vessel_array:
            # first map variables between connection and the venous sections
            if vessel_vec["vessel_type"] == 'terminal':
                vessel_name = vessel_vec["name"]
                out_vessel_name = vessel_vec["out_vessel_1"]
                v_1 = 'v_T'
                v_2 = f'v_{vessel_name}'
    
                self.__write_mapping(wf, vessel_name+'_module','terminal_venous_connection',
                              [v_1], [v_2])
    
            if vessel_vec["vessel_type"] == 'venous' and not vessel_vec['inp_vessel_1']:
                vessel_name = vessel_vec["name"]
                first_venous_names.append(vessel_name)
                vessel_BC_type = vessel_vec["BC_type"]
                v_1 = f'v_{vessel_name}'
                if vessel_BC_type == 'vp':
                    v_2 = 'v_in'
                else:
                    print(f'first venous vessel BC type of {vessel_BC_type} has not'
                          f'been implemented')
    
                self.__write_mapping(wf, 'terminal_venous_connection', vessel_name+'_module',
                              [v_1], [v_2])
    
        # loop through vessels to get the terminals to add up for each first venous section
        terminal_names_for_first_venous = [[] for i in range(len(first_venous_names))]
        for vessel_vec in vessel_array:
            if vessel_vec['vessel_type'] == 'terminal':
                vessel_name = vessel_vec["name"]
                for idx, venous_name in enumerate(first_venous_names):
                    if vessel_vec['out_vessel_1'] == venous_name:
                        terminal_names_for_first_venous[idx].append(vessel_name)
    
        # create computation environment for connection and write the
        # variable definition and calculation of flow to each first venous module.
        wf.write(f'<component name="terminal_venous_connection">\n')
        variables = []
        units = []
        in_outs = []
        for idx, venous_name in enumerate(first_venous_names):
            for terminal_name in terminal_names_for_first_venous[idx]:
                variables.append(f'v_{terminal_name}')
                units.append('m3_per_s')
                in_outs.append('in')
    
            variables.append(f'v_{venous_name}')
            units.append('m3_per_s')
            in_outs.append('out')
    
        self.__write_variable_declarations(wf, variables, units, in_outs)
        for idx, venous_name in enumerate(first_venous_names):
            rhs_variables = []
            lhs_variable = f'v_{venous_name}'
            for terminal_name in terminal_names_for_first_venous[idx]:
                rhs_variables.append(f'v_{terminal_name}')
    
            self.__write_variable_sum(wf, lhs_variable, rhs_variables)
        wf.write('</component>\n')

    def __write_access_variables(self, wf, vessel_array):
        for vessel_vec in vessel_array:
            if vessel_vec["vessel_type"] in ['heart', 'pulmonary']:
                continue
            wf.writelines([f'<component name="{vessel_vec["name"]}">\n',
            '   <variable name="u" public_interface="in" units="J_per_m3"/>\n',
            '   <variable name="v" public_interface="in" units="m3_per_s"/>\n'])
            if vessel_vec['vessel_type']=='terminal':
                wf.write('   <variable name="R_T" public_interface="in" units="Js_per_m6"/>\n')
                wf.write('   <variable name="C_T" public_interface="in" units="m6_per_J"/>\n')
            if vessel_vec['vessel_type']=='venous':
                wf.write('   <variable name="R" public_interface="in" units="Js_per_m6"/>\n')
                wf.write('   <variable name="C" public_interface="in" units="m6_per_J"/>\n')
            wf.write('</component>\n')

    def __write_comp_to_module_mappings(self, wf, vessel_array):
        for vessel_vec in vessel_array:
            # input and output vessels
            vessel_name = vessel_vec["name"]
            if vessel_name in ['heart', 'pulmonary']:
                # TODO make the heart and pulmonary sections modules instead
                # of prewritten comp environments in the base cellml code.
                continue
            if vessel_vec["vessel_type"] == 'terminal':
                inp_vars = ['u', 'v', 'R_T', 'C_T']
                out_vars = ['u', 'v_T', 'R_T', 'C_T']
            elif vessel_vec["vessel_type"] == 'venous':
                inp_vars = ['u', 'v', 'C', 'R']
                out_vars = ['u', 'v', 'C', 'R']
            else:
                inp_vars = ['u', 'v']
                out_vars = ['u', 'v']
            self.__write_mapping(wf, vessel_name, vessel_name+'_module', inp_vars, out_vars)

    def __write_param_mappings(self, wf, vessel_array, params_array=None):
        for vessel_vec in vessel_array:
            # input and output vessels
            vessel_name = vessel_vec["name"]
            if vessel_name in ['heart', 'pulmonary']:
                continue
    
            if vessel_vec["vessel_type"] == 'terminal':
                vessel_name_minus_T = re.sub('_T$', '', vessel_name)
                systemic_vars = [f'R_T_{vessel_name_minus_T}',
                                 f'C_T_{vessel_name_minus_T}',
                                 f'alpha_{vessel_name_minus_T}',
                                 f'v_nom_{vessel_name_minus_T}',
                                 'gain_int']
                module_vars = ['R_T',
                               'C_T',
                               'alpha',
                               'v_nominal',
                               'gain_int']
    
            elif vessel_vec["vessel_type"]=='venous':
                systemic_vars = [f'R_{vessel_name}',
                                 f'C_{vessel_name}',
                                 f'I_{vessel_name}']
                module_vars = ['R',
                               'C',
                               'I']
            elif vessel_vec["vessel_type"] == 'arterial_simple':
                systemic_vars = [f'R_{vessel_name}',
                                 f'C_{vessel_name}',
                                 f'I_{vessel_name}']
                module_vars = ['R',
                               'C',
                               'I']
            else:
                systemic_vars = [f'l_{vessel_name}',
                                 f'E_{vessel_name}',
                                 f'r_{vessel_name}',
                                 f'theta_{vessel_name}',
                                 'beta_g']
                module_vars = ['l',
                               'E',
                               'r',
                               'theta',
                               'beta_g']
    
            # check that the variables are in the parameter array
            if params_array is not None:
                for variable_name in systemic_vars:
                    if variable_name not in params_array["variable_name"]:
                        print(f'variable {variable_name} is not in the parameter '
                              f'dataframe/csv file')
                        exit()
            module_addon = '_module'
            self.__write_mapping(wf, 'parameters_systemic', vessel_name+module_addon,
                          systemic_vars, module_vars)
    
    def __write_time_mappings(self, wf, vessel_array):
        for vessel_vec in vessel_array:
            # input and output vessels
            vessel_name = vessel_vec["name"]
            if vessel_name in ['heart', 'pulmonary']:
                continue
    
            module_addon = '_module'
            self.__write_mapping(wf, 'environment', vessel_name+module_addon,
                          ['time'], ['t'])

    def __write_import(self, wf, vessel_vec):
        if vessel_vec['vessel_type'] == 'terminal':
            module_type = f'{vessel_vec["BC_type"]}_T_type'
        elif vessel_vec['vessel_type'] == 'split_junction':
            module_type = f'{vessel_vec["BC_type"]}_split_type'
        elif vessel_vec['vessel_type'] == 'merge_junction':
            module_type = f'{vessel_vec["BC_type"]}_merge_type'
        elif vessel_vec['vessel_type'] == '2in2out_junction':
            module_type = f'{vessel_vec["BC_type"]}_2in2out_type'
        elif vessel_vec['vessel_type'] == 'venous':
            module_type = f'{vessel_vec["BC_type"]}_simple_type'
        elif vessel_vec['vessel_type'] == 'arterial_simple':
            module_type = f'{vessel_vec["BC_type"]}_simple_type'
        else:
            module_type = f'{vessel_vec["BC_type"]}_type'
    
        if vessel_vec["name"] == 'heart':
            str_addon = ''
        else:
            str_addon = '_module'
    
        wf.writelines([f'<import xlink:href="{self.filename_prefix}_modules.cellml">\n',
        f'    <component component_ref="{module_type}" name="{vessel_vec["name"]+str_addon}"/>\n',
        '</import>\n'])

    def __check_input_output_vessels(self, vessel_array, main_vessel, out_vessel,
                                   main_vessel_BC_type, out_vessel_BC_type,
                                   main_vessel_type, out_vessel_type):
        if not out_vessel:
            print(f'connection vessels incorrectly defined for {main_vessel}')
            exit()
        if main_vessel_type == 'terminal':
            pass
        elif main_vessel not in vessel_array[np.where(vessel_array["name"]==out_vessel
                                                    )][["inp_vessel_1", "inp_vessel_2"]][0]:
            print(f'"{main_vessel}" and "{out_vessel}" are incorrectly connected, '
                  f'check that they have eachother as output/input')
            exit()
    
        if main_vessel_BC_type.endswith('v'):
            if not out_vessel_BC_type.startswith('p'):
                print(f'"{main_vessel}" output BC is v, the input BC of "{out_vessel}"',
                      ' should be p')
                exit()
        if main_vessel_BC_type.endswith('p'):
            if not out_vessel_BC_type.startswith('v'):
                print(f'"{main_vessel}" output BC is p, the input BC of "{out_vessel}"',
                      ' should be v')
                exit()

    def __write_mapping(self, wf, inp_name, out_name, inp_vars_list, out_vars_list):
        wf.writelines(['<connection>\n',
        f'   <map_components component_1="{inp_name}" component_2="{out_name}"/>\n'])
        for inp_var, out_var in zip(inp_vars_list, out_vars_list):
            if inp_var and out_var:
                wf.write(f'   <map_variables variable_1="{inp_var}" variable_2="{out_var}"/>\n')
        wf.write('</connection>\n')

    def __write_variable_declarations(self, wf, variables, units, in_outs):
        for variable, unit, in_out in zip(variables, units, in_outs):
            wf.write(f'<variable name="{variable}" public_interface="{in_out}" units="{unit}"/>\n')
    
    def __write_constant_declarations(self, wf, variable_names, units, values):
        for variable, unit, value in zip(variable_names, units, values):
            wf.write(f'<variable initial_value="{value}" name="{variable}" '
                     f'public_interface="out" units="{unit}"/>\n')

    def __write_variable_sum(self, wf, lhs_variable, rhs_variables):
        wf.writelines('<math xmlns="http://www.w3.org/1998/Math/MathML">\n'
                      '   <apply>\n'
                      '       <eq/>\n'
                      f'       <ci>{lhs_variable}</ci>\n')
        if len(rhs_variables) > 1:
            wf.write('       <apply>\n')
            wf.write('           <plus/>\n')
            for variable in rhs_variables:
                wf.write(f'            <ci>{variable}</ci>\n')
            wf.write('       </apply>\n')
        else:
            wf.write(f'            <ci>{rhs_variables[0]}</ci>\n')
    
        wf.write('   </apply>\n')
        wf.write('</math>\n')
