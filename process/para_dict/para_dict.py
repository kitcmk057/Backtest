import itertools
import os




############# All para combination 係一個list of dictionary 裝哂所有既para combination #############
def get_all_para_combination(para_dict, df_dict, sec_profile, start_date, end_date,
                                run_mode, summary_mode, freq, file_format):


    base_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_folder = os.path.join(base_folder, 'data')
    secondary_data_folder = os.path.join(base_folder, 'secondary_data')
    backtest_output_folder = os.path.join(base_folder, 'backtest_output')
    signal_output_folder = os.path.join(base_folder, 'signal_output')

    if not os.path.isdir(data_folder): os.mkdir(data_folder)
    if not os.path.isdir(secondary_data_folder): os.mkdir(secondary_data_folder)
    if not os.path.isdir(backtest_output_folder): os.mkdir(backtest_output_folder)
    if not os.path.isdir(signal_output_folder): os.mkdir(signal_output_folder)


    py_filename = os.path.basename(__file__).replace('.py','')


    para_values = list(para_dict.values())
    para_keys = list(para_dict.keys())
    para_list = list(itertools.product(*para_values))

    print('number of combination:', len(para_list))

    intraday = True if freq != '1D' else False
    output_folder = backtest_output_folder if run_mode == 'backtest' else signal_output_folder

    all_para_combination = []

    for reference_index in range(len(para_list)):
        para = para_list[reference_index]
        code = para[0]
        df = df_dict[code]
        para_combination = {}
        for i in range(len(para)):
            key = para_keys[i]
            para_combination[key] = para[i]

        para_combination['para_dict'] = para_dict
        para_combination['sec_profile'] = sec_profile
        para_combination['start_date'] = start_date
        para_combination['end_date'] = end_date
        para_combination['reference_index'] = reference_index
        para_combination['freq'] = freq
        para_combination['file_format'] = file_format
        para_combination['df'] = df
        para_combination['intraday'] = intraday
        para_combination['output_folder'] = output_folder
        para_combination['data_folder'] = data_folder
        para_combination['run_mode'] = run_mode
        para_combination['summary_mode'] = summary_mode
        para_combination['py_filename'] = py_filename

        all_para_combination.append(para_combination)

    return all_para_combination