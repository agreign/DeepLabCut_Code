#import packages that are needed
#import csv files
####eventually add user input on top of csv files
#import rat position data
#figure out amount of points that the rat is in the cage
#from there figure out the amount of time that the rat is in each section
#return the amount of time that rat is in each section

import csv
import pandas as pd
import numpy as np
import math
from styleframe import StyleFrame, Styler, utils

def open_files(f, eval):
    dic_final = {}
    with open(f) as csv_file:
        for dic in csv.reader(csv_file):
            if not dic:
                continue
            else:
                dic_final[dic[0]] = dic[1]
    return dic_final

def create_complicated_dic(f):
    df_x = pd.read_csv(f)
    name = 0
    new_dic = {}
    for column_name, column_data in df_x.iteritems():
        new_dic_part_1 = {}
        if name == 0:
            line_data = column_data.tolist()
            name += 1
            continue
        for j in range(0, len(line_data)):
            if isinstance(column_data.iloc[j], float) == False:
                t = column_data.iloc[j].replace('[', '')
                t = t.replace(']', '')
                ls = t.split(' ')
                copy_ls = []
                for i in ls:
                    if len(i) == 0:
                        continue
                    else:
                        copy_ls.append(float(i))
                copy_ls = np.array(copy_ls)
                new_dic_part_1[line_data[j]] = copy_ls
        new_dic[column_name] = new_dic_part_1
    return new_dic

def colmn_to_list(df, line_number, indx):
    position = df.iloc[2:line_number[-1] + 3, indx]
    position = position.tolist()
    return position

def grab_center(df, part):
    df.columns = df.iloc[0]  # change it so that the columns are equal to the bodyparts
    line_number = list(range(len(df.index) - 2))  # list of all of the line numbers
    column_center = df.pop(part)  # grab the center of the rat's body from the dataframe
    x_position = colmn_to_list(column_center, line_number, 0)
    y_position = colmn_to_list(column_center, line_number, 1)
    likelihood = colmn_to_list(column_center, line_number, 2)
    return x_position, y_position, likelihood, line_number

def x_y_high_likelihood(x_position, y_position, likelihood):
    x_position_lik_update = []
    y_position_lik_update = []
    for i in range(0, len(likelihood)):
        if float(likelihood[i]) >=0.9:
            x_position_lik_update.append(float(x_position[i]))
            y_position_lik_update.append(float(y_position[i]))
        else:
            x_position_lik_update.append(np.nan)
            y_position_lik_update.append(np.nan)
    return x_position_lik_update, y_position_lik_update

def edit_x_y_likelihood(position_lik):
    print(len(position_lik))
    element_start_and_end = []
    start_elem = 0 #True or False value in terms of a binary value
    if math.isnan(position_lik[0]):
        start_elem = 1
    for i in range(1, len(position_lik)-1):
        if math.isnan(position_lik[i]):
            if math.isnan(position_lik[i-1]) == False:
                element_start_and_end.append(i-1)
                start_elem = 1
            if math.isnan(position_lik[i+1]) == False:
                element_start_and_end.append(i+1)
                if (start_elem == 1) and (len(element_start_and_end) == 1):
                    position_lik[0:(i+1)] = [position_lik[element_start_and_end[0]]]*(i+1)
                else:
                    position_lik[element_start_and_end[0]+1:i+1] = np.linspace(position_lik[element_start_and_end[0]], position_lik[i+1], (i+1)-(element_start_and_end[0]+1), endpoint=False).tolist()
                start_elem = 0
                element_start_and_end = []
    if (start_elem == 1) and (len(element_start_and_end) == 1):
        position_lik[element_start_and_end[0]+1:] = [position_lik[element_start_and_end[0]]]*abs(element_start_and_end[0]+1 - len(position_lik))
    return position_lik



def zones(dic):
    zone = []
    for keys, values in dic.items():
        zone.append(keys)
    return zone

def determine_boundaries(dic_right_click, dic, zone):
    dic_lower_and_upper = {}
    for i in zone:
        temp_dic = {}
        lower_bound = []
        upper_bound = []
        for val, key in dic[i].items():
            if True in set(key < np.array(float(dic_right_click[i]))):
                lower_bound.append(val)
            else:
                upper_bound.append(val)
        temp_dic['upper'] = upper_bound
        temp_dic['lower'] = lower_bound
        dic_lower_and_upper[i] = temp_dic
    return dic_lower_and_upper

def determine_if_in_zone(x_position, y_position, lower_and_upper_bound_dic_x, lower_and_upper_bound_dic_y, dic_x, dic_y, line_number):
    list_of_j = []
    dic_in_zone = {}
    dic_time_in_zone = {}
    print(len(x_position))
    bad_in_first = []
    for j in range(0, len(x_position)):
        for key, val in lower_and_upper_bound_dic_y.items():
            is_in_this_zone = True
            for key1, val1 in lower_and_upper_bound_dic_y[key].items():
                if key1 == 'upper':
                    for i in val1:
                        if False in set(y_position[j] < dic_y[key][i]):
                            print('')
                            #is_in_this_zone = False
                    for i in lower_and_upper_bound_dic_x[key][key1]:
                        if False in set(x_position[j] < dic_x[key][i]):
                            is_in_this_zone = False
                elif key1 == 'lower':
                    for i in val1:
                        if False in set(y_position[j] >= dic_y[key][i]):
                            print('')
                            #is_in_this_zone = False
                    for i in lower_and_upper_bound_dic_x[key][key1]:
                        if False in set(x_position[j] >= dic_x[key][i]):
                            is_in_this_zone = False
                else:
                    is_in_this_zone = False
                    #is_in_this_zone = True
            if is_in_this_zone == True:
                dic_in_zone[j] = key
                if key not in dic_time_in_zone.keys():
                    dic_time_in_zone[key] = 1
                else:
                    dic_time_in_zone[key] += 1

                #new stuff I added
                if len(bad_in_first) >= 1:
                    for k in bad_in_first:
                        dic_in_zone[k] = dic_in_zone[j]
                        dic_time_in_zone[dic_in_zone[j]] += 1
                    bad_in_first = []
                #end of new stuff
        if j not in dic_in_zone.keys():
            #new stuff I added
            if j == 0 or len(bad_in_first) >= 1:
                bad_in_first.append(j)
            if not bad_in_first:
                #end of new stuff
                dic_in_zone[j] = dic_in_zone[j-1]
                dic_time_in_zone[dic_in_zone[j-1]] += 1
                list_of_j.append(j)
    return dic_in_zone, dic_time_in_zone, list_of_j

def time_in_zone_throughout(dic_in_zone, time, line_number):
    time_frame = time/line_number
    dic_time_in_zone_throughout = {}
    dic_time_in_zone_position = {}
    for key, val in dic_in_zone.items():
        if val not in dic_time_in_zone_throughout:
            dic_time_in_zone_throughout[val] = [key*time_frame]
            dic_time_in_zone_position[val] = [key]
        else:
            dic_time_in_zone_throughout[val].append(key*time_frame)
            dic_time_in_zone_position[val].append(key)
    return dic_time_in_zone_throughout, dic_time_in_zone_position

def make_sure_list_in_ascending_order(dic_time_in_position):
    edited_dic_time_in_position = {}
    for key, val in dic_time_in_position.items():
        test_list = [int(i) for i in val]
        test_list.sort()
        edited_dic_time_in_position[key] = test_list
    return edited_dic_time_in_position

def dic_time_in_zone_with_zeros_and_time(edited_dic_time_in_position, num_lines):
    dic_time_in_zone_with_zeros = {}
    max_val = 0
    for key, val in edited_dic_time_in_position.items():
        if val[-1] > max_val:
            max_val = val[-1]
    for key, val in edited_dic_time_in_position.items():
        dic_time_in_zone_with_zeros[key] = []
        for i in range(0, max_val+1):
            if i in val:
                dic_time_in_zone_with_zeros[key].append(i)
            else:
                dic_time_in_zone_with_zeros[key].append(0)
    return dic_time_in_zone_with_zeros

def dic_official_time_in_zone(dic_time_in_zone_with_zeros, time_frame):
    dic_time_in_zone_start = {}
    dic_time_in_zone_end = {}
    real_time = time_frame
    for key, val in dic_time_in_zone_with_zeros.items():

        if val[0] != 0:
            dic_time_in_zone_start[key] = [real_time]
        else:
            dic_time_in_zone_start[key] = []
        dic_time_in_zone_end[key] = []
        real_time += time_frame
        for i in range(1, len(val)-1):
            if val[i-1] == 0 and val[i] != 0:
                dic_time_in_zone_start[key].append(real_time)
            if val[i+1] == 0 and val[i] != 0:
                dic_time_in_zone_end[key].append(real_time+time_frame)
            real_time += time_frame
        if val[-1] != 0:
            dic_time_in_zone_end[key].append(real_time-time_frame)
        real_time = time_frame
    return dic_time_in_zone_start, dic_time_in_zone_end

def create_df_for_times(dic_time_in_zone_start, dic_time_in_zone_end):
    list_in_df = []
    for key, val in dic_time_in_zone_start.items():
        start_list = []
        start_list.append(key + ' Start')
        for i in val:
            start_list.append(i)
        list_in_df.append(start_list)
    for key, val in dic_time_in_zone_end.items():
        end_list = []
        end_list.append(key + ' End')
        for i in val:
            end_list.append(i)
        list_in_df.append(end_list)
    #df_behavior = pd.DataFrame(np.array(list_in_df))
    df_behavior_start = pd.DataFrame.from_dict(dic_time_in_zone_start, orient='index').T
    df_behavior_end = pd.DataFrame.from_dict(dic_time_in_zone_end, orient='index').T
    return df_behavior_start, df_behavior_end

def change_coloring(df, path):
    sf = StyleFrame(df)
    sf.apply_column_style(cols_to_style=sf.columns, styler_obj=Styler(font_size=11, font=utils.fonts.calibri))
    sf.apply_headers_style(styler_obj=Styler(bold=True, font_size=12, font=utils.fonts.calibri))
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['Pre Hidden'] > 6], cols_to_style=['Pre Hidden'], styler_obj=Styler(bg_color=utils.colors.blue, font_size=11, font_color=utils.colors.white))
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['Pre Center'] > 6], cols_to_style=['Pre Center'],
                              styler_obj=Styler(bg_color=utils.colors.blue, font_size=11,
                                                font_color=utils.colors.white))
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['Pre Approach'] > 6], cols_to_style=['Pre Approach'],
                              styler_obj=Styler(bg_color=utils.colors.blue, font_size=11,
                                                font_color=utils.colors.white))

    sf.apply_style_by_indexes(indexes_to_style=sf[sf['Pre Hidden'] > 6], cols_to_style=['Hidden Start Time'],
                              styler_obj=Styler(bg_color=utils.colors.yellow, font_size=11))
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['Pre Center'] > 6], cols_to_style=['Center Start Time'],
                              styler_obj=Styler(bg_color=utils.colors.yellow, font_size=11))
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['Pre Approach'] > 6], cols_to_style=['Approach Start Time'],
                              styler_obj=Styler(bg_color=utils.colors.yellow, font_size=11))
    ew = StyleFrame.ExcelWriter(path)
    sf.to_excel(ew)
    ew.save()

def dataframe_to_csv(is_true, df_position_behavior, name_of_file_path):
    if is_true == True:
        df_position_behavior.to_csv(name_of_file_path)


list_of_vids = ['Test 12 - 8T2 - 28 DAYS']
for v in list_of_vids:
    dic_x = create_complicated_dic('D:\\Documents\\deeplabcut_trial_2\\Dictionaries_of_positions\\' + v + '\\df_dict_test_x.csv')
    dic_y = create_complicated_dic('D:\\Documents\\deeplabcut_trial_2\\Dictionaries_of_positions\\' + v + '\\df_dict_test_y.csv')
    dic_right_click_x = open_files('D:\\Documents\\deeplabcut_trial_2\\Dictionaries_of_positions\\' + v + '\\dict_test_right_click_x.csv', False)
    dic_right_click_y = open_files('D:\\Documents\\deeplabcut_trial_2\\Dictionaries_of_positions\\' + v + '\\dict_test_right_click_y.csv', False)

    #read table of rat positions
    df = pd.read_csv("D:\\Videos\\Lab_Videos\\Fab_Videos\\New_Videos\\" + v + "DLC_resnet50_Combined_VideosNov14shuffle1_1030000.csv", low_memory=False)

    [x_position, y_position, likelihood, line_number] = grab_center(df, 'centerhead')

    [x_position_update, y_position_update] = x_y_high_likelihood(x_position, y_position, likelihood)
    x_position_without_nan = edit_x_y_likelihood(x_position_update)
    y_position_without_nan = edit_x_y_likelihood(y_position_update)
    #print(dic_x)
    #print(x_position_without_nan)
    zone = zones(dic_right_click_x)

    lower_and_upper_bound_dic_x = determine_boundaries(dic_right_click_x, dic_x, zone)

    lower_and_upper_bound_dic_y = determine_boundaries(dic_right_click_y, dic_y, zone)

    [dic_in_zone, dic_time_in_zone, list_of_j] = determine_if_in_zone(x_position_without_nan, y_position_without_nan, lower_and_upper_bound_dic_x, lower_and_upper_bound_dic_y, dic_x, dic_y, line_number)
    #print(dic_in_zone)
    count = 0
    t = 60
    type = 'min'
    #t = input('What is the total amount of time for the video?: ')
    #type = input('seconds or minutes: (sec, min) ')
    if type == 'min':
        time = 60*float(t)
    else:
        time = t
    time_dic = {}
    for i in dic_time_in_zone.items():
        time_dic[i[0]] = i[1]/len(line_number) * time
    dic_time_in_zone_throughout, dic_time_in_position = time_in_zone_throughout(dic_in_zone, time, line_number[-1])
    sorted_dic_time_in_position = make_sure_list_in_ascending_order(dic_time_in_position)
    dic_time_in_zone_with_zeros = dic_time_in_zone_with_zeros_and_time(sorted_dic_time_in_position, line_number[-1])
    dic_time_in_zone_start, dic_time_in_zone_end = dic_official_time_in_zone(dic_time_in_zone_with_zeros, time/line_number[-1])
    df_behavior_start, df_behavior_end = create_df_for_times(dic_time_in_zone_start, dic_time_in_zone_end)

    print(v)
    name_of_file_path = "D:\\Documents\\deeplabcut_trial_2\\csv_behavioral_data\\position_data_" + v + ".xlsx"
    df_behavior_start = df_behavior_start.rename(columns={'center': 'Center Start Time', 'approach':'Approach Start Time', 'hidden':'Hidden Start Time'})
    df_behavior_end = df_behavior_end.rename(columns={'center': 'Center End Time', 'approach':'Approach End Time', 'hidden':'Hidden End Time'})
    #print(df_behavior_end)
    #print(df_behavior_start)
    df_behavior_position = pd.concat([df_behavior_start, df_behavior_end], axis=1)
    #print(df_behavior_position)
    df_behavior_position['Pre Hidden'] = 0
    df_behavior_position = df_behavior_position.reset_index(drop=True)
    edit_col = abs(df_behavior_position['Hidden Start Time'].iloc[1:].astype(float).reset_index(drop=True)-df_behavior_position['Hidden End Time'].iloc[:-1].astype(float).reset_index(drop=True))
    edit_col.index +=1
    df_behavior_position.loc[1:, 'Pre Hidden'] = edit_col
    df_behavior_position['Pre Approach'] = 0
    edit_col1 = abs(df_behavior_position['Approach Start Time'].iloc[1:].astype(float).reset_index(drop=True)-df_behavior_position['Approach End Time'].iloc[:-1].astype(float).reset_index(drop=True))
    edit_col1.index +=1
    df_behavior_position.loc[1:, 'Pre Approach'] = edit_col1
    df_behavior_position['Pre Center'] = 0
    edit_col2 = abs(df_behavior_position['Center Start Time'].iloc[1:].astype(float).reset_index(drop=True)-df_behavior_position['Center End Time'].iloc[:-1].astype(float).reset_index(drop=True))
    edit_col2.index +=1
    df_behavior_position.loc[1:, 'Pre Center'] = edit_col2
    df_behavior_position = df_behavior_position[['Hidden Start Time', 'Hidden End Time', 'Pre Hidden', 'Center Start Time', 'Center End Time', 'Pre Center', 'Approach Start Time', 'Approach End Time', 'Pre Approach']]
    #print(df_behavior_position.head())
    change_coloring(df_behavior_position, name_of_file_path)
    dataframe_to_csv(False, df_behavior_end, name_of_file_path)
    #print(time_dic)
    #print(dic_time_in_zone_start)
    #print(dic_time_in_zone_end)
    #print("dic_time_in_zone_throughout")
    #print(dic_time_in_zone_throughout)

    #####look for y and see what happens

