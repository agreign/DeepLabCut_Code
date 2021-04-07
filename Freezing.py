import pandas as pd
import numpy as np
from styleframe import StyleFrame, Styler, utils

# helper functions
def column_to_list(df, line_number_list, index):
    """
    :param df: dataframe that is just a column of numbers (specifically looing at a column of data in csv)
    :param line_number_list: the line number in list format
    :param index: the index which determine if you want x, y, and likelihood positions.
    :return: the position numbers with as a list
    """
    position = df.iloc[2:line_number_list[-1]+3, index]
    position = position.tolist()
    return position
# functions that are used separately

def time_to_seconds(num_time, type_time):
    """
    :param num_time: the actual number for the time that the user inputed
    :param type_time: the units of time
    :return: the amount of time with respect to seconds
    """
    if type_time == 'sec':
        time_in_sec = num_time
    elif type_time == 'min':
        time_in_sec = num_time*60
    elif type_time == 'hr':
        time_in_sec =  num_time*360
    return time_in_sec

def grab_x_y_likelihood(df_of_video_information, body_part):
    """
    :param df_of_video_information: the csv file that contains all of the data for the rat posiitons
    :param body_part: which body part the user would like to examine
    :return: the x and y position, likelihood of that body part being correctly identified, and the number of frames
    that were recorded in csv
    """
    df_of_video_information.columns = df_of_video_information.iloc[0] #column labels change to the body part names
    line_number_len = len(df_of_video_information.index)-2 # list of all of the line numbers
    line_number_list = list(range(line_number_len)) #list of all the line numbers
    columns_body_part = df_of_video_information.pop(body_part) #columns that are associated with the user wanted body part
    # find the 3 different returns
    x_position = column_to_list(columns_body_part, line_number_list, 0)
    y_position = column_to_list(columns_body_part, line_number_list, 1)
    likelihood = column_to_list(columns_body_part, line_number_list, 2)
    return x_position, y_position, likelihood, line_number_len

def bad_likelihood_times(likelihood, percent_likelihood_threshold):
    """
    :param likelihood: list of the likelihoods
    :param percent_likelihood_threshold: the percent likelihood that the user determined is too low if below that number
    :return: the line numbers that contain likelihoods less than a percent
    """
    line_with_bad_likelihood = [] #line numbers that contain the likelihood values under percent_likelihood_threshold
    for i in range(0, len(likelihood)):
        if float(likelihood[i]) < percent_likelihood_threshold:
            line_with_bad_likelihood.append(i)
    return line_with_bad_likelihood

def change_x_y_based_on_likelihood(line_with_bad_likelihood, x_position, y_position):
    """
    :param line_with_bad_likelihood: the lines in the data that do not have a good likelihood
    :param x_position: x position of the body part as a list
    :param y_position: y position of the body part as a list
    :return: x_position and y_position of the position of the body part as a list that gets rid of bad likelihood
    """
    num_likelihood_wrong_check = []
    for i in line_with_bad_likelihood:
        num_likelihood_wrong_check.append(i)
        # if there are two lines that are in a row
        if i+1 in line_with_bad_likelihood:
            continue
        else:
            #change x and y position to the average of the two good likelihoods around the bad ones
            if num_likelihood_wrong_check[-1] == line_number_len - 1:
                for j in num_likelihood_wrong_check:
                    x_position[j] = str((float(x_position[num_likelihood_wrong_check[0] - 1])))
                    y_position[j] = str((float(y_position[num_likelihood_wrong_check[0] - 1])))
            else:
                for j in num_likelihood_wrong_check:
                    x_position[j] = str((float(x_position[num_likelihood_wrong_check[0] - 1]) +
                                         float(x_position[num_likelihood_wrong_check[-1] + 1])) / 2)
                    y_position[j] = str((float(y_position[num_likelihood_wrong_check[0] - 1]) +
                                         float(y_position[num_likelihood_wrong_check[-1] + 1])) / 2)
            num_likelihood_wrong_check = []
    return x_position, y_position

def create_closeness_vector(x_position, y_position, pixel_close):
    """
    :param x_position: x position of the body part as a list which takes into consideration the bad likelihoods
    :param y_position: y position of the body part as a list which takes into consideration the bad likelihoods
    :param pixel_close: number of pixels that determine if two positions are close together
    :return: determine if two values in position lists are close (1 if close and 0 if not)
    """
    are_close = []
    for i in range(0, len(x_position)-1):
        if abs(round(float(x_position[i]), 2)-round(float(x_position[i+1]), 2)) <= pixel_close:
            if abs(round(float(y_position[i]), 2)-round(float(y_position[i+1]), 2)) <= pixel_close:
                are_close.append(1)
            else:
                are_close.append(0)
        else:
            are_close.append(0)
    return are_close

def edit_closeness_vec_for_zeros_in_middle_of_all_1s(closeness_vec):
    """
    :param closeness_vec: vector that says closeness of positions
    :return: a new closeness vector with getting rid of random zeros in the middle of ones
    """
    new_closeness_vec = closeness_vec
    for i in range(11, len(closeness_vec)):
        if sum(closeness_vec[i-11:i-1]) == 10 and sum(closeness_vec[i+1:i+11]) == 10:
            new_closeness_vec[i] = 1
    return new_closeness_vec

def time_per_frame(time, line_number_len):
    """
    :param time: amount of time that the video is
    :param line_number_len: number of frames
    :return: time per frame rate
    """
    time_frame = time/line_number_len
    return time_frame

def is_freezing(time_frame, time_considered_freezing):
    """
    :param time_frame: amount of time per frame (rate of the video)
    :param time_considered_freezing: amount of time in seconds that would be considered freezing
    :return: number of frames that are considered freezing
    """
    freezing_frame_threshold = time_considered_freezing/time_frame
    return freezing_frame_threshold

def determine_frames_freezing(freezing_frame_threshold, closeness_vec, time_frame):
    """
    :param freezing_frame_threshold: the number of frames that are considered to be freezing
    :param closeness_vec: the vector that determines closeness of positions
    :param time_frame: the amount of time per frame (rate of the video)
    :return: the number of frames that were freezing a list of time stamps where the rat is freezing
    """
    is_freezing_count = 0
    freezing_frame = 0
    actual_timeline = 0
    freezing_frame_vec = []
    for i in range(len(closeness_vec)):
        if closeness_vec[i] == 0:
            is_freezing_count = 0
            actual_timeline += time_frame
            freezing_frame_vec.append(0)
        elif (closeness_vec[i] == 1) and (is_freezing_count < freezing_frame_threshold):
            is_freezing_count += 1
            actual_timeline += time_frame
            freezing_frame_vec.append(0)
        else:
            freezing_frame += 1
            actual_timeline += time_frame
            freezing_frame_vec.append(actual_timeline)
    return freezing_frame, freezing_frame_vec

def amount_of_time_freezing(freezing_frame, time_frame):
    """
    :param freezing_frame: number of frames freezing
    :param time_frame: time per frame of video (rate of the video)
    :return: total amount of time spent freezing
    """
    return freezing_frame*time_frame

def create_dataframe_for_behavior_start_and_end(freezing_vec, time_frame):
    start_vec = ['Freezing Start Time'] #start time of the behavior
    end_vec = ['Freezing End Time'] #end time of the behavior
    for i in range(1, len(freezing_vec)-1):
        if freezing_vec[i] != 0 and freezing_vec[i-1] == 0:
            start_vec.append(freezing_vec[i])
        if freezing_vec[i] != 0 and freezing_vec[i+1] == 0:
            end_vec.append(freezing_vec[i] + time_frame)
    if freezing_vec[-1] != 0 and len(start_vec)!= len(end_vec):
        end_vec.append(freezing_vec[-1] + time_frame)
    print(len(start_vec))
    print(type(start_vec[1]))
    print(len(end_vec))
    print(np.asarray([start_vec, end_vec]))
    df_behavior = pd.DataFrame(np.asarray([start_vec, end_vec]))
    return df_behavior

def change_coloring(df, path):
    sf = StyleFrame(df)
    sf.apply_column_style(cols_to_style=sf.columns, styler_obj=Styler(font_size=11, font=utils.fonts.calibri))
    sf.apply_headers_style(styler_obj=Styler(bold=True, font_size=12, font=utils.fonts.calibri))
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['Pre Freezing'] > 6], cols_to_style=['Pre Freezing'], styler_obj=Styler(bg_color=utils.colors.blue, font_size=11, font_color=utils.colors.white))
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['Duration'] >= 1], cols_to_style=['Duration'],
                              styler_obj=Styler(bg_color=utils.colors.yellow, font_size=11))
    ew = StyleFrame.ExcelWriter(path)
    sf.to_excel(ew)
    ew.save()

def dataframe_to_csv(is_true, df_freezing_behavior, name_of_file_path):
    if is_true == True:
        df_freezing_behavior.to_csv(name_of_file_path)

list_of_vids = ['Test 23 - 4I4 - TEST']#['Test 22 - 8H4 - TEST']
#list_of_vids = ['Test 22 - 8H4 - TEST']
for i in list_of_vids:
    #change to make this the file path of the video that you would like to analyze. This will be an input from the C# program
    path_to_Video_data = "D:\\Videos\\Lab_Videos\\Douglas_Videos\\H3_I3_Group"
    name_of_csv_file = i + "DLC_resnet50_Combined_VideosNov14shuffle1_1030000.csv" 
    df_of_video_information = pd.read_csv(path_to_Video_data + "\\" + name_of_csv_file, low_memory=False)
    freezing_behavior_file_df = "D:\\Documents\\deeplabcut_trial_2\\csv_behavioral_data\\freezing_data_" + i + ".csv"
    freezing_behavior_file_df_xlsx = "D:\\Documents\\deeplabcut_trial_2\\csv_behavioral_data\\Douglas_Data\\freezing_data_" + i + ".xlsx"

    #This time stuff will be inputs from the C# program
    num_time = 60
    type_time = 'min'
    body_part = 'spine2'
    percent_likelihood_threshold = 0.9
    pixel_close = 1.05
    time_considered_freezing = 0.5

    time = time_to_seconds(num_time, type_time)
    x_position, y_position, likelihood, line_number_len = grab_x_y_likelihood(df_of_video_information, body_part)
    line_with_bad_likelihood = bad_likelihood_times(likelihood, percent_likelihood_threshold)
    x_position_likelihood, y_position_likelihood = change_x_y_based_on_likelihood(line_with_bad_likelihood, x_position, y_position)
    closeness_vec = create_closeness_vector(x_position_likelihood, y_position_likelihood, pixel_close)
    edited_closeness_vec = edit_closeness_vec_for_zeros_in_middle_of_all_1s(closeness_vec)
    time_frame = time_per_frame(time, line_number_len)
    freezing_frame_threshold = is_freezing(time_frame, time_considered_freezing)
    freezing_frame, freezing_vec = determine_frames_freezing(freezing_frame_threshold, closeness_vec, time_frame)
    total_time_freezing = amount_of_time_freezing(freezing_frame, time_frame)
    print(total_time_freezing)
    #print(total_time_freezing)
    #print(freezing_vec)
    #print(time_frame)
    df_freezing_behavior = create_dataframe_for_behavior_start_and_end(freezing_vec, time_frame)
    print(df_freezing_behavior)
    df_freezing_behavior = df_freezing_behavior.transpose()
    #print(df_freezing_behavior.iloc[0].iloc[0])

    df_freezing_behavior.columns = df_freezing_behavior.iloc[0]
    print(df_freezing_behavior.columns)
    df_freezing_behavior = df_freezing_behavior.drop(df_freezing_behavior.index[0])
    df_freezing_behavior['Duration'] = abs(df_freezing_behavior['Freezing Start Time'].astype(float) - df_freezing_behavior['Freezing End Time'].astype(float))
    df_freezing_behavior['Pre Freezing'] = 0
    df_freezing_behavior = df_freezing_behavior.reset_index(drop=True)
    edit_col = abs(df_freezing_behavior['Freezing Start Time'].iloc[1:].astype(float).reset_index(drop=True)-df_freezing_behavior['Freezing End Time'].iloc[:-1].astype(float).reset_index(drop=True))
    edit_col.index +=1
    df_freezing_behavior.loc[1:, 'Pre Freezing'] = edit_col
    change_coloring(df_freezing_behavior, freezing_behavior_file_df_xlsx)
    dataframe_to_csv(False, df_freezing_behavior, freezing_behavior_file_df)
