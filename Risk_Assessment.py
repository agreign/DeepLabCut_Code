import pandas as pd
import numpy as np
import cv2
import os
import random
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

#functions
def get_frames(full_path_to_video, name_of_video, path_to_frames):
    """
    :param full_path_to_video: the full path to the video
    :param name_of_video: the name of the video file
    :param path_to_frames: path that you would like to create for the frames of the video
    :return: Null. Creating the frames at path given if there are not already frames there
    """
    camera = cv2.VideoCapture(full_path_to_video)
    is_dir = False
    if not os.path.exists(path_to_frames + str(name_of_video)):
        os.makedirs(path_to_frames + str(name_of_video))
        is_dir = True
    current_frame = 0
    while(is_dir):
        ret, frame = camera.read()
        if ret and current_frame < 100: # 99 frames created
            #if video is still left continue creating images
            name = path_to_frames + str(name_of_video) + "\\" + str(current_frame) + ".jpg"
            cv2.imwrite(name, frame) # writing the extracted images
            current_frame += 1 # increasing counter so that it will show how many frames are created
        else:
            break
    camera.release()
    cv2.destroyAllWindows()

def choose_a_frame(name_of_video, path_to_frames):
    """
    :param name_of_video: the name of the video file
    :param path_to_frames: path that you would like to create for the frames of the video
    :return: the path to a frame in the list of frames created
    """
    full_frame_path = path_to_frames + str(name_of_video)
    path_to_a_frame = random.choice(os.listdir(full_frame_path))
    return path_to_a_frame

coords_x = []
coords_y = []
def click_event(event, x, y, flags, param):
    global coords
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(coords_x) % 2 == 0 and len(coords_x) > 0:
            pass
        elif len(coords_x) != 0:
            cv2.line(img, (coords_x[-1], coords_y[-1]), (x, y), (255, 0, 0), 5)
            cv2.imshow("Frame of Video", img)
        coords_x.append(x)
        coords_y.append(y)

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

def find_head_length(x_snout, y_snout, likelihood_snout, x_centerhead, y_centerhead, likelihood_centerhead):
    """
    :param x_snout:
    :param y_snout:
    :param likelihood_snout:
    :param x_centerhead:
    :param y_centerhead:
    :param likelihood_centerhead:
    :return:
    """
    length_to_average = []
    for i in range(0, len(x_snout)):
        if float(likelihood_centerhead[i]) >= 0.9 and float(likelihood_snout[i]) >= 0.9:
            length_to_average.append(np.sqrt((float(x_centerhead[i]) - float(x_snout[i]))**2 + (float(y_centerhead[i]) -
                                                                                                float(y_snout[i]))**2))
    avg_head_length = sum(length_to_average)/len(length_to_average)
    return avg_head_length+5

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

def change_x_y_based_on_likelihood(line_with_bad_likelihood, x_position, y_position, line_number_len):
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
            if num_likelihood_wrong_check[-1] == line_number_len-1:
                for j in num_likelihood_wrong_check:
                    x_position[j] = str((float(x_position[num_likelihood_wrong_check[0]-1])))
                    y_position[j] = str((float(y_position[num_likelihood_wrong_check[0] - 1])))
            else:
                for j in num_likelihood_wrong_check:
                    x_position[j] = str((float(x_position[num_likelihood_wrong_check[0]-1]) +
                                        float(x_position[num_likelihood_wrong_check[-1]+1]))/2)
                    y_position[j] = str((float(y_position[num_likelihood_wrong_check[0] - 1]) +
                                        float(y_position[num_likelihood_wrong_check[-1] + 1]))/2)
            num_likelihood_wrong_check = []
    return x_position, y_position

def determine_smallest_ear_distance(x_ear_right, y_ear_right, x_ear_left, y_ear_left, likelihood_ear_right, likelihood_ear_left):
    dist_list = []
    for i in range(0, len(x_ear_right)):
        if float(likelihood_ear_right[i]) >= 0.9 and float(likelihood_ear_left[i]) >= 0.9:
            dist_of_ear = np.sqrt(abs(float(x_ear_right[i]) - float(x_ear_left[i]))**2 + abs(float(y_ear_right[i]) -
                                                                                             float(y_ear_left[i]))**2)
            dist_list.append(dist_of_ear)
        else:
            dist_list.append(1000)
    first_quant = np.quantile(dist_list, .25)
    third_quant = np.quantile(dist_list, .97)
    return first_quant, third_quant, dist_list

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

def time_per_frame(time, line_number_len):
    """
    :param time: amount of time that the video is
    :param line_number_len: number of frames
    :return: time per frame rate
    """
    time_frame = time/line_number_len
    return time_frame

def is_risk_assessment(head_length, x_snout, y_snout, likelihood_snout, x_centerhead, x_spine3, where_door_x,
                       where_door_y, x_spine2, quant, dist_ear_list, approach_x, time_frame):
    is_in_hidden = False
    risk_assessment_vec = []
    risk_assessment_time_vec = []
    time_current = 0
    for i in range(0, len(x_snout)):
        time_current += time_frame
        if float(x_spine2[i]) > sum(where_door_x)/2 or float(x_spine3[i]) > sum(where_door_x)/2:
            is_in_hidden = True
        else:
            is_in_hidden = False
        if is_in_hidden == True:
            if float(x_snout[i]) < sum(where_door_x)/2 and float(likelihood_snout[i]) >= 0.9:
                if (float(y_snout[i]) > max(where_door_y) or float(y_snout[i]) < min(where_door_y)) and float(x_snout[i]) > sum(approach_x)/2:
                    risk_assessment_vec.append(0)
                    risk_assessment_time_vec.append(0)
                else:
                    risk_assessment_time_vec.append(time_current)
                    risk_assessment_vec.append(1)
            elif float(likelihood_snout[i]) < 0.9 and abs(sum(where_door_x)/2 - float(x_centerhead[i])) < head_length and dist_ear_list[i] < quant:
                risk_assessment_time_vec.append(time_current)
                risk_assessment_vec.append(1)
            else:
                risk_assessment_vec.append(0)
                risk_assessment_time_vec.append(0)
    return risk_assessment_vec, risk_assessment_time_vec

def time_risk_assessment(risk_assessment_vec, time_frame):
    sum_risk_assessment = sum(risk_assessment_vec)
    time_risk = time_frame*sum_risk_assessment
    return time_risk

def average_calc(lst):
    return sum(lst)/len(lst)

def create_dataframe_for_behavior_start_and_end(freezing_vec, time_frame):
    start_vec = ['Risk Assessment Start Time'] #start time of the behavior
    end_vec = ['Risk Assessment End Time'] #end time of the behavior
    for i in range(0, len(freezing_vec)-1):
        if freezing_vec[i] != 0 and freezing_vec[i-1] == 0:
            start_vec.append(freezing_vec[i])
        if freezing_vec[i] != 0 and freezing_vec[i+1] == 0:
            end_vec.append(freezing_vec[i] + time_frame)
    if len(start_vec) > len(end_vec):
        end_vec.append(freezing_vec[-1] + time_frame)
    print(start_vec)
    print(end_vec)
    df_behavior = pd.DataFrame(np.array([start_vec, end_vec]))
    return df_behavior

def change_coloring(df, path):
    sf = StyleFrame(df)
    sf.apply_column_style(cols_to_style=sf.columns, styler_obj=Styler(font_size=11, font=utils.fonts.calibri))
    sf.apply_headers_style(styler_obj=Styler(bold=True, font_size=12, font=utils.fonts.calibri))
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['Pre Risk Assessment'] > 6], cols_to_style=['Pre Risk Assessment'], styler_obj=Styler(bg_color=utils.colors.blue, font_size=11, font_color=utils.colors.white))
    sf.apply_style_by_indexes(indexes_to_style=sf[sf['Duration'] >= 1], cols_to_style=['Duration'],
                              styler_obj=Styler(bg_color=utils.colors.yellow, font_size=11))
    ew = StyleFrame.ExcelWriter(path)
    sf.to_excel(ew)
    ew.save()

def dataframe_to_csv(is_true, df_freezing_behavior, name_of_file_path):
    if is_true == True:
        df_freezing_behavior.to_csv(name_of_file_path)

list_of_vids = ['Test 54 - 9G3 - TEST']
for i in list_of_vids:
    #change to make this the file path of the video that you would like to analyze. This will be an input from the C# program
    path_to_Video_data = "D:\\Videos\\Lab_Videos\\Douglas_Videos\\F3_and_G3_Group" #change
    name_of_csv_file = i + "DLC_resnet50_Combined_VideosNov14shuffle1_1030000.csv" #change
    df_of_video_information = pd.read_csv(path_to_Video_data + "\\" + name_of_csv_file, low_memory=False)

    path_to_video = "D:\\Videos\\Lab_Videos\\Douglas_Videos\\F3_and_G3_Group\\" #change
    name_of_video = i + ".mp4" #change
    full_path_to_video = path_to_video + name_of_video

    path_to_frames = "C:\\Users\\Admin\\Desktop\\DeepLabCut\\Frames_Doug\\" #maybe change

    freezing_behavior_file_df = "D:\\Documents\\deeplabcut_trial_2\\csv_behavioral_data\\Douglas_Data\\risk_assessment_data_" + i + ".csv"
    freezing_behavior_file_df_xlsx = "D:\\Documents\\deeplabcut_trial_2\\csv_behavioral_data\\Douglas_Data\\risk_assessment_data_" + i + ".xlsx"

    #variables to input
    percent_likelihood_threshold = 0.9
    num_time = 60
    type_time = 'min'

    # functions
    get_frames(full_path_to_video, name_of_video, path_to_frames)
    picture_path = choose_a_frame(name_of_video, path_to_frames)
    img = cv2.imread(path_to_frames + str(name_of_video) + "\\" + str(picture_path), 0)
    cv2.imshow("Frame of Video", img)
    cv2.setMouseCallback("Frame of Video", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    where_door_x = coords_x
    where_door_y = coords_y

    coords_y = []
    coords_x = []
    img = cv2.imread(path_to_frames + str(name_of_video) + "\\" + str(picture_path), 0)
    cv2.imshow("Frame of Video", img)
    cv2.setMouseCallback("Frame of Video", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    approach_x = coords_x
    approach_y = coords_y

    x_snout, y_snout, likelihood_snout, line_number_len = grab_x_y_likelihood(df_of_video_information, 'snout')
    x_centerhead, y_centerhead, likelihood_centerhead, line_number_len = grab_x_y_likelihood(df_of_video_information,
                                                                                         'centerhead')
    x_spine3, y_spine3, likelihood_spine3, line_number_len = grab_x_y_likelihood(df_of_video_information, 'spine3')
    x_spine2, y_spine2, likelihood_spine2, line_number_len = grab_x_y_likelihood(df_of_video_information, 'spine2')
    x_rightear, y_rightear, likelihood_rightear, line_number_len = grab_x_y_likelihood(df_of_video_information, 'rightear')
    x_leftear, y_leftear, likelihood_leftear, line_number_len = grab_x_y_likelihood(df_of_video_information, 'leftear')

    head_length = find_head_length(x_snout, y_snout, likelihood_snout, x_centerhead, y_centerhead, likelihood_centerhead)

    bad_likelihood_spine2 = bad_likelihood_times(likelihood_spine2, percent_likelihood_threshold)
    x_position_likelihood_spine2, y_position_likelihood_spine2 = change_x_y_based_on_likelihood(bad_likelihood_spine2,
                                                                                            x_spine2, y_spine2, line_number_len)
    bad_likelihood_spine3 = bad_likelihood_times(likelihood_spine3, percent_likelihood_threshold)
    x_position_likelihood_spine3, y_position_likelihood_spine3 = change_x_y_based_on_likelihood(bad_likelihood_spine3,
                                                                                            x_spine3, y_spine3, line_number_len)

    first_quant, third_quant, dist_ear_list = determine_smallest_ear_distance(x_rightear, y_rightear, x_leftear, y_leftear,
                                                                      likelihood_rightear, likelihood_leftear)

    time = time_to_seconds(num_time, type_time)
    time_frame = time_per_frame(time, line_number_len)

    # one time calculation of risk assessment
    risk_assessment_vec, risk_assessment_time_vec = is_risk_assessment(head_length, x_snout, y_snout, likelihood_snout, x_centerhead, x_spine3, where_door_x,
                       where_door_y, x_spine2, third_quant, dist_ear_list, approach_x, time_frame)
    time_risk = time_risk_assessment(risk_assessment_vec, time_frame)

    df_behavior_risk_assessment = create_dataframe_for_behavior_start_and_end(risk_assessment_time_vec, time_frame)
    df_behavior_risk_assessment = df_behavior_risk_assessment.transpose()
    df_behavior_risk_assessment.columns = df_behavior_risk_assessment.iloc[0]
    df_behavior_risk_assessment = df_behavior_risk_assessment.drop(df_behavior_risk_assessment.index[0])
    df_behavior_risk_assessment['Duration'] = abs(df_behavior_risk_assessment['Risk Assessment Start Time'].astype(float) - df_behavior_risk_assessment['Risk Assessment End Time'].astype(float))
    df_behavior_risk_assessment['Pre Risk Assessment'] = 0
    df_behavior_risk_assessment = df_behavior_risk_assessment.reset_index(drop=True)
    edit_col = abs(df_behavior_risk_assessment['Risk Assessment Start Time'].iloc[1:].astype(float).reset_index(drop=True)-df_behavior_risk_assessment['Risk Assessment End Time'].iloc[:-1].astype(float).reset_index(drop=True))
    edit_col.index +=1
    df_behavior_risk_assessment.loc[1:, 'Pre Risk Assessment'] = edit_col
    print(df_behavior_risk_assessment)
    change_coloring(df_behavior_risk_assessment, freezing_behavior_file_df_xlsx)
    dataframe_to_csv(False, df_behavior_risk_assessment, freezing_behavior_file_df)

# print(time_risk)
# print(risk_assessment_time_vec)
#using an average of many times
# list_of_time_risk = []
# for i in np.linspace(-2, 3, 40):
#     risk_assessment_vec, risk_assessment_time_vec = is_risk_assessment(head_length, x_snout, y_snout, likelihood_snout,
#                                                                        x_centerhead, x_spine3, where_door_x,
#                                                                        where_door_y, x_spine2, third_quant,
#                                                                        dist_ear_list, approach_x, time_frame)
#     #print(risk_assessment_time_vec)
#     time_risk = time_risk_assessment(risk_assessment_vec, time_frame)
#     list_of_time_risk.append(time_risk)
# avg_risk_assessment_time = average_calc(list_of_time_risk)


