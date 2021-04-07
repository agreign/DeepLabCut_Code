import pandas as pd
import numpy as np
import cv2
import os
import random
import tkinter as tk
import yaml
import csv

#helper functions
def name_of_line(name_line):
    master = tk.Tk()

    canvas2 = tk.Canvas(master, width=400, height=300)
    canvas2.pack()

    label1 = tk.Label(master, text='Name of the line you would like to make')
    label1.config(font=('Times New Roman', 11))
    canvas2.create_window(200, 100, window=label1)

    entry2 = tk.Entry(master)
    canvas2.create_window(200, 140, window=entry2)

    def new_button():
        n2 = entry2.get()
        name_line.append(n2)
        master.destroy()

    def quit_button():
        master.destroy()

    button2 = tk.Button(text='Enter', command=new_button, bg='brown', fg='white', font=('Times New Roman', 11, 'bold'))
    canvas2.create_window(100, 180, window=button2)
    button3 = tk.Button(text='Exit', command=quit_button, bg='brown', fg='white', font=('Times New Roman', 11, 'bold'))
    canvas2.create_window(300, 180, window=button3)

    master.mainloop()
    if len(name_line) > 0:
        return name_line[0]
    else:
        return []

def name_of_location_and_first_line(name, name_line):
    master = tk.Tk()
    canvas1 = tk.Canvas(master, width = 400, height = 300)
    canvas1.pack()

    label1 = tk.Label(master, text='Name of location that you are labeling')
    label1.config(font=('Times New Roman', 11))
    canvas1.create_window(200, 100, window=label1)

    entry1 = tk.Entry(master)
    canvas1.create_window(200, 140, window=entry1)

    def get_name():
        n1 = entry1.get()
        name.append(n1)
        master.destroy()

    button1 = tk.Button(text='Enter',command=get_name, bg='brown', fg='white', font=('Times New Roman', 10, 'bold'))
    canvas1.create_window(200, 180, window=button1)
    master.mainloop()


    master = tk.Tk()

    canvas2 = tk.Canvas(master, width=400, height=300)
    canvas2.pack()


    label1 = tk.Label(master, text='Name of the line you would like to make')
    label1.config(font=('Times New Roman', 11))
    canvas2.create_window(200, 100, window=label1)

    entry2 = tk.Entry(master)
    canvas2.create_window(200, 140, window=entry1)

    def new_button():
        n2 = entry2.get()
        name_line.append(n2)
        master.destroy()


    button2 = tk.Button(text='Enter', command = new_button, bg='brown', fg='white', font=('Times New Roman', 11, 'bold'))
    canvas2.create_window(200, 180, window=button2)

    master.mainloop()
    return name[0], name_line[0]

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
position_types = []
inp_1 = []
right_click_x = []
right_click_y = []
def click_event(event, x, y, flags, param):
    global inp_1
    global coords
    if event == cv2.EVENT_RBUTTONDOWN:
        inp_1, inp = name_of_location_and_first_line([], [])
        cv2.putText(img, inp_1, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
        right_click_x.append(x)
        right_click_y.append(y)
        position_types.append(inp_1 + '_' + inp)
        cv2.imshow("Frame of Video", img)
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(coords_x) % 2 == 0 and len(coords_x) > 0:
            pass
        elif len(coords_x) == len(position_types):
            cv2.line(img, (coords_x[-1], coords_y[-1]), (x, y), (255, 0, 0), 5)
            cv2.imshow("Frame of Video", img)
            position_types.append(position_types[-1])
            inp = name_of_line([])
            if type(inp) is not list:
                position_types.append(inp_1 + '_' + inp)
        elif len(coords_x) > 0 and position_types[-1].split("_")[0] == position_types[-2].split("_")[0]: #len(coords_x) > 0:
            cv2.line(img, (coords_x[-1], coords_y[-1]), (x, y), (255, 0, 0), 5)
            cv2.imshow("Frame of Video", img)
            position_types.append(position_types[-1])
            inp = name_of_line([])
            if type(inp) is not list:
                position_types.append(inp_1 + '_' + inp)
        coords_x.append(x)
        coords_y.append(y)

def create_line_dictionary(df, coords_x, coords_y):
    #make a dictionary for the lines that designate position area
    dic_box = {}
    for i in range(1, len(position_types)):
        if i % 2 != 0:
            dic_box[position_types[i-1]] = []
            dic_box[position_types[i-1]].append([coords_x[i-1], coords_y[i-1]])
            dic_box[position_types[i-1]].append([coords_x[i], coords_y[i]])
    df.columns = df.iloc[0] #change it so that the columns are equal to the bodyparts
    dic_x = {}
    dic_y = {}
    for key, val in dic_box.items():
        split_line = key.split("_")
        if split_line[0] in dic_x:
            dic_x[split_line[0]].update({split_line[1]: np.linspace(val[0][0], val[1][0], 50)})
            dic_y[split_line[0]].update({split_line[1]: np.linspace(val[0][1], val[1][1], 50)})
        else:
            dic_x[split_line[0]] = {split_line[1]: np.linspace(val[0][0], val[1][0], 50)}
            dic_y[split_line[0]] = {split_line[1]: np.linspace(val[0][1], val[1][1], 50)}

    return dic_x, dic_y

def create_dictionary_for_position(right_click_x, right_click_y):
    dic_x = {}
    dic_y = {}
    areas = []
    for i in range(0, len(position_types)):
        name = position_types[i].split('_')[0]
        if name not in areas:
            areas.append(name)
    for i in range(0, len(right_click_x)):
        dic_x[areas[i]] = right_click_x[i]
        dic_y[areas[i]] = right_click_y[i]

    return dic_x, dic_y, areas

def combine_line_values_x(dic_x, areas):
    copy_dic = dic_x.copy()
    index_im_on_plus_one = 1
    for i in range(0, len(areas)-1):
        for key, value in dic_x[areas[i]].items():
            for j in range(index_im_on_plus_one, len(areas)):
                for key1, value1 in dic_x[areas[j]].items():
                    if False not in set(abs(np.subtract(value,value1)) < 3):
                        v = value.tolist()
                        v1 = value1.tolist()
                        new_line_value = [float(sum(k)) / max(len(k), 1) for k in zip(v, v1)]
                        copy_dic[areas[i]][key] = np.array(new_line_value)
                        copy_dic[areas[j]][key1] = np.array(new_line_value)
    index_im_on_plus_one += 1
    return copy_dic

def open_file(f):
    with open(f) as file:
        documents = yaml.full_load(file)

        for item, doc in documents.items():
            if item == 'video_sets':
                dic_video_crop = {}
                for key, it in doc.items():
                    split_key = key.split("\\")
                    video = split_key[-1]
                    for i, j in it.items():
                        nums_crop = j.split(", ")
                        total_crop_width = nums_crop[1]
                        total_crop_height = nums_crop[3]
                    dic_video_crop[video] = [total_crop_width, total_crop_height]
                    where_videos = doc.values()
    return dic_video_crop

def open_boundary_above(pixel_of_video, dic_y, areas):
    for i in areas:
        for key, values in dic_y[i].items():
            dic_y[i][key] = np.linspace(0, int(pixel_of_video[1]), 50)
    return dic_y

list_of_vids = ['Test 12 - 8T2 - 28 DAYS']
for i in list_of_vids:
    path_to_video = "D:\\Videos\\Lab_Videos\\Fab_Videos\\New_Videos\\" #change
    name_of_video = i + ".mp4" #change
    full_path_to_video = path_to_video + name_of_video

    path_to_frames = "C:\\Users\\Admin\\Desktop\\DeepLabCut\\Frames_Fab\\" #maybe change

    path_to_Video_data = "D:\\Videos\\Lab_Videos\\Fab_Videos\\New_Videos" #change
    name_of_csv_file = i + "DLC_resnet50_Combined_VideosNov14shuffle1_1030000.csv" #change
    df_of_video_information = pd.read_csv(path_to_Video_data + "\\" + name_of_csv_file, low_memory=False)

    config_yaml_path = r"C:\Users\Admin\Desktop\DeepLabCut\conda-environments\Experiment1-Ali-2020-10-17\config.yaml"

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

    [dic_x, dic_y] = create_line_dictionary(df_of_video_information, coords_x, coords_y)
    [right_click_dic_x, right_click_dic_y, area] = create_dictionary_for_position(right_click_x, right_click_y)
    new_dic_x = combine_line_values_x(dic_x, area)
    dic_video_crop = open_file(config_yaml_path)
    print(dic_video_crop )
    new_dic_y = open_boundary_above(dic_video_crop[i + '.mp4'], dic_y, area)
    new_img = cv2.imread(path_to_frames + str(name_of_video) + "\\" + str(picture_path), 0)

    for key, values in new_dic_x.items():
        for key1, values1 in new_dic_x[key].items():
            cv2.line(new_img, (int(values1[0]), int(new_dic_y[key][key1][0])), (int(values1[-1]), int(new_dic_y[key][key1][-1])), (255, 0, 0), 5)
    for key, value in right_click_dic_x.items():
        cv2.putText(new_img, key, (value, right_click_dic_y[key]), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Final Image', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    data_dic_x = pd.DataFrame(dic_x)
    data_dic_x.to_csv('D:\\Documents\\deeplabcut_trial_2\\Dictionaries_of_positions\\' + i + '\\df_dict_test_x.csv')
    data_dic_y = pd.DataFrame(dic_y)
    data_dic_y.to_csv('D:\\Documents\\deeplabcut_trial_2\\Dictionaries_of_positions\\' + i + '\\df_dict_test_y.csv')
    with open('D:\\Documents\\deeplabcut_trial_2\\Dictionaries_of_positions\\' + i + '\\dict_test_right_click_x.csv', 'w') as csv_file2:
        writer = csv.writer(csv_file2)
        for key, value in right_click_dic_x.items():
            writer.writerow([key, value])
    with open('D:\\Documents\\deeplabcut_trial_2\\Dictionaries_of_positions\\' + i + '\\dict_test_right_click_y.csv', 'w') as csv_file3:
        writer = csv.writer(csv_file3)
        for key, value in right_click_dic_y.items():
            writer.writerow([key, value])
