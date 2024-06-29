import json
import os
import re
from datetime import datetime
import torch
from skimage.transform import resize
import numpy as np
from moviepy.editor import VideoFileClip
import pandas as pd


def read_srt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def parse_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    pattern = re.compile(r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+((?:.|\s)*?)(?=\n\d+|\Z)')
    matches = pattern.findall(content)

    subtitles = []
    for match in matches:
        subtitles.append({
            'index': int(match[0]),
            'start_time': match[1],
            'end_time': match[2],
            'text': match[3].strip()
        })
    return subtitles


def time_str_to_seconds(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S,%f')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000


def get_subtitles_in_time_range(subtitles, start_time, end_time):
    start_seconds = start_time
    end_seconds = end_time

    result = []
    for subtitle in subtitles:
        subtitle_start = time_str_to_seconds(subtitle['start_time'])
        subtitle_end = time_str_to_seconds(subtitle['end_time'])

        if subtitle_end >= start_seconds and subtitle_start <= end_seconds:
            result.append(subtitle['text'])

    return " ".join(result)


def read_videos_and_step_start_end(root_path):
    csv_root_path = os.path.join(root_path, 'csvs')
    all_csvs = sorted(os.listdir(csv_root_path))

    video_names = []
    for i in range(len(all_csvs)):
        video_names.append(all_csvs[i].split('.')[0])

    csv_path = []
    vid_path = []
    for i in range(len(video_names)):
        csv_path.append(os.path.join(csv_root_path, video_names[i] + '.csv'))
        video_type = video_names[i].replace('_' + video_names[i].split('_')[-1], '')
        vid_path.append(os.path.join(root_path, video_type, 'videos', video_names[i] + '.mpg'))

    return csv_path, vid_path


def get_video_path(root_path):
    action_list = ['changing_tire', 'coffee', 'cpr', 'jump_car', 'repot']
    video_paths = []
    for action in action_list:
        list = os.listdir(os.path.join(root_path, action, 'videos'))
        for item in list:
            video_paths.append(os.path.join(root_path, action, 'videos', item))

    return video_paths


def get_csv_path_from_video_path(path):
    root_dir = '/scratch/users/tang/data/niv/csvs'
    file_name = os.path.basename(path)
    file_name = file_name.split('.')[0]
    csv = os.path.join(root_dir, file_name + '.csv')

    return csv


def get_srt_path_from_video_path(path):
    file_name = os.path.basename(path)
    file_name = file_name.split('.')[0]
    root_path = path.split('videos')[0]

    srt = os.path.join(root_path, 'subtitles', 'manual', file_name + '.srt')

    return srt


def read_csv(path):
    column_names = ['Action', 'Start', 'End']
    df = pd.read_csv(path, header=None, names=column_names)

    return df


def get_video_clip(path, start, end):
    video = VideoFileClip(path)
    fps = video.fps
    start_frame, end_frame = int(start*fps), int(end*fps)
    #print(f"Original video size: {video.size}")
    #print(f"Video FPS: {video.fps}")
    #print(start, end)
    start_clip, end_clip = [], []

    for i, frame in enumerate(video.iter_frames()):
        if start_frame-2 <= i <= start_frame+1:
            start_clip.append(frame)
        elif end_frame-1 <= i <= end_frame+2:
            end_clip.append(frame)

    return np.array(start_clip), np.array(end_clip)


def preprocess_frames(video_array, target_size=(224, 224)):
    video_array = video_array / 255.0
    preprocessed_frames = []
    for frame in video_array:
        frame_resized = resize(frame, target_size, anti_aliasing=True)
        preprocessed_frames.append(frame_resized)

    preprocessed_array = np.array(preprocessed_frames)
    preprocessed_array = preprocessed_array.reshape(1, 1, 4, *preprocessed_array.shape[1:]) # (1, 2, 22, 224, 224, 3)
    video_input = torch.from_numpy(preprocessed_array)

    return video_input


def nearest_even(n):
    if n % 2 == 0:
        return n
    else:
        return n - 1 if n % 2 == 1 else n + 1


ok = False
if ok :
    root_dir = '/scratch/users/tang/data/niv'
    video_paths = get_video_path(root_dir)
    for video_path in video_paths:
        csv_path = get_csv_path_from_video_path(video_path)
        srt_path = get_srt_path_from_video_path(video_path)
        if os.path.exists(csv_path):
            df = read_csv(csv_path)
            subtitles = parse_srt(srt_path)
            print(df.shape)

            for index, row in df.iterrows():
                action = row['Action']
                start_seconds = row['Start']
                end_seconds = row['End']

                extracted_text = get_subtitles_in_time_range(subtitles, start_seconds, end_seconds)
                start_clip, end_clip = get_video_clip(video_path, start_seconds, end_seconds)
                start_clip = preprocess_frames(start_clip)
                end_clip = preprocess_frames(end_clip)

                print(f"Action: {action}")
                print(f"Start: {start_seconds} seconds")
                print(f"End: {end_seconds} seconds")
                print(end_clip.shape)
                print(f"Extracted Text: {extracted_text}\n")


        break


if __name__ == '__main__':
    json_dir = '/Users/tangwenwu/Documents/GitHub/Action_diffusion/action_diffusion/dataset/NIV/train_split_T3.json'
    with open(json_dir, 'r') as file:
        data = json.load(file)

    for item in data:
        feature = item['id']['feature']
        legal_range = item['id']['legal_range']
        task_id = item['id']['task_id']
        instruction_len = item['instruction_len']
        for i in legal_range:
            steps_ids = i[2]
            start_time = i[0]
            end_time = i[1]


        print(f"Feature: {feature}")
        print(f"Legal Range: {legal_range}")
        print(f"Task ID: {task_id}")
        print(f"Instruction Length: {instruction_len}")
        print("----------------------")


    print(j)

