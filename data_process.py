import glob
import os
import re
from datetime import datetime
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
    print(start_time)
    start_seconds = time_str_to_seconds(start_time)
    end_seconds = time_str_to_seconds(end_time)

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
    print(srt)

    return srt


def read_csv(path):
    column_names = ['Action', 'Start', 'End']
    df = pd.read_csv(path, header=None, names=column_names)

    return df


def get_video_clip(path, start, end):
    video = VideoFileClip(path)
    cropped_video = video.subclip(start, end)

    return cropped_video


if __name__ == '__main__':
    root_dir = '/scratch/users/tang/data/niv'
    video_paths = get_video_path(root_dir)
    for video_path in video_paths:
        csv_path = get_csv_path_from_video_path(video_path)
        srt_path = get_srt_path_from_video_path(video_path)
        if os.path.exists(csv_path):
            df = read_csv(csv_path)
            subtitles = parse_srt(srt_path)

            for index, row in df.iterrows():
                action = row['Action']
                start_seconds = row['Start']
                end_seconds = row['End']

                extracted_text = get_subtitles_in_time_range(subtitles, start_seconds, end_seconds)

                print(f"Action: {action}")
                print(f"Start: {start_seconds} seconds")
                print(f"End: {end_seconds} seconds")
                print(f"Extracted Text: {extracted_text}\n")

                break



