import sys

sys.path.append('/scratch/users/tang/fairseq/examples/MMPT')
import os
import re
from datetime import datetime
import torch
from skimage.transform import resize
import numpy as np
from moviepy.editor import VideoFileClip
import pandas as pd
import torch
from mmpt.models import MMPTModel
import math


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


def get_video_clip(path, start, end, num=16, also_end=True):
    video = VideoFileClip(path)
    fps = video.fps
    start_frame, end_frame = int(start * fps), int(end * fps)
    # print(f"Original video size: {video.size}")
    # print(f"Video FPS: {video.fps}")
    # print(start, end)
    start_clip, end_clip = [], []

    for i, frame in enumerate(video.iter_frames()):
        if start_frame - (num - 2) <= i <= start_frame + 1:
            start_clip.append(frame)
        elif end_frame - 1 <= i <= end_frame + (num - 2) and also_end:
            end_clip.append(frame)

    return np.array(start_clip), np.array(end_clip)


def preprocess_frames(video_array, target_size=(224, 224), num=16):
    video_array = (video_array / 255.0) * 2.0 - 1.0
    preprocessed_frames = []
    for frame in video_array:
        frame_resized = resize(frame, target_size, anti_aliasing=True)
        preprocessed_frames.append(frame_resized)

    preprocessed_array = np.array(preprocessed_frames)
    num_frames = preprocessed_array.shape[0]
    if num_frames % num != 0:
        num_additional_frames = num - (num_frames % 4)
        black_frame = np.zeros((224, 224, 3), dtype=np.float32)
        for _ in range(num_additional_frames):
            preprocessed_frames.append(black_frame)
        preprocessed_array = np.array(preprocessed_frames)

    preprocessed_array = preprocessed_array.reshape(1, 1, num, *preprocessed_array.shape[1:])  # (1, 2, 22, 224, 224, 3)
    video_input = torch.from_numpy(preprocessed_array)

    return video_input.float()


def nearest_even(n):
    if n % 2 == 0:
        return n
    else:
        return n - 1 if n % 2 == 1 else n + 1


class_list = {
    "BRAKE ON": {
        "steps_ids": 8,
        "task_id": 0
    },
    "GET THINGS OUT": {
        "steps_ids": 9,
        "task_id": 0
    },
    "START LOOSE": {
        "steps_ids": 0,
        "task_id": 0
    },
    "JACK UP": {
        "steps_ids": 1,
        "task_id": 0
    },
    "UNSCREW WHEEL": {
        "steps_ids": 2,
        "task_id": 0
    },
    "WITHDRAW WHEEL": {
        "steps_ids": 3,
        "task_id": 0
    },
    "PUT WHEEL": {
        "steps_ids": 4,
        "task_id": 0
    },
    "SCREW WHEEL": {
        "steps_ids": 5,
        "task_id": 0
    },
    "JACK DOWN": {
        "steps_ids": 6,
        "task_id": 0
    },
    "TIGHT WHEEL": {
        "steps_ids": 7,
        "task_id": 0
    },
    "PUT THINGS BACK": {
        "steps_ids": 10,
        "task_id": 0
    },
    "PUT SOIL": {
        "steps_ids": 11,
        "task_id": 4
    },
    "LOOSEN ROOT": {
        "steps_ids": 13,
        "task_id": 4
    },
    "PLACE PLANT": {
        "steps_ids": 14,
        "task_id": 4
    },
    "ADD TOP": {
        "steps_ids": 15,
        "task_id": 4
    },
    "TAKE PLANT": {
        "steps_ids": 12,
        "task_id": 4
    },
    "WATER PLANT": {
        "steps_ids": 16,
        "task_id": 4
    },
    "TAP POT": {
        "steps_ids": 17,
        "task_id": 4
    },
    "COVER HOLE": {
        "steps_ids": 18,
        "task_id": 4
    },
    "FILL WATER": {
        "steps_ids": 19,
        "task_id": 1
    },
    "ADD COFFEE": {
        "steps_ids": 20,
        "task_id": 1
    },
    "EVEN SURFACE": {
        "steps_ids": 21,
        "task_id": 1
    },
    "PUT FILTER": {
        "steps_ids": 22,
        "task_id": 1
    },
    "SCREW TOP": {
        "steps_ids": 23,
        "task_id": 1
    },
    "SEE COFFEE": {
        "steps_ids": 24,
        "task_id": 1
    },
    "WITHDRAW STOVE": {
        "steps_ids": 25,
        "task_id": 1
    },
    "POUR COFEE": {
        "steps_ids": 26,
        "task_id": 1
    },
    "PUT STOVE": {
        "steps_ids": 27,
        "task_id": 1
    },
    "GRIND COFFE": {
        "steps_ids": 28,
        "task_id": 1
    },
    "CHECK BREATHING": {
        "steps_ids": 29,
        "task_id": 2
    },
    "CHECK RESPONSE": {
        "steps_ids": 30,
        "task_id": 2
    },
    "GIVE BREATH": {
        "steps_ids": 31,
        "task_id": 2
    },
    "GIVE COMPRESSION": {
        "steps_ids": 32,
        "task_id": 2
    },
    "CHECK PULSE": {
        "steps_ids": 33,
        "task_id": 2
    },
    "OPEN AIRWAY": {
        "steps_ids": 34,
        "task_id": 2
    },
    "CALL EMERGENCY": {
        "steps_ids": 35,
        "task_id": 2
    },
    "GET CAR": {
        "steps_ids": 36,
        "task_id": 3
    },
    "CONNECT REDEMPTY": {
        "steps_ids": 37,
        "task_id": 3
    },
    "CONNECT REDFULL": {
        "steps_ids": 38,
        "task_id": 3
    },
    "CONNECT BLACK": {
        "steps_ids": 39,
        "task_id": 3
    },
    "GROUND BLACK": {
        "steps_ids": 40,
        "task_id": 3
    },
    "START FULLCAR": {
        "steps_ids": 41,
        "task_id": 3
    },
    "START EMPTYCAR": {
        "steps_ids": 42,
        "task_id": 3
    },
    "REMOVE GROUND": {
        "steps_ids": 43,
        "task_id": 3
    },
    "REMOVE BLACK": {
        "steps_ids": 44,
        "task_id": 3
    },
    "DISCONNECT REDFULL": {
        "steps_ids": 45,
        "task_id": 3
    },
    "DISCONNECT REDEMPTY": {
        "steps_ids": 46,
        "task_id": 3
    },
    "OPEN HOOD": {
        "steps_ids": 47,
        "task_id": 3
    }
}

if __name__ == '__main__':
    root_dir = '/scratch/users/tang/data/niv'
    # save_root_path = '/scratch/users/tang/data/niv/processed_data'
    save_root_path = '/scratch/users/tang/data/niv/processed_data_16_onlystart_pooled'
    os.makedirs(save_root_path, exist_ok=True)
    video_paths = get_video_path(root_dir)
    model, tokenizer, aligner = MMPTModel.from_pretrained(
        "/scratch/users/tang/fairseq/examples/MMPT/projects/retri/videoclip/how2.yaml")
    model.eval()

    for video_path in video_paths:
        data_list = []
        csv_path = get_csv_path_from_video_path(video_path)
        srt_path = get_srt_path_from_video_path(video_path)
        if os.path.exists(csv_path):
            file_name = os.path.basename(csv_path).split('.')[0]
            print(file_name)
            df = read_csv(csv_path)
            subtitles = parse_srt(srt_path)
            also_end = False
            for index, row in df.iterrows():
                action = row['Action']
                steps_ids, task_id = class_list[action.strip()]["steps_ids"], class_list[action.strip()]["task_id"]
                print(steps_ids, task_id)
                start_seconds = math.floor(row['Start'])
                end_seconds = math.ceil(row['End'])
                extracted_text = get_subtitles_in_time_range(subtitles, start_seconds, end_seconds)
                start_clip, end_clip = get_video_clip(video_path, start_seconds, end_seconds, num=16, also_end=also_end)
                if start_clip.size == 0:  # or end_clip.size == 0:
                    continue
                start_clip = preprocess_frames(start_clip)
                if also_end:
                    end_clip = preprocess_frames(end_clip)
                details = False
                print(f"Action: {action}")
                if details:
                    print(f"Start: {start_seconds} seconds")
                    print(f"End: {end_seconds} seconds")
                    print(video_input.shape)
                    print(video_input)
                    print(f"Extracted Text: {extracted_text}\n")

                # B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
                # video_frames = torch.randn(1, 1, 4, 224, 224, 3)

                caps, cmasks = aligner._build_text_seq(tokenizer(extracted_text, add_special_tokens=False)["input_ids"])
                caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
                # start_clip = start_clip.to(torch.float64)
                with torch.no_grad():
                    output1 = model(start_clip, caps, cmasks, return_score=False)
                    if also_end:
                        output2 = model(end_clip, caps, cmasks, return_score=False)

                start_video_feature, start_text_feature = output1["pooled_video"], output1[
                    "pooled_text"]  # torch.Size([1, 768])
                start_video_feature_np = start_video_feature.cpu().numpy()
                start_text_feature_np = start_text_feature.cpu().numpy()
                print(start_video_feature_np.shape)
                print(start_text_feature_np.shape)
                data_dict_start = {
                    'file_name': file_name,
                    'steps_ids': steps_ids,
                    'task_id': task_id,
                    'start_seconds': start_seconds,
                    'end_seconds': end_seconds,
                    'video_feature': start_video_feature_np,
                    'text_feature': start_text_feature_np,
                    'start_end': 0

                }
                data_list.append(data_dict_start)
                if also_end:
                    end_video_feature, end_text_feature = output2["video"], output2["text"]
                    end_video_feature_np = end_video_feature.cpu().numpy()
                    end_text_feature_np = end_text_feature.cpu().numpy()
                    data_dict_end = {
                        'file_name': file_name,
                        'steps_ids': steps_ids,
                        'task_id': task_id,
                        'start_seconds': start_seconds,
                        'end_seconds': end_seconds,
                        'video_feature': end_video_feature_np,
                        'text_feature': end_text_feature_np,
                        'start_end': 1
                    }

                    data_list.append(data_dict_end)
        file_name = os.path.basename(video_path).split('.')[0] + '.npy'
        save_path = os.path.join(save_root_path, file_name)
        np.save(save_path, data_list)








