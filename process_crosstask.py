import os
import csv
from skimage.transform import resize
import numpy as np
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
from mmpt.models import MMPTModel


def get_video_clip(video, start, end, num=16, also_end=True):

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
    if num_frames < (num / 2):
        return []
    if num_frames % num != 0:
        num_additional_frames = num - (num_frames % num)
        black_frame = np.zeros((224, 224, 3), dtype=np.float32)
        for _ in range(num_additional_frames):
            preprocessed_frames.append(black_frame)
        preprocessed_array = np.array(preprocessed_frames)

    preprocessed_array = preprocessed_array.reshape(1, 1, num, *preprocessed_array.shape[1:])  # (1, 2, 22, 224, 224, 3)
    video_input = torch.from_numpy(preprocessed_array)

    return video_input.float()


def parse_tasks_file(file_path):
    tasks = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        task = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.isdigit() and 'Task ID' not in task:
                task['Task ID'] = line
            elif line.startswith("http"):
                task['URL'] = line
            elif line.isnumeric() and 'Number of steps' not in task:
                task['Number of steps'] = int(line)
            elif 'Task ID' in task and 'Task name' not in task:
                task['Task name'] = line
            else:
                task['Steps'] = line.split(',')
                tasks[task['Task ID']] = task
                task = {}

    return tasks


def parse_annotation_file(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            annotations.append({'Step number': int(row[0]), 'Start': float(row[1]), 'End': float(row[2])})
    return annotations



if __name__ =="__main__":
    aug_times = 1  # how many times do you want to augment the data
    also_end = False
    if also_end:
        a = 'start_end'
    else:
        a = 'onlystart'
    root_path = "/scratch/users/tang/data/crosstask_release"
    annotation_path = "/scratch/users/tang/data/crosstask_release/annotations"
    video_path = "/scratch/users/tang/data/crosstask_release/videos"
    save_root_path = f"/scratch/users/tang/data/crosstask_release/processed_{a}_x{aug_times}/"
    os.makedirs(save_root_path, exist_ok=True)
    model, tokenizer, aligner = MMPTModel.from_pretrained(
        "/scratch/users/tang/fairseq/examples/MMPT/projects/retri/videoclip/how2.yaml")
    model.eval()
    also_end = False
    primary_tasks = parse_tasks_file('/scratch/users/tang/data/crosstask_release/tasks_primary.txt')
    related_tasks = parse_tasks_file('/scratch/users/tang/data/crosstask_release/tasks_related.txt')
    primary_tasks.update(related_tasks)
    video_list = os.listdir(video_path)
    n=0
    action_list = {}
    for i in range(aug_times):
        for video_name in video_list:
            try:
                data_list = []
                anotation_name = video_name.split('.')[0] + '.csv'
                anotation_name_path = os.path.join(annotation_path, anotation_name)
                video_name_path = os.path.join(video_path, video_name)
                if os.path.exists(anotation_name_path):
                    annotations = parse_annotation_file(anotation_name_path)
                else:
                    continue
                task_id = video_name.split('_')[0]
                video = VideoFileClip(video_name_path)

                for clip in annotations:
                    step_number = clip['Step number']
                    step_name = primary_tasks[task_id]['Steps'][step_number - 1]
                    print(step_name)
                    if step_name not in action_list:
                        action_list[step_name] = n
                        action_id = n
                        print('new action id', n)
                        n += 1
                    else:
                        action_id = action_list[step_name]

                    start_time, end_time = clip['Start'], clip['End']
                    text = step_name
                    start_clip, end_clip = get_video_clip(video, start_time, end_time)
                    start_video_input = preprocess_frames(start_clip, target_size=(224, 224), num=16)
                    if start_video_input.shape != (1, 1, 16, 224, 224, 3):
                        print('input error')
                        continue
                    if also_end:
                        end_video_input = preprocess_frames(end_clip, target_size=(224, 224), num=16)
                        if end_video_input.shape != (1, 1, 16, 224, 224, 3):
                            print('input error')
                            continue

                    caps, cmasks = aligner._build_text_seq(tokenizer(text, add_special_tokens=False)["input_ids"])
                    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
                    # start_clip = start_clip.to(torch.float64)
                    with torch.no_grad():
                        output1 = model(start_video_input, caps, cmasks, return_score=False)
                        if also_end:
                            output2 = model(end_video_input, caps, cmasks, return_score=False)

                    start_video_feature, start_text_feature = output1["pooled_video"], output1[
                        "pooled_text"]  # torch.Size([1, 768])
                    start_video_feature_np = start_video_feature.cpu().numpy()
                    start_text_feature_np = start_text_feature.cpu().numpy()
                    print(start_video_feature_np.shape)
                    print(start_text_feature_np.shape)
                    data_dict_start = {
                        'steps_ids': action_id,
                        'task_id': task_id,
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
                            'steps_ids': action_id,
                            'task_id': task_id,
                            'video_feature': start_video_feature_np,
                            'text_feature': start_text_feature_np,
                            'start_end': 1

                        }

                        data_list.append(data_dict_end)
            except:
                continue

            file_name = os.path.basename(video_name_path).split('.')[0] + f'_{i}.npy'
            save_path = os.path.join(save_root_path, file_name)
            np.save(save_path, data_list)
            file_name = 'class_list.npy'
            save_path = os.path.join(save_root_path, file_name)
            np.save(save_path, action_list)




