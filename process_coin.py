import json
import os
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
    if num_frames % num != 0:
        num_additional_frames = num - (num_frames % 4)
        black_frame = np.zeros((224, 224, 3), dtype=np.float32)
        for _ in range(num_additional_frames):
            preprocessed_frames.append(black_frame)
        preprocessed_array = np.array(preprocessed_frames)

    preprocessed_array = preprocessed_array.reshape(1, 1, num, *preprocessed_array.shape[1:])  # (1, 2, 22, 224, 224, 3)
    video_input = torch.from_numpy(preprocessed_array)

    return video_input.float()


if __name__ =="__main__":
    root_path = "/scratch/users/tang/data/COIN/videos"
    json_path = "/scratch/users/tang/data/COIN/COIN.json"
    save_root_path = "/scratch/users/tang/data/COIN/"
    with open(json_path, 'r') as f:
        file_content = f.read()
        if not file_content.strip():
            raise ValueError("JSON file is empty")
        data = json.loads(file_content)
    model, tokenizer, aligner = MMPTModel.from_pretrained(
        "/scratch/users/tang/fairseq/examples/MMPT/projects/retri/videoclip/how2.yaml")
    model.eval()
    also_end = False
    for video_id, video_info in data['database'].items():
        data_list = []
        task_id = video_info['recipe_type']
        video_path = os.path.join(root_path, str(task_id), video_id + ".mp4")
        try:
            video = VideoFileClip(video_path)
        except:
            print(video_path, 'was not found')
        for clip in video_info['annotation']:
            action_id = clip['id']
            start_time, end_time = clip['segment'][0], clip['segment'][1]
            text = clip['label']
            start_clip, end_clip = get_video_clip(video, start_time, end_time)
            start_video_input = preprocess_frames(start_clip, target_size=(224, 224), num=16)
            if also_end:
                end_video_input = preprocess_frames(end_clip, target_size=(224, 224), num=16)

            caps, cmasks = aligner._build_text_seq(tokenizer(text, add_special_tokens=False)["input_ids"])
            caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
            # start_clip = start_clip.to(torch.float64)
            with torch.no_grad():
                output1 = model(start_clip, caps, cmasks, return_score=False)
                if also_end:
                    output2 = model(end_clip, caps, cmasks, return_score=False)

            start_video_feature, start_text_feature = output1["pooled_video"], output1["pooled_text"]  # torch.Size([1, 768])
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
    file_name = os.path.basename(video_path).split('.')[0] + '.npy'
    save_path = os.path.join(save_root_path, file_name)
    np.save(save_path, data_list)


