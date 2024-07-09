import csv
import os

import numpy as np
import yt_dlp


def read_videos(file_path):
    videos = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            videos.append({
                'task_id': int(row[0]),
                'video_id': row[1],
                'url': row[2]
            })
    return videos


def download_video(video_url, output_path, video):
    fail_case=[]

    filename = f"{video['task_id']}_{video['video_id']}.mp4"
    ydl_opts = {
        'outtmpl': os.path.join(output_path, filename),
        'format': 'best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
        except:
            print(video_url, 'is not available')
            fail_case.append(video)

    np.save('/scratch/users/tang/data/crosstask_release/', 'fail_case.py')


videos = read_videos('/scratch/users/tang/data/crosstask_release/videos.csv')

output_dir = '/scratch/users/tang/data/crosstask_release/videos'
os.makedirs(output_dir, exist_ok=True)

for video in videos:
    download_video(video['url'], output_dir, video)