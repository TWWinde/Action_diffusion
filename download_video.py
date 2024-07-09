import csv
import os
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


def download_video(video_url, output_path):
    ydl_opts = {
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'format': 'best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


videos = read_videos('/scratch/users/tang/data/crosstask_release/videos.csv')

output_dir = '/scratch/users/tang/data/crosstask_release/videos'
os.makedirs(output_dir, exist_ok=True)

for video in videos:
    download_video(video['url'], output_dir)