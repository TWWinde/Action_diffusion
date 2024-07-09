import csv
import os
import numpy as np
import yt_dlp
import json



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


cross_task=False
coin = True
if cross_task:
    videos = read_videos('/scratch/users/tang/data/crosstask_release/videos.csv')

    output_dir = '/scratch/users/tang/data/crosstask_release/videos'
    os.makedirs(output_dir, exist_ok=True)

    for video in videos:
        download_video(video['url'], output_dir, video)


if coin:

    output_path = '/scratch/users/tang/data/COIN/videos'
    json_path = '/scratch/users/tang/data/COIN.json'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    data = json.load(open(json_path, 'r'))['database']
    youtube_ids = list(data.keys())
    fail_case = []
    for youtube_id in data:
        info = data[youtube_id]
        type = info['recipe_type']
        url = info['video_url']
        vid_loc = output_path + '/' + str(type)

        if not os.path.exists(vid_loc):
            os.mkdir(vid_loc)
        os.system('youtube-dl -o ' + vid_loc + '/' + youtube_id + '.mp4' + ' -f best ' + url)

        filename = f"{youtube_id}.mp4"
        ydl_opts = {
            'outtmpl': os.path.join(vid_loc, filename),
            'format': 'best',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download(url)
            except:
                print(url, 'is not available')
                fail_case.append(youtube_id)

        np.save('/scratch/users/tang/data/COIN/', 'fail_case.py')


