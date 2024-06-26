import glob
import os

import pandas as pd


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


def read_csv(path):
    df = pd.read_csv(path)

    print(df)



if __name__ =='__main__':

    csv_path, vid_path = read_videos_and_step_start_end('/scratch/users/tang/data/niv')
    read_csv(csv_path[7])