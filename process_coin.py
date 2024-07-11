import json



json_path = "/scratch/users/tang/data/COIN/COIN.json"
data = json.loads(json_path)

for video_id in data['database'].items():
    print(f"Video ID: {video_id}")
    print(f"  Recipe Type: {video_info['recipe_type']}")
    print(f"  Class: {video_info['class']}")
    print(f"  Subset: {video_info['subset']}")
    print(f"  Video URL: {video_info['video_url']}")
    print(f"  Duration: {video_info['duration']}")
    print("  Annotations:")
    for annotation in video_info['annotation']:
        print(f"    ID: {annotation['id']}")
        print(f"    Segment: {annotation['segment']}")
        print(f"    Label: {annotation['label']}")