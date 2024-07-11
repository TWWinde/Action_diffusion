import json



json_path = "/scratch/users/tang/data/COIN/COIN.json"
with open(json_path, 'r') as f:
    file_content = f.read()
    if not file_content.strip():
        raise ValueError("JSON file is empty")
    data = json.loads(file_content)

for video_id, video_info in data['database'].items():
    print(f"Video ID: {video_id}")
    break
