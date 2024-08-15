import os
import json
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, help='video root')
    parser.add_argument('--vid_id', type=str, help='video name')
    args = parser.parse_args()
    raw_vid_path = f'{args.work_dir}/{args.vid_id}.mp4'
    clip_json = f'{args.work_dir}/{args.vid_id}/pyavi/clips.json'
    save_folder = f'{args.work_dir}/clips/{args.vid_id}'
    if os.path.isdir(save_folder):
        subprocess.call(f'rm -r {save_folder}',shell=True)
    os.makedirs(save_folder,exist_ok=True)
    ffmpeg_cmd = ['ffmpeg']
    filters = []
    cmd_post = []
    filters_dic = json.load(open(clip_json))
    for index, save_vid_name in enumerate(filters_dic.keys()):
        item = filters_dic[save_vid_name]
        out_path = os.path.join(save_folder, save_vid_name+'.mp4')
        left, right, top, bottom = item['box'].values()
        start_sec, end_sec = item['time'].values()
        video_filter = f"[0:v]trim=start={start_sec}:end={end_sec},setpts=PTS-STARTPTS,crop={right-left}:{bottom-top}:{left}:{top},scale=512:512[v{index}];"
        audio_filter = f"[0:a]atrim=start={start_sec}:end={end_sec},asetpts=PTS-STARTPTS[a{index}];"
        filters.append(video_filter + audio_filter)
        cmd_post.extend(['-map', f'[v{index}]', '-map', f'[a{index}]', '-async', '1', '-r', '25', '-c:v' ,'libx264' ,'-c:a' ,'aac', '-y', '-loglevel', 'panic', out_path])
    filter_complex = ''.join(filters)[:-1]
    ffmpeg_cmd.insert(1, '-i')
    ffmpeg_cmd.insert(2, raw_vid_path)
    ffmpeg_cmd.extend(['-filter_complex', filter_complex])
    ffmpeg_cmd.extend(cmd_post)
    subprocess.run(ffmpeg_cmd)
