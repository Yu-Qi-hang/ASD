import os
import argparse
import subprocess

from multiprocessing import Pool

def async_proxy(cmd):
    subprocess.run(cmd,shell=True)

vid_ext = ['.mp4','.avi']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='demo', help='dir of videos')
    parser.add_argument('--vid_id', type=str, default='', help='name of videos')
    parser.add_argument('--num_worker', type=int, default=1, help='num-workers of process')
    args = parser.parse_args()

    videos = [x[:-4] for x in os.listdir(args.video_dir) if x[-4:] in vid_ext] if args.vid_id == '' else [x.split('.')[0] for x in args.vid_id.split(',')]

    pool = Pool(args.num_worker)
    for video in videos:
        pool.apply_async(async_proxy,(f'python evaluate.py  --videoFolder {args.video_dir} --videoName {video} && python process.py --work_dir {args.video_dir} --vid_id {video}',))
    pool.close()
    pool.join()
