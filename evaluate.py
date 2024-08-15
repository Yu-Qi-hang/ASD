import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features, json, threading

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from ASD import ASD

import matplotlib.pyplot as plt

def mergeclip(mylist=[],threshold=5):
	for i in range(len(mylist)-1):
		if mylist[i]==mylist[i+1]:
			continue
		else:
			cnt = 1
			while True:
				if i+cnt > len(mylist)-2:
					break
				if mylist[i] != mylist[i+cnt]:
					cnt = cnt+1
				else:
					break
			if i+cnt > len(mylist)-2:
				break
			if cnt < threshold:
				for idx in range(cnt):
					mylist[i+idx+1]=mylist[i]
	return mylist

def add_box(area,box):
	area[0] = min(area[0],box[0])#left
	area[1] = min(area[1],box[1])#top
	area[2] = max(area[2],box[2])#right
	area[3] = max(area[3],box[3])#bottom
	return area

def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList

def inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cuda')
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		dets.append([])
		for bbox in bboxes:
		  dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])#left
	yA = max(boxA[1], boxB[1])#top
	xB = min(boxA[2], boxB[2])#right
	yB = min(boxA[3], boxB[3])#bottom
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(args, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v libx264 -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	numpy.save(featuresPath, mfcc)

def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained model
	s = ASD()
	s.loadParameters(args.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
		_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
		videoFeature = []
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break
		video.release()
		videoFeature = numpy.array(videoFeature)
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * 25)),:,:]
		allScore = [] # Evaluation use model
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)	
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)	
	return allScores

def visualization(tracks, scores, args):
	crop_list = []
	all_scores = []
	# CPU: visulize the result for video format
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			# s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			# s = numpy.mean(s)
			s = score[fidx] if fidx<len(score) - 1 else score[-1]
			all_scores.append(s)
			if len(faces[frame]) == 0:
				faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
			elif float(s) > faces[frame][0]['score']:
				faces[frame][0]= {'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]}
	firstImage = cv2.imread(flist[0])
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	# vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
	# colorDict = {0: 0, 1: 255}
	spike_stat = -1
	pre_area = [fw,fh,0,0]
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		# image = cv2.imread(fname)
		for face in faces[fidx]:
			# clr = colorDict[int((face['score'] >= 0))]
			# txt = round(face['score'], 1)
			# cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),3)
			# cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s']-3)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
			if spike_stat < 0 and face['score'] > 0:
				spike_stat = 1
				area = [int(face['x']-face['s']), int(face['y']-face['s']), int(face['x']+face['s']), int(face['y']+face['s'])]
				start_idx = fidx+1
				end_idx = start_idx
				pre_area = area
				# print(area)
			elif spike_stat > 0 and face['score'] > 0:
				new_area = [int(face['x']-face['s']), int(face['y']-face['s']), int(face['x']+face['s']), int(face['y']+face['s'])]
				# print(new_area)
				if bb_intersection_over_union(pre_area, new_area) < 0.14:
					end_idx = fidx-2
					if end_idx - start_idx >= 60:
						cx = (area[0]+area[2])/2
						cy = (area[1]+area[3])/2
						length = min(int(max(area[2]-area[0],area[3]-area[1])*1.5),cx*2,cy*2,(fw-cx)*2,(fh-cy)*2)
						left = int(cx-length/2)
						top = int(cy-length/2)
						crop_list.append({"time":{"start_sec":start_idx/25,"end_sec":end_idx/25},"box":{"left":left,"right":left+length,"top":top,"bottom":top+length}})
					spike_stat = -1
				else:
					area = add_box(area, new_area)
					pre_area = new_area
			elif spike_stat > 0 and face['score'] < 0:
				end_idx = fidx-2
				if end_idx - start_idx >= 60:
					cx = (area[0]+area[2])/2
					cy = (area[1]+area[3])/2
					length = min(int(max(area[2]-area[0],area[3]-area[1])*1.5),cx*2,cy*2,(fw-cx)*2,(fh-cy)*2)
					left = int(cx-length/2)
					top = int(cy-length/2)
					crop_list.append({"time":{"start_sec":start_idx/25,"end_sec":end_idx/25},"box":{"left":left,"right":left+length,"top":top,"bottom":top+length}})
				spike_stat = -1
		# vOut.write(image)
	# vOut.release()
	crop_json = {}
	for idx,item in enumerate(crop_list):
		crop_json[f'{args.videoName}_{idx}'] = item
	with open(f'{args.pyaviPath}/clips.json','w') as fp:
		json.dump(crop_json,fp,indent=4)
	# command = ("ffmpeg -y -i %s -i %s -threads %d -c:v libx264 -c:a aac %s -loglevel panic" % \
	# 	(os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
	# 	args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.mp4'))) 
	# output = subprocess.call(command, shell=True, stdout=None)
	return all_scores

def inference_video_proxy(args):
	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	if os.path.isfile(savePath):
		with open(savePath, 'rb') as fil:
			faces = pickle.load(fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face already existed in %s \r\n" %(args.pyworkPath))
	else:
		faces = inference_video(args)
		with open(savePath, 'wb') as fil:
			pickle.dump(faces, fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))
	return faces

def scene_detect_proxy(args):
	savePath = os.path.join(args.pyworkPath, 'scene.pckl')
	if os.path.isfile(savePath):
		with open(savePath, 'rb') as fil:
			scene = pickle.load(fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene already existed in %s \r\n" %(args.pyworkPath))
	else:
		scene = scene_detect(args)
		with open(savePath, 'wb') as fil:
			pickle.dump(scene, fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))
	return scene

def thread_target(target_function, args, result, index):
    result[index] = target_function(args)

if __name__ == '__main__':
	warnings.filterwarnings("ignore")

	parser = argparse.ArgumentParser(description = "Columbia ASD Evaluation")

	parser.add_argument('--videoName',             type=str, default="col",   help='Demo video name')
	parser.add_argument('--videoFolder',           type=str, default="colDataPath",  help='Path for inputs, tmps and outputs')
	parser.add_argument('--pretrainModel',         type=str, default="weight/pretrain_AVA_CVPR.model",   help='Path for the pretrained model')

	parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
	parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
	parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
	parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
	parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
	parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

	parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
	parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

	parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columbia dataset')
	parser.add_argument('--colSavePath',           type=str, default="/colDataPath",  help='Path for inputs, tmps and outputs')

	args = parser.parse_args()


	args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
	args.savePath = os.path.join(args.videoFolder, args.videoName)
	# Initialization 
	args.pyaviPath = os.path.join(args.savePath, 'pyavi')
	args.pyframesPath = os.path.join(args.savePath, 'pyframes')
	args.pyworkPath = os.path.join(args.savePath, 'pywork')
	args.pycropPath = os.path.join(args.savePath, 'pycrop')
	# if os.path.exists(args.savePath):
	# 	rmtree(args.savePath)
	os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
	os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
	os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
	os.makedirs(args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process

	# # Extract video
	args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
	# If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
	if os.path.isfile(args.videoFilePath):
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Video savalready existede in %s \r\n" %(args.videoFilePath))
	else:
		if args.duration == 0:
			command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
				(args.videoPath, args.nDataLoaderThread, args.videoFilePath))
		else:
			command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
				(args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
		subprocess.call(command, shell=True, stdout=None)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args.videoFilePath))
	
	# Extract audio
	args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
	if os.path.isfile(args.audioFilePath):
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Audio already existed in %s \r\n" %(args.audioFilePath))
	else:
		command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
			(args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
		subprocess.call(command, shell=True, stdout=None)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))

	# Extract the video frames
	if len(os.listdir(args.pyframesPath)):
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Frames already existed in %s \r\n" %(args.pyframesPath))
	else:
		command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
			(args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg'))) 
		subprocess.call(command, shell=True, stdout=None)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(args.pyframesPath))

	scene_face = [None, None]

	# 创建两个线程，分别运行两个函数
	thread1 = threading.Thread(target=thread_target, args=(scene_detect_proxy, args, scene_face, 0))
	thread2 = threading.Thread(target=thread_target, args=(inference_video_proxy, args, scene_face, 1))
	# 启动线程
	thread1.start()
	thread2.start()
	# 等待线程完成
	thread1.join()
	thread2.join()
	scene = scene_face[0]
	faces = scene_face[1]
	# # Scene detection for the video frames
	# scene = scene_detect_proxy(args)

	# # Face detection for the video frames
	# faces = inference_video_proxy(args)

	# Face tracking
	allTracks, vidTracks = [], []
	for shot in scene:
		if shot[1].frame_num - shot[0].frame_num >= args.minTrack: # Discard the shot frames less than minTrack frames
			allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

	# Face clips cropping
	savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
	if os.path.isfile(savePath):
		with open(savePath, 'rb') as fil:
			vidTracks = pickle.load(fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop already existed in %s \r\n" %(args.pycropPath))
	else:
		for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
			vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii)))
		with open(savePath, 'wb') as fil:
			pickle.dump(vidTracks, fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args.pycropPath)

	# Active Speaker Detection
	savePath = os.path.join(args.pyworkPath, 'scores.pckl')
	if os.path.isfile(savePath):
		with open(savePath, 'rb') as fil:
			scores = pickle.load(fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores already existed in %s \r\n" %(args.pyworkPath))
	else:
		files = glob.glob("%s/*.avi"%args.pycropPath)
		files.sort()
		scores = evaluate_network(files, args)
		with open(savePath, 'wb') as fil:
			pickle.dump(scores, fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

	all_scores = []
	new_scores = []
	di_scores = []
	for clips in tqdm.tqdm(scores):
		all_scores.extend(clips)
		newclip = [min(1,x+1) for x in clips]
		for iidx,item in enumerate(newclip):
			newclip[iidx] = item if item==1 else -1
		newclip[0] = -1
		newclip[-1] = -1
		newclip = mergeclip(newclip,15)
		new_scores.append(newclip)
		di_scores.extend(newclip)
	ret_score = visualization(vidTracks, new_scores, args)
	# xs1 = [i for i in range(len(all_scores))]
	# xs2 = [i for i in range(len(ret_score))]
	# plt.figure(figsize=(12, 5))
	# plt.plot(xs1, all_scores, color='olive')
	# plt.plot(xs2, ret_score, color='salmon')
	# plt.savefig(os.path.join(args.pyaviPath,'scores.png'), dpi=300, bbox_inches='tight')