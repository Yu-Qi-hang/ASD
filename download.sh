ossutil cp oss://pixelai/qianyu/model/sfd_face.pth ./model/faceDetector/s3fd/
mkdir weight
ossutil cp oss://pixelai/qianyu/model/finetuning_TalkSet.model ./weight
ossutil cp oss://pixelai/qianyu/model/pretrain_AVA_CVPR.model ./weight