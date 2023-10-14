EXTRACTED_MASKED_PERSON_PATH='../data/masked_players/'
EXTRACTED_PERSON_PATH='../data/players/'
MAIN_PICTURES_PATH='../data/pics'
COCO_PB='../models/person_detector/frozen_inference_graph_coco.pb'
COCO_PBTXT='../models/person_detector/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
COCO_NAMES='../models/person_detector/coco.names'
MASKED=True

SAVE_MODEL_PATH='../models/'
INPUT_SHAPE=[64,64,3]
EMBEDDING_DIM=4

LEARNING_RATE=0.01
LOSS='mae'
METRICS=['mse']

EPOCHS=10
BATCH_SIZE=32
VALIDATION_SPLIT=0.2