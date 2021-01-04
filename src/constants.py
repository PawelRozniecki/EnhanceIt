from torch import cuda,device

DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DATASET_PATH = '/home/pawel/PycharmProjects/EnhanceIt/src/Datasets/DIV2K/Train'
TEST_DATAPATH = '/home/pawel/PycharmProjects/EnhanceIt/src/Datasets/DIV2K/Validation'
ARCNN_DATASET = '/home/pawel/PycharmProjects/EnhanceIt/src/Datasets/BSDS500'
BATCH_SIZE = 8
UPSCALE_FACTOR = 2
EPOCHS = 500

SIZE = 32
MODEL_SAVE_PATH = '/home/pawel/PycharmProjects/EnhanceIt/src/models/checkpoints/cp424.pth'
DIS_PATH = '/home/pawel/PycharmProjects/EnhanceIt/src/models/checkpoints/dis32px.pth'
LEARNING_RATE = 10e-5
CHECKPOINT_DIR ='/home/pawel/PycharmProjects/EnhanceIt/src/models/checkpoints'
CP  = '/home/pawel/PycharmProjects/EnhanceIt/src/models/checkpoints/testcp'
EXTRACTED_FRAMES_DIR = '/home/pawel/PycharmProjects/EnhanceIt/src/extracted_frames/'
ENHANCED_FRAMES_DIR = '/home/pawel/PycharmProjects/EnhanceIt/src/enhanced_frames/'
ENHANCED_IMG_DIR = '/home/pawel/PycharmProjects/EnhanceIt/src/Single_Image_Results/'
ARCNN_FRAMES_DIR = '/home/pawel/PycharmProjects/EnhanceIt/src/arcnn_frames/'
BICUBIC_FRAMES_DIR = '/home/pawel/PycharmProjects/EnhanceIt/src/bicubic_resampling/'
