from torch import cuda,device

DEVICE = device('cuda:0' if cuda.is_available() else 'cpu')
DATASET_PATH = '/run/timeshift/backup/thesis/EnhanceIt/src/Datasets/DIV2K/Train'
TEST_DATAPATH = '../src/Datasets/DIV2K/Validation'
ARCNN_DATASET = '../src/Datasets/BSDS500'
BATCH_SIZE = 4
UPSCALE_FACTOR = 4
EPOCHS = 500
ARCNN_MODEL = '../src/models/ARCNN.pth'
SIZE = 128
MODEL_SAVE_PATH = '/run/timeshift/backup/thesis/EnhanceIt/src/experiment_models/SRGAN_with_blur/models/bestSRGAN.pth'
DIS_PATH = '/run/timeshift/backup/thesis/EnhanceIt/src/experiment_models/disTest.pth'
LEARNING_RATE = 10e-5
CHECKPOINT_DIR ='../src/models/testSrgan/cp/'
MODEL_X2_DIR = '../src/model_x2.pth'
MODEL_X4_DIR = '../src/model_x4.pth'
CP = '../src/models/checkpoints/testcp'
EXTRACTED_FRAMES_DIR = '../src/extracted_frames/'
ENHANCED_FRAMES_DIR = '../src/enhanced_frames/'
ENHANCED_IMG_DIR = '../src/Single_Image_Results/'
ARCNN_FRAMES_DIR = '../src/arcnn_frames/'
BICUBIC_FRAMES_DIR = '../src/bicubic_resampling/'
