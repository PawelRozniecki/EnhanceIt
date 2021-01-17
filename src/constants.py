from torch import cuda,device

DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DATASET_PATH = '../src/Datasets/DIV2K/Train'
TEST_DATAPATH = '../src/Datasets/DIV2K/Validation'
ARCNN_DATASET = '../src/Datasets/BSDS500'
BATCH_SIZE = 24
UPSCALE_FACTOR = 4
EPOCHS = 200
ARCNN_MODEL = '../src/BestARCNN.pth'
SIZE = 96
MODEL_SAVE_PATH = '../src/models/testSrgan/testSrgan.pth'
DIS_PATH = '../src/models/testSrgan/disTest.pth'
LEARNING_RATE = 10e-5
CHECKPOINT_DIR ='../src/models/testSrgan/cp/'
MODEL_X2_DIR = '../src/model_x2.pth'
MODEL_X4_DIR = '../src/model_x4.pth'
CP  = '../src/models/checkpoints/testcp'
EXTRACTED_FRAMES_DIR = '../src/extracted_frames/'
ENHANCED_FRAMES_DIR = '../src/enhanced_frames/'
ENHANCED_IMG_DIR = '../src/Single_Image_Results/'
ARCNN_FRAMES_DIR = '../src/arcnn_frames/'
BICUBIC_FRAMES_DIR = '../src/bicubic_resampling/'
