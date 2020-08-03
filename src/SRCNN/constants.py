from torch import cuda,device

DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DATASET_PATH = '/Users/pingwin/PycharmProjects/EnhanceIt/src/dataset/T91'
BATCH_SIZE = 7
NO_WORKERS = 1
UPSCALE_FACTOR = 20
EPOCHS = 20
MODEL_SAVE_PATH = '/Users/pingwin/PycharmProjects/EnhanceIt/src/model.pth'