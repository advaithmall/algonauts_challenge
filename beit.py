import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
from transformers import AutoImageProcessor, BeitModel
from scipy.ndimage import zoom
from skimage.measure import block_reduce
print("Done Importing...")

data_dir = ""
parent_submission_dir = ""

parent_submission_dir = '../algonauts_2023_challenge_submission'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ", device)

subj = 2


class argObj:
  def __init__(self, data_dir, parent_submission_dir, subj):

    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
    self.parent_submission_dir = parent_submission_dir
    self.subject_submission_dir = os.path.join(self.parent_submission_dir,
                                               'subj'+self.subj)

    # Create the submission directory if not existing
    if not os.path.isdir(self.subject_submission_dir):
        os.makedirs(self.subject_submission_dir)


args = argObj(data_dir, parent_submission_dir, subj)


train_img_dir = os.path.join(
    args.data_dir, 'training_split', 'training_images')
test_img_dir = os.path.join(args.data_dir, 'test_split', 'test_images')

# Create lists will all training and test image file names, sorted
train_img_list = os.listdir(train_img_dir)
train_img_list.sort()
test_img_list = os.listdir(test_img_dir)
test_img_list.sort()
print('Training images: ' + str(len(train_img_list)))
print('Test images: ' + str(len(test_img_list)))


rand_seed = 5  # @param
np.random.seed(rand_seed)

# Calculate how many stimulus images correspond to 90% of the training data
num_train = int(np.round(len(train_img_list) / 100 * 90))
# Shuffle all training stimulus images
idxs = np.arange(len(train_img_list))
np.random.shuffle(idxs)
# Assign 90% of the shuffled stimulus images to the training partition,
# and 10% to the test partition
idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
# No need to shuffle or split the test stimulus images
idxs_test = np.arange(len(test_img_list))

print('Training stimulus images: ' + format(len(idxs_train)))
print('\nValidation stimulus images: ' + format(len(idxs_val)))
print('\nTest stimulus images: ' + format(len(idxs_test)))

image_processor = AutoImageProcessor.from_pretrained(
    "microsoft/beit-base-patch16-224-pt22k")
model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")


class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs):
        self.imgs_paths = np.array(imgs_paths)[idxs]

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        #img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        #if self.transform:
        #    img = self.transform(img).to(device)
        image = Image.open(img_path)
        inputs = image_processor(image, return_tensors="pt")
        #print(type(inputs))
        return inputs


train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

dataset_l = ImageDataset(train_imgs_paths, idxs_train)
dataset_v = ImageDataset(train_imgs_paths, idxs_val)
dataset_t = ImageDataset(test_imgs_paths, idxs_test)
fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

lh_fmri_train = lh_fmri[idxs_train]
lh_fmri_val = lh_fmri[idxs_val]
rh_fmri_train = rh_fmri[idxs_train]
rh_fmri_val = rh_fmri[idxs_val]

print("Starting feature extraction for training data...")
feature_list = []
for image in tqdm(dataset_l, total=len(dataset_l), desc="Extracting Features..."):
    with torch.no_grad():
      outputs = model(**image)
    last_hidden_state = outputs.last_hidden_state
    feature_list.append(last_hidden_state)
features = feature_list
for i in range(len(features)):
  features[i] = features[i].reshape(1, -1)
features = np.array(features)
features = features.reshape(features.shape[0], -1)

print("Downsampling training features...")
downsample_factor = 4
downsampled_features = block_reduce(features, (1, downsample_factor), np.mean)
features = downsampled_features

print("Starting feature extraction for validation data...")
feature_list = []
for image in tqdm(dataset_v, total=len(dataset_v), desc="Extracting Features..."):
    with torch.no_grad():
      outputs = model(**image)
    last_hidden_state = outputs.last_hidden_state
    feature_list.append(last_hidden_state)
features_val = feature_list
for i in range(len(features_val)):
    features_val[i] = features_val[i].reshape(1, -1)
features_val = np.array(features_val)
features_val = features_val.reshape(features_val.shape[0], -1)

print("Downsampling validation features...")
downsample_factor = 4
downsampled_features = block_reduce(
    features_val, (1, downsample_factor), np.mean)
features_val = downsampled_features

print("Starting feature extraction for test data...")
feature_list = []
for image in tqdm(dataset_t, total=len(dataset_t), desc="Extracting Features..."):
    with torch.no_grad():
      outputs = model(**image)
    last_hidden_state = outputs.last_hidden_state
    feature_list.append(last_hidden_state)
features_test = feature_list
for i in range(len(features_test)):
    features_test[i] = features_test[i].reshape(1, -1)
features_test = np.array(features_test)
features_test = features_test.reshape(features_test.shape[0], -1)

print("Downsampling test features...")
downsample_factor = 4
downsampled_features = block_reduce(
    features_test, (1, downsample_factor), np.mean)
features_test = downsampled_features

print("Doing Regression...")
reg_lh = LinearRegression().fit(features, lh_fmri_train)
reg_rh = LinearRegression().fit(features, rh_fmri_train)

print("Regression Done...")

lh_fmri_val_pred = reg_lh.predict(features_val)
rh_fmri_val_pred = reg_rh.predict(features_val)
lh_fmri_test_pred = reg_lh.predict(features_test)
rh_fmri_test_pred = reg_rh.predict(features_test)
#delete all useless stuff and models to save memory
print("Freeing Memory...")
del features
del features_val
del features_test
del dataset_l
del dataset_v
del dataset_t
del feature_list
del model
del image_processor
del reg_lh
del reg_rh

print("Saving Predictions...")
lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)
torch.save(lh_fmri_val_pred, "./subj2/lh_fmri_val_pred.pt")
torch.save(rh_fmri_val_pred, "./subj2/rh_fmri_val_pred.pt")
torch.save(lh_fmri_test_pred, "./subj2/lh_fmri_test_pred.pt")
torch.save(rh_fmri_test_pred, "./subj2/rh_fmri_test_pred.pt")
torch.save(rh_fmri_val, "./subj2/rh_fmri_val.pt")
torch.save(lh_fmri_val, "./subj2/lh_fmri_val.pt")
