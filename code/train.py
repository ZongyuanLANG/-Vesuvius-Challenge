import torch
import torch.nn as nn
from model import UnetPlusPlus,deeplab
import gc

import numpy as np
import torch.utils.data as data
import os
import PIL.Image as Image
from tqdm import tqdm
import glob
import torch.nn as nn
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter

# Ref:https://www.kaggle.com/code/chenyh368/0-16-pytorch-unet-bce-baseline/notebook

# choose ramdom patch from papyrus
class RandomOpt():
    def __init__(self):
        self.SHARED_HEIGHT = 4096  # Height to resize all papyrii
        self.BUFFER = 64  # Half-size of papyrus patches we'll use as model inputs
        self.Z_DIM = 16  # Number of slices in the z direction. Max value is 64 - Z_START
        self.Z_START = 25  # Offset of slices in the z direction
        self.DATA_DIR = "./vesuvius-challenge-ink-detection"

#resize the image
def resize(img, SHARED_HEIGHT=RandomOpt().SHARED_HEIGHT):
    current_width, current_height = img.size
    aspect_ratio = current_width / current_height
    new_width = int(SHARED_HEIGHT * aspect_ratio)
    new_size = (new_width, SHARED_HEIGHT)
    img = img.resize(new_size)
    return img

#load the mask
def load_mask(split, index, DATA_DIR=RandomOpt().DATA_DIR):
    img = Image.open(f"{DATA_DIR}/{split}/{index}/mask.png").convert('1')
    img = resize(img)
    return torch.from_numpy(np.array(img))

#load the label
def load_labels(split, index, DATA_DIR=RandomOpt().DATA_DIR):
    img = Image.open(f"{DATA_DIR}/{split}/{index}/inklabels.png")
    img = resize(img)
    return torch.from_numpy(np.array(img)).gt(0).float()

#stack all the volume
def load_volume(split, index, DATA_DIR=RandomOpt().DATA_DIR, Z_START=RandomOpt().Z_START, Z_DIM=RandomOpt().Z_DIM):
    # Load the 3d x-ray scan, one slice at a time
    z_slices_fnames = sorted(glob.glob(f"{DATA_DIR}/{split}/{index}/surface_volume/*.tif"))[Z_START:Z_START + Z_DIM]
    z_slices = []
    for z, filename in  tqdm(enumerate(z_slices_fnames)):
        img = Image.open(filename)
        img = resize(img)
        z_slice = np.array(img, dtype="float32")
        z_slices.append(torch.from_numpy(z_slice))
    return torch.stack(z_slices, dim=0)

# Random choice of patches for training
def sample_random_location(shape, BUFFER=RandomOpt().BUFFER):
    a=BUFFER
    random_train_x = (shape[0] - BUFFER - 1 - a)*torch.rand(1)+a
    random_train_y = (shape[1] - BUFFER - 1 - a)*torch.rand(1)+a
    random_train_location = torch.stack([random_train_x, random_train_y])
    return random_train_location

# Check if the patch is in the masked zone
def is_in_masked_zone(location, mask):
    return mask[location[0].long(), location[1].long()]

# Check if the patch is in the validation zone
def is_in_val_zone(location, val_location, val_zone_size, BUFFER=RandomOpt().BUFFER):
    x = location[0]
    y = location[1]
    x_match = val_location[0] - BUFFER <= x <= val_location[0] + val_zone_size[0] + BUFFER
    y_match = val_location[1] - BUFFER <= y <= val_location[1] + val_zone_size[1] + BUFFER
    return x_match and y_match

# ============= Dataset ==============
class RandomPatchLocDataset(data.Dataset):
    def __init__(self, mask, val_location, val_zone_size):
        self.mask = mask
        self.val_location = val_location
        self.val_zone_size = val_zone_size
        self.sample_random_location_train = lambda x: sample_random_location(mask.shape)
        self.is_in_mask_train = lambda x: is_in_masked_zone(x, mask)

    def is_proper_train_location(self, location):
        return not is_in_val_zone(location, self.val_location, self.val_zone_size) and self.is_in_mask_train(location)

    def __len__(self):
        return 1280

    def __getitem__(self, index):
        # Generate a random patch
        # Ignore the index
        loc = self.sample_random_location_train(0)
        while not self.is_proper_train_location(loc):
            loc = self.sample_random_location_train(0)
        return loc.int().squeeze(1)

#model parameter
class ModelOpt:
    def __init__(self):
        # self.GPU_ID = '0'
        self.Z_DIM = RandomOpt().Z_DIM
        self.BUFFER = RandomOpt().BUFFER
        self.SEED = 0
        self.BATCH_SIZE = 64
        self.LEARNING_RATE =1e-4
        self.TRAINING_EPOCH = 40
        #Deeplabv3
        # self.LOG_DIR = './Deeplabv3'
        #Unet++
        self.LOG_DIR = './Unet++'
        self.LOAD_VOLUME = [1, 2, 3]
        # Val
        self.VAL_LOC = (1300, 1000)
        self.VAL_SIZE = (300, 7000)

# ============= Model ==============
class RandomPatchModel():
    def __init__(self, opt = ModelOpt()):
        self.opt = opt
        self._setup_all()
        self.volume_list = [load_volume('train', i) for i in opt.LOAD_VOLUME]
        # Here volume: [Z_DIM, SHARED_HEIGHT, W_V1 + W_V2 + ...]
        self.volume = torch.cat(self.volume_list, dim=2)
        # Same for mask and label
        self.mask_list = [load_mask('train', i) for i in opt.LOAD_VOLUME]
        self.labels_list = [load_labels('train', i) for i in opt.LOAD_VOLUME]
        # [SHARED_HEIGHT, W_V1 + W_V2 + ...]
        self.labels = torch.cat(self.labels_list, dim=1)
        self.mask = torch.cat(self.mask_list, dim=1)

        #Unet
        # self.net = UNet(in_ch=opt.Z_DIM).to(self.device)
        #Unet++
        self.net = UnetPlusPlus().to(self.device)
        #Deeplabv3
        # self.net=deeplab(in_ch=opt.Z_DIM).to(self.device)

        #loss
        self.train_loss=[]
        self.val_loss=[]

        # Dataset
        self.loc_datast = RandomPatchLocDataset(self.mask, val_location=opt.VAL_LOC, val_zone_size=opt.VAL_SIZE)
        self.loc_loader = data.DataLoader(self.loc_datast, batch_size=opt.BATCH_SIZE)
        # Val
        self.val_loc = []
        for x in range(opt.VAL_LOC[0], opt.VAL_LOC[0] + opt.VAL_SIZE[0], opt.BUFFER):
            for y in range(opt.VAL_LOC[1], opt.VAL_LOC[1] + opt.VAL_SIZE[1], opt.BUFFER):
                if is_in_masked_zone([torch.tensor(x),torch.tensor(y)], self.mask):
                    self.val_loc.append([[x, y]])
        print(f"======> Num Patches Val: {len(self.val_loc)}")

    #ramdom seed
    def _setup_all(self):
        # random seed
        np.random.seed(self.opt.SEED)
        torch.manual_seed(self.opt.SEED)
        torch.cuda.manual_seed_all(self.opt.SEED)
        # torch
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.GPU_ID
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Log
        self.log_dir = self.opt.LOG_DIR
        self.ckpt = os.path.join(self.log_dir)

    #get subvolume and label
    def get_subvolume(self, batch_loc, volume, labels):
        # batch_loc : [batch_size, 2]
        subvolume = []
        label = []
        for l in batch_loc:
            x = l[0]
            y = l[1]
            sv = volume[:, x - self.opt.BUFFER:x + self.opt.BUFFER, y - self.opt.BUFFER:y + self.opt.BUFFER]
            sv = sv / 65535.
            subvolume.append(sv)
            if labels is not None:
                lb = labels[x - self.opt.BUFFER:x + self.opt.BUFFER, y - self.opt.BUFFER:y + self.opt.BUFFER]
                lb = lb.unsqueeze(0)
                label.append(lb)
        # [batch, Z_DIM, BUFFER, BUFFER]
        subvolume = torch.stack(subvolume)
        # [batch, 1, BUFFER, BUFFER]
        if labels is not None:
            label = torch.stack(label)
        return subvolume, label


    def train_loop(self):
        print("=====> Begin training")
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        #ADAM
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.opt.LEARNING_RATE)
        #SGD
        # self.optimizer=optim.SGD(self.net.parameters(),lr=self.opt.LEARNING_RATE,momentum=0.9,weight_decay=0.0005)
        #RMSprop
        # self.optimizer=optim.RMSprop(self.net.parameters(),lr=self.opt.LEARNING_RATE,alpha=0.99,eps=1e-08,weight_decay=0,momentum=0,centered=False)
        self.net.train()

        best_val_loss = 100
        best_val_acc = 0
        meter = AverageMeter()
        for epoch in range(self.opt.TRAINING_EPOCH):
            bar = tqdm(enumerate(self.loc_loader), total=len(self.loc_datast) / self.opt.BATCH_SIZE)
            bar.set_description_str(f"Epoch: {epoch}")
            for i, loc in bar:
                subvolume, label = self.get_subvolume(loc, self.volume, self.labels)
                loss = self._train_step(subvolume, label)
                meter.update(loss)
                bar.set_postfix_str(f"Avg loss: {np.round(meter.get_value(),3)}")
            self.train_loss.append(meter.get_value())

            val_loss, val_acc = self.validataion_loop()
            self.val_loss.append(val_loss)
            print(f"======> Val Loss:{np.round(val_loss,3)} | Val Acc:{np.round(val_acc,3)} ")
            # Save model
            if val_loss < best_val_loss and val_acc > best_val_acc:
                torch.save(self.net.state_dict(), os.path.join(self.ckpt, "best.pt"))
                print("======> Save best val model")

                best_val_loss = val_loss
                best_val_acc = val_acc



    def _train_step(self, subvolume, label):
        self.optimizer.zero_grad()
        # inputs: subvolume: [batch, Z_DIM, BUFFER, BUFFER]
        #         label: [batch, 1, BUFFER, BUFFER]
        outputs = self.net(subvolume.to(self.device))
        # outputs = self.net(subvolume.to(self.device))['out']
        # print(f"======> c_recall:{c_recall} | c_precision:{c_precision} | dice:{dice}")
        loss = self.criterion(outputs, label.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss

    def validataion_loop(self):
        meter_loss = AverageMeter()
        meter_acc = AverageMeter()
        self.net.eval()
        for loc in self.val_loc:
            subvolume, label = self.get_subvolume(loc, self.volume, self.labels)
            outputs = self.net(subvolume.to(self.device))
            # outputs = self.net(subvolume.to(self.device))['out']
            loss = self.criterion(outputs, label.to(self.device))
            meter_loss.update(loss)
            pred = torch.sigmoid(outputs) > 0.5
            meter_acc.update(
                (pred == label.to(self.device)).sum(),
                int(torch.prod(torch.tensor(label.shape)))
            )
        self.net.train()
        return meter_loss.get_value(), meter_acc.get_value()

    def load_best_ckpt(self):
        self.net.load_state_dict(torch.load(os.path.join(self.ckpt, "best.pt")))


# For the metric
class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.n = 0

    def update(self, x, n=1):
        self.sum += float(x)
        self.n += n

    def reset(self):
        self.sum = 0
        self.n = 0

    def get_value(self):
        if self.n:
            return self.sum / self.n
        return 0

def fbeta_score(preds, targets, threshold, beta=0.5, smooth=1e-5):
    preds_t = torch.where(preds > threshold, 1.0, 0.0).float()
    y_true_count = targets.sum()

    ctp = preds_t[targets == 1].sum()
    cfp = preds_t[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return c_recall, c_precision, dice

# Define model
model = RandomPatchModel()
# print(np.array(model.volume_list).shape)
# print(model.volume.shape)
# print(model.net)

#calculate the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {count_parameters(model.net):,} trainable parameters")
#calculate training time
import time
start_time = time.time()
# Training
model.train_loop()
# print("--- %s seconds ---" % (time.time() - start_time))

#plot loss
import matplotlib.pyplot as plt
plt.title('Training Loss and Validation Loss')
plt.plot(model.train_loss, label='train')
plt.plot(model.val_loss, label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid()
# plt.savefig('./Deeplabv3/loss.png')
plt.savefig('Adam_loss.png')
plt.show()



import torch

# Load the best model
model.load_best_ckpt()
# model.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
# loss, acc = model.validataion_loop()
# model.net.eval()
# print(f"Val loss: {np.round(loss,3)} | Val acc: {np.round(acc, 3)}")

from tqdm import tqdm
def compute_predictions_map(split, index):
    print(f"======> Load data for {split}/{index}")
    test_volume = load_volume(split=split, index=index)
    test_mask = load_mask(split=split, index=index)
    print(f"======> Volume shape: {test_volume.shape}")
    test_locations = []
    BUFFER = model.opt.BUFFER
    stride = BUFFER // 2

    for x in range(BUFFER, test_volume.shape[1] - BUFFER, stride):
        for y in range(BUFFER, test_volume.shape[2] - BUFFER, stride):
            if is_in_masked_zone([torch.tensor(x),torch.tensor(y)], test_mask):
                test_locations.append((x, y))
    print(f"======> {len(test_locations)} test locations (after filtering by mask)")

    predictions_map = torch.zeros((1, 1, test_volume.shape[1], test_volume.shape[2]))
    predictions_map_counts = torch.zeros((1, 1, test_volume.shape[1], test_volume.shape[2]))
    print(f"======> Compute predictions")

    with torch.no_grad():
        bar = tqdm(test_locations)
        for loc in bar:
            subvolume, label = model.get_subvolume([loc], test_volume, None)
            # print(subvolume.shape)
            # print(np.array(label).shape)
            outputs = model.net(subvolume.to(model.device))
            # outputs = model.net(subvolume.to(model.device))['out']
            pred = torch.sigmoid(outputs)
            # print(loc, (pred > 0.5).sum())
            # Here a single location may be with multiple result
            predictions_map[:, :, loc[0] - BUFFER : loc[0] + BUFFER, loc[1] - BUFFER : loc[1] + BUFFER] += pred.cpu()
            predictions_map_counts[:, :, loc[0] - BUFFER : loc[0] + BUFFER, loc[1] - BUFFER : loc[1] + BUFFER] += 1

    # print(predictions_map_b[:,:, 2500, 1000])
    # print(predictions_map_counts[:,:, 2500, 1000])
    predictions_map /= (predictions_map_counts + 1e-7)
    return predictions_map

predictions_map_a = compute_predictions_map(split="test", index="a")
predictions_map_b = compute_predictions_map(split="test", index="b")
predictions_map_1=compute_predictions_map(split="train", index="1")
#
#
import matplotlib.pyplot as plt

#Deeplabv3+
# Threshold = 0.05
# Unet++
Threshold = 0.3


plt.title(f'Test a Threshold={Threshold}')
plt.imshow(predictions_map_a.squeeze() > Threshold, cmap='gray')
plt.axis('off')
plt.savefig('./Deeplabv3/a.png')
# plt.show()
plt.title(f'Test b Threshold={Threshold}')
plt.imshow(predictions_map_b.squeeze() > Threshold, cmap='gray')
plt.axis('off')
plt.savefig('./Deeplabv3/b.png')
# plt.show()
plt.title(f'Train 1 Threshold={Threshold}')
plt.imshow(predictions_map_1.squeeze() > Threshold, cmap='gray')
plt.axis('off')
plt.savefig('./Deeplabv3/1.png')
plt.show()
#

c_recall, c_precision, dice = fbeta_score(predictions_map_1.squeeze(), load_mask(split="train", index="1"), threshold=0.15)
print(f"Train 1: c_recall: {c_recall}, c_precision: {c_precision}, dice: {dice}")



