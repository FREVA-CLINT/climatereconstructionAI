import os
import torch
import sys
sys.path.append('./')

from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm
from model.PConvLSTM import PConvLSTM
from utils.featurizer import VGG16FeatureExtractor
from utils.io import load_ckpt, save_ckpt
from utils.netcdfloader import LSTMNetCDFDataLoader as NetCDFDataLoader, InfiniteSampler
from utils.evaluator import create_snapshot_image
from model.loss import InpaintingLoss
import config as cfg

cfg.set_train_args()

if not os.path.exists(cfg.snapshot_dir):
    os.makedirs('{:s}/images'.format(cfg.snapshot_dir))
    os.makedirs('{:s}/ckpt'.format(cfg.snapshot_dir))

if not os.path.exists(cfg.log_dir):
    os.makedirs(cfg.log_dir)
writer = SummaryWriter(log_dir=cfg.log_dir)

# define data set + iterator
dataset_train = NetCDFDataLoader(cfg.data_root_dir, cfg.mask_dir, 'train', cfg.data_type, cfg.lstm_steps)
dataset_val = NetCDFDataLoader(cfg.data_root_dir, cfg.mask_dir, 'val', cfg.data_type, cfg.lstm_steps)
iterator_train = iter(data.DataLoader(dataset_train, batch_size=cfg.batch_size,
                                      sampler=InfiniteSampler(len(dataset_train)),
                                      num_workers=cfg.n_threads))

# define network model
model = PConvLSTM(image_size=cfg.image_size,
                  num_enc_dec_layers=cfg.encoding_layers,
                  num_pool_layers=cfg.pooling_layers,
                  num_in_channels=1).to(cfg.device)

# define learning rate
if cfg.finetune:
    lr = cfg.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = cfg.lr

# define optimizer and loss functions
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss().to(cfg.device)

# define start point
start_iter = 0
if cfg.resume:
    start_iter = load_ckpt(
        cfg.resume, [('model', model)], cfg.device, [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, cfg.max_iter)):
    # train model
    model.train()
    image, mask, gt = [x.to(cfg.device) for x in next(iterator_train)]
    output = model(image, mask)

    # calculate loss function and apply backpropagation
    loss_dict = criterion(image[:, cfg.lstm_steps, :, :, :], mask[:, cfg.lstm_steps, :, :, :], output[:, :, :, :],
                          gt[:, cfg.lstm_steps, :, :, :])
    loss = 0.0
    for key, coef in cfg.LAMBDA_DICT_IMG_INPAINTING.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % cfg.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save checkpoint
    if (i + 1) % cfg.save_model_interval == 0 or (i + 1) == cfg.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(cfg.snapshot_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    # create snapshot image
    if (i + 1) % cfg.log_interval == 0:
        model.eval()
        create_snapshot_image(model, dataset_val, '{:s}/images/test_{:d}.jpg'.format(cfg.snapshot_dir, i + 1),
                              cfg.lstm_steps)

writer.close()
