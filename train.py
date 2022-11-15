from __future__ import print_function

import argparse
import datetime
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data as tudata
import yaml
from tqdm import tqdm

from data import WiderFaceDetection, detection_collate, preproc
from layers.functions.prior_box import PriorBox
from layers.modules import MultiBoxLoss
from utils.dataloader1 import create_dataloader
from utils.general import init_seeds, LOGGER, print_args, increment_path, select_device, form_net, check_suffix, \
    torch_distributed_zero_first, intersect_dicts, adjust_learning_rate, check_dataset, colorstr, check_file, \
    check_yaml, smart_DDP, one_cycle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

from torch.optim import lr_scheduler


def parse_opt(known=False):
    parser = argparse.ArgumentParser(description='Training')
    # parser.add_argument('--data', default=r'D:\Users\yl3146\Downloads\widerface\train/label.txt',
    #                     help='Training dataset directory')
    parser.add_argument('--data', type=str, default=ROOT / 'param/key_point.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT / 'slim.pth', help='initial weights path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'param/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--network', default='slim', help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate (origin lr=1e-3)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    #
    parser.add_argument('--rgb_mean', default=(104, 117, 123), help='')
    parser.add_argument('--num_classes', default=2, help='')
    parser.add_argument('--img_size', type=int, default=320, help='')
    parser.add_argument('--num_gpu', type=int, default=1, help='')
    #
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--name', type=str, default='exp', help='')
    parser.add_argument('--project', default=ROOT / 'runs/train-lmk', help='save to project/name')

    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=1, help='Global training seed')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--output', default=False, help='')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def train(model, opt, cfg, device):
    save_dir, epochs, batch_size, weights, data, workers = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.data, \
        opt.num_workers

    weights = str(opt.weights)
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)

    imgsz = opt.img_size
    # Model
    check_suffix(weights, '.pth')  # check weights
    pretrained = weights.endswith('.pth')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = weights
        state_dict = torch.load(weights, map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        new_state_dict = intersect_dicts(state_dict)
        model.load_state_dict(new_state_dict, strict=False)
        LOGGER.info(f'Transferred {len(new_state_dict)}/{len(model.state_dict())} items from {weights}')  # report

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    criterion = MultiBoxLoss(opt.num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(imgsz, imgsz))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    model.train()
    epoch = 0
    print('Loading Dataset...')

    dataset = WiderFaceDetection(data, preproc(imgsz, opt.rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = epochs * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    # if args.resume_epoch > 0:
    #     start_iter = args.resume_epoch * epoch_size
    # else:
    #     start_iter = 0

    for iteration in range(0, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(tudata.DataLoader(dataset, batch_size, shuffle=True, num_workers=opt.num_workers,
                                                    collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(model.state_dict(), save_dir + cfg['name'] + '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, opt.gamma, epoch, step_index, iteration, epoch_size, opt.lr)

        # load train data
        images, targets = next(batch_iterator)
        # images = images.cuda()
        # targets = [anno.cuda() for anno in targets]
        images = images.cpu()
        targets = [anno.cpu() for anno in targets]

        # forward
        out = model(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print(
            'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
            .format(epoch, epochs, (iteration % epoch_size) + 1,
                    epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr,
                    batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(model.state_dict(), save_dir + cfg['name'] + '_Final.pth')


def train1(model, opt, cfg, device):
    save_dir, epochs, batch_size, weights, data, workers = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.data, \
        opt.num_workers

    data, opt.hyp, opt.weights, opt.project = \
        check_file(data), check_yaml(opt.hyp), str(opt.weights), str(opt.project)

    weights = str(opt.weights)
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pth', w / 'best.pth'

    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)

    hyp = opt.hyp
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    data_dict = None
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']

    imgsz = opt.img_size
    # Model
    check_suffix(weights, '.pth')  # check weights
    pretrained = weights.endswith('.pth')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = weights
        state_dict = torch.load(weights, map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        new_state_dict = intersect_dicts(state_dict)
        model.load_state_dict(new_state_dict, strict=False)
        LOGGER.info(colorstr(
            'Transferred :') + f'{len(new_state_dict)}/{len(model.state_dict())} items from {weights}')  # report

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    gs = 32
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              opt,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '))
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       opt,
                                       hyp=hyp,
                                       cache=None if opt.noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not opt.resume:
            model.half().float()  # pre-reduce anchor precision
    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)
    model.to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    compute_loss = MultiBoxLoss(opt.num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(imgsz, imgsz))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    t0 = time.time()
    start_epoch = 0
    nb = len(train_loader)
    min_loss = [10000]
    scheduler.last_epoch = start_epoch - 1  # do not move

    for epoch in range(start_epoch, epochs):
        mloss = torch.zeros(3, device=device)
        model.train()
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'cls_loss', 'lmk_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

        total_loss = 0.0
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float()
            targets = [anno.to(device) for anno in targets]

            # Forward
            pred = model(imgs)  # forward
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = compute_loss(pred, priors, targets, device)  # loss scaled by batch_size
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            if RANK != -1:
                loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            if opt.quad:
                loss *= 4.
            loss.backward()
            optimizer.step()
            s = 0
            for anno in targets:
                s += anno.shape[0]
            total_loss += loss.item()
            # Log
            if RANK in {-1, 0}:
                loss_items = torch.cat((loss_l.unsqueeze(0), loss_c.unsqueeze(0), loss_landm.unsqueeze(0))).detach()
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, s, imgs.shape[-1]))
            # end batch -------------------------------------------------------------------------------------
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        # print(lr)
        scheduler.step()
        LOGGER.info(f'Epoch: {epoch} LR {(lr[-1]):.8f}.')

        if RANK in {-1, 0}:
            if total_loss <= min_loss[-1]:
                torch.save(model.state_dict(), best)
                torch.save(model.state_dict(), last)
                min_loss[-1] = total_loss
            else:
                torch.save(model.state_dict(), last)
    LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    LOGGER.info(f'\n{epoch - start_epoch + 1} model save in {w}')


def main(opt):
    print_args(vars(opt))
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    device = select_device(opt.device, batch_size=opt.batch_size)
    net, cfg = form_net(opt, device, output=opt.output)

    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    train1(net, opt, cfg, device)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
