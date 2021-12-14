import os
import shutil
import time
from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SSD, ResNet
from utils import generate_dboxes, Encoder, coco_classes, set_seed
from transform import SSDTransformer
from loss import Loss
from dataset import collate_fn, CocoDataset

import numpy as np
from tqdm.autonotebook import tqdm
from pycocotools.cocoeval import COCOeval
import GPUtil

def get_args():
    parser = ArgumentParser(description="Implementation of SSD")
    parser.add_argument("--data-path", type=str, default='/tmp/local/'+os.environ.get('SLURM_JOBID'), help="the root folder of dataset")
    parser.add_argument("--save-folder", type=str, default='/tmp/local/'+os.environ.get('SLURM_JOBID')+'/logs/trained_models',help="path to folder containing model checkpoint file")
    parser.add_argument("--log-path", type=str, default='/tmp/local/'+os.environ.get('SLURM_JOBID')+'/logs/tensorboard/SSD')

    parser.add_argument("--model", type=str, default="ssd",help="ssd-resnet50")
    parser.add_argument("--epochs", type=int, default=30, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=64, help="number of samples for each iteration")
    parser.add_argument("--multistep", nargs="*", type=int, default=[18, 25],help="epochs at which to decay learning rate")
    parser.add_argument("--lr", type=float, default=2.6e-3, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum argument for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="momentum argument for SGD optimizer")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4242)

    args = parser.parse_args()
    return args


def train(model, train_loader, epoch, writer, criterion, optimizer, scheduler):
    # switch to train mode
    model.train()
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    scheduler.step()

    end = time.time()
    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
        print('----------------'+str(i)+'-----------------')
        print(img.size(),gloc.size(),glabel.size())
        GPUtil.showUtilization()

        # measure data loading time
        data_time=time.time() - end

        if torch.cuda.is_available():
            img = img.cuda(opt.gpu, non_blocking=True)
            gloc = gloc.cuda(opt.gpu, non_blocking=True)
            glabel = glabel.cuda(opt.gpu, non_blocking=True)

        ploc, plabel = model(img)
        print(ploc.size(),plabel.size())
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        print(gloc.size())
        loss = criterion(ploc, plabel, gloc, glabel)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time=time.time() - end
        end = time.time()

        progress_bar.set_description(
            "Epoch: {} Loss: {:.5f} Data_Time: {:6.3f} Batch_Time: {:6.3f}".format(epoch + 1, loss.item(), data_time,batch_time))

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)


def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold):
    # switch to evaluate mode
    model.eval()
    detections = []
    category_ids = test_loader.dataset.coco.getCatIds()

    with torch.no_grad():
        for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):
            print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
            if torch.cuda.is_available():
                img = img.cuda(opt.gpu, non_blocking=True)
            # Get predictions
            ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()

            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,(loc_[3] - loc_[1]) * height, prob_,category_ids[label_ - 1]])

    detections = np.array(detections, dtype=np.float32)

    coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)


def main(opt):
    #if torch.cuda.is_available():
    #    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #    num_gpus = torch.distributed.get_world_size()
    #    torch.cuda.manual_seed(123)
    #else:
    #    torch.manual_seed(123)
    #    num_gpus = 1

    set_seed(opt.seed)
    num_gpus=1

    train_params = {"batch_size": opt.batch_size * num_gpus,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": opt.num_workers,
                    "collate_fn": collate_fn}

    test_params = {"batch_size": opt.batch_size * num_gpus,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": opt.num_workers,
                   "collate_fn": collate_fn}


    dboxes = generate_dboxes(model="ssd")
    model = SSD(backbone=ResNet(), num_classes=len(coco_classes))

    train_set = CocoDataset(opt.data_path, 2017, "train", SSDTransformer(dboxes, (300, 300), val=False))
    train_loader = DataLoader(train_set, **train_params)
    test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
    test_loader = DataLoader(test_set, **test_params)

    encoder = Encoder(dboxes)

    opt.lr = opt.lr * num_gpus * (opt.batch_size / 32)
    criterion = Loss(dboxes)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,weight_decay=opt.weight_decay,nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
        criterion.cuda(opt.gpu)


        #    from torch.nn.parallel import DistributedDataParallel as DDP
        # It is recommended to use DistributedDataParallel, instead of DataParallel
        # to do multi-GPU training, even if there is only a single node.
        #model = DDP(model)


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    checkpoint_path = os.path.join(opt.save_folder, "SSD.pth")

    writer = SummaryWriter(opt.log_path)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        first_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0

    print('Starting from epoch: '+str(first_epoch))
    GPUtil.showUtilization()

    for epoch in range(first_epoch, opt.epochs):
        train(model, train_loader, epoch, writer, criterion, optimizer, scheduler)
        evaluate(model, test_loader, epoch, writer, encoder, opt.nms_threshold)

        checkpoint = {"epoch": epoch,
                      "model_state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    opt = get_args()
    main(opt)
