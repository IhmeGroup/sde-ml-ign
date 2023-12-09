# %%
import os
import numpy as np
import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from pytorch_lightning.strategies import DDPStrategy
import time
import sys
sys.path.append('../')
from models.unet import UNet,LitModel
from common.data import my_read_csv,ProtoDataset,paths_from_cases



from argparse import ArgumentParser

parser = ArgumentParser()

# Trainer arguments
parser.add_argument("--shrink", type=int, default=1)
parser.add_argument("--dt", type=int, default=4)
parser.add_argument("--classes", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--nepochs", type=int, default=600)
parser.add_argument("--ckpt_name", type=str, default="last.ckpt")
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--port", type=int, default=55001)
parser.add_argument("--tune",action='store_true')
parser.add_argument("--fast_dev_run", type=int, default=0)
parser.add_argument("--profiler",type=str,default=None)
parser.add_argument("--accumulate_grad_batches",type=int,default=1)
parser.add_argument("--log_gpu_memory",action='store_true')
parser.add_argument("--timeit", action='store_true')
parser.add_argument("--gpu",type=int,default=4)
parser.add_argument("--save_period",type=int,default=5)
parser.add_argument("--max_time",type=str,default="00:11:58:00")
parser.add_argument("--precision",type=int,default=16)
parser.add_argument("--rank_file",type=str,default='../rank_file.txt')
parser.add_argument("--mode",type=str,default='pbatch')
parser.add_argument("--case_name",type=str,default='tmp')
parser.add_argument("--lr_sched_factor",type=int,default=1,help='lr scheduler multiplier')
parser.add_argument("--train_path",type=str,default='./trainset_sheet.csv')
parser.add_argument("--val_path",type=str,default='./valset_sheet.csv')
parser.add_argument("--with_temporal",action='store_true')

parser.set_defaults(tune=False)
parser.set_defaults(timeit=True)
parser.set_defaults(log_gpu_memory=False)
parser.set_defaults(with_temporal=False)


# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()
shrink = args.shrink
dt = args.dt
classes = args.classes
learning_rate = args.learning_rate
num_workers = args.num_workers
batch_size = args.batch_size
nepochs = args.nepochs
ckpt_name = args.ckpt_name
num_nodes = args.num_nodes
seed = args.seed
port = args.port
tune = args.tune
fast_dev_run = args.fast_dev_run
profiler = args.profiler
accumulate_grad_batches = args.accumulate_grad_batches
log_gpu_memory = args.log_gpu_memory
timeit = args.timeit
gpu = args.gpu
save_period = args.save_period
max_time = args.max_time
precision = args.precision
rank_file = args.rank_file
mode = args.mode
case_name = args.case_name
lr_sched_factor = args.lr_sched_factor
train_path = args.train_path
val_path = args.val_path
with_temporal = args.with_temporal


os.environ["WORLD_SIZE"] = str(num_nodes*gpu)
world_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
info = np.loadtxt(rank_file,delimiter=',',dtype=str)
master_addr = info[0]
port = info[1]
os.environ['MASTER_PORT'] = str(port)
os.environ["NODE_RANK"] = str(world_rank//gpu)
os.environ["LOCAL_RANK"] = str(world_rank%gpu)
os.environ['MASTER_ADDR'] = master_addr
print("WORLD_RANK: ",world_rank,", NODE_RANK: ",os.environ["NODE_RANK"],", LOCAL_RANK: ",os.environ["LOCAL_RANK"], ", WORLD_SIZE: ",os.environ["WORLD_SIZE"],
      ", MASTER_PORT: ",os.environ['MASTER_PORT'],", MASTER_ADDR: ",os.environ['MASTER_ADDR'])

detect_anomaly = False
if mode == 'pdebug':
    nepochs = save_period*2
    log_gpu_memory = True
    profiler = 'simple'
    detect_anomaly = True

log_dir = './logs/'+mode+'/'+case_name
checkpoint_dir = '../ckpt/'+mode+'/'+case_name+'/'

if __name__ == '__main__':
    pl.seed_everything(seed)
    # %%
    train_dict = my_read_csv(train_path)
    train_cases = train_dict['Case #']

    val_dict = my_read_csv(val_path)
    val_cases = val_dict['Case #']


    # %%
    train_paths = paths_from_cases(train_cases)
    val_paths = paths_from_cases(val_cases)

    print('# Train: ',len(train_paths))
    print('# Val: ',len(val_paths))



    training_transformations = T.Compose([T.Resize((256,160))])
    validation_transformations = T.Compose([T.Resize((256,160))])
    target_transformations = T.Compose([T.Resize((256,160),interpolation=0)])
    if with_temporal:
        train_ds = ProtoDatasetWithNt(path_list=train_paths,dt=dt,transform=training_transformations,mode='train',target_transform=target_transformations,classes=classes)
        val_ds = ProtoDatasetWithNt(path_list=val_paths,dt=dt,transform=validation_transformations,mode='val',target_transform=target_transformations,classes=classes)
    else:
        train_ds = ProtoDataset(path_list=train_paths,dt=dt,transform=training_transformations,mode='train',target_transform=target_transformations,classes=classes)
        val_ds = ProtoDataset(path_list=val_paths,dt=dt,transform=validation_transformations,mode='val',target_transform=target_transformations,classes=classes)
    train_ds_len = len(train_ds)
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size,
                            shuffle=True,num_workers=num_workers,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds,batch_size=batch_size,
                            shuffle=False,num_workers=num_workers,pin_memory=True)



    # %%
    if with_temporal:
        model = UNet(10,classes,firstC=64)
    else:
        model = UNet(9,classes,firstC=64)
    model = LitModel(model,classes=classes,lr=learning_rate)


    # %%
    #set up logger
    csv_logger = pl.loggers.CSVLogger(save_dir=log_dir, name="csv_logs",
                                        flush_logs_every_n_steps=save_period*train_ds_len//(batch_size*num_nodes*gpu))
    my_logger = [csv_logger]

    plugin_list = [pl.plugins.environments.LightningEnvironment()]   

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=3, monitor="val_loss", mode="min",save_last=True, filename='model-{epoch:02d}-{val_loss:.4f}')
    acc_checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=3, monitor="val_acc", mode="max",save_last=False, filename='model-{epoch:02d}-{val_acc:.4f}')
    f1_checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=3, monitor="val_f1", mode="max",save_last=False, filename='model-{epoch:02d}-{val_f1:.4f}')
    recall_checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=3, monitor="val_recall", mode="max",save_last=False, filename='model-{epoch:02d}-{val_recall:.4f}')
    precision_checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=3, monitor="val_precision", mode="max",save_last=False, filename='model-{epoch:02d}-{val_precision:.4f}')
    callback_list = [checkpoint_callback,acc_checkpoint_callback,f1_checkpoint_callback,recall_checkpoint_callback,precision_checkpoint_callback]



    #Train!
    trainer = pl.Trainer(logger=my_logger,max_epochs=nepochs,
                                max_time=max_time,
                            accelerator="gpu", devices=gpu,strategy=DDPStrategy(find_unused_parameters=False),
                            num_nodes=num_nodes, sync_batchnorm=True, precision=precision,
                            log_every_n_steps = 1,detect_anomaly =detect_anomaly, 
                            accumulate_grad_batches=accumulate_grad_batches,
                            plugins=plugin_list,callbacks=callback_list,
                            check_val_every_n_epoch=save_period,
                            fast_dev_run = fast_dev_run, profiler=profiler)
    if world_rank == 0:
        print('Training...')

    if world_rank == 0:
        if timeit:
            start_time = time.time()
    if os.path.exists(checkpoint_dir+ckpt_name):
        trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=val_loader,ckpt_path=checkpoint_dir+ckpt_name)
    else:
        trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=val_loader)

    if timeit:
        end_time = time.time()
        if world_rank == 0:    
            print('Training time: ' + str((end_time-start_time)/60))







