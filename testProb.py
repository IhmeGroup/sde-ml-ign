
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
import sys
# import matplotlib.pyplot as plt
import time

sys.path.append('../')
from models.unet import UNet
from common.data import my_read_csv,ProtoDataset,get_flow_stats,single_frame_from_cases,shifted_paths
from common.sde import forward_rollout_batch, get_fraction_ones_batch,evolve_batch,get_fraction_ones_batch,find_centroid_batch
import random

from argparse import ArgumentParser

parser = ArgumentParser()

# Trainer arguments
parser.add_argument("--dt", type=int, default=4)
parser.add_argument("--nt_sde", type=int, default=1)
parser.add_argument("--dt_allsde", type=float, default=8e-6)
parser.add_argument("--classes", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--ckpt_path", type=str, default="../ckpt/pbatch/unet_default/model-epoch=45-val_acc=0.9976.ckpt")
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=20)
parser.add_argument("--tune",action='store_true')
parser.add_argument("--profiler",type=str,default=None)
parser.add_argument("--timeit", action='store_true')
parser.add_argument("--gpu",type=int,default=4)
parser.add_argument("--precision",type=int,default=16)
parser.add_argument("--case_name",type=str,default='tmp')
parser.add_argument("--test_path",type=str,default='./all_sheet.csv')
parser.add_argument("--nseed",type=int,default=10)
parser.add_argument("--start_seed",type=int,default=0)
parser.add_argument("--output_path",type=str,default='./outputs/')
parser.add_argument("--prefactor",type=int,default=10)
parser.add_argument("--data_path",type=str,default='../ign_bin/')
parser.add_argument("--ori_shape",type=str,default='217')
parser.add_argument("--stats_mode",type=str,default='default')
parser.set_defaults(tune=False)
parser.set_defaults(timeit=True)
parser.set_defaults(evolve_mean=False)


# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()
dt = args.dt
nt_sde = args.nt_sde
dt_allsde = args.dt_allsde
classes = args.classes
batch_size = args.batch_size
checkpoint_path = args.ckpt_path
num_nodes = args.num_nodes
tune = args.tune
profiler = args.profiler
timeit = args.timeit
gpu = args.gpu
precision = args.precision
case_name = args.case_name
test_path = args.test_path
nseed = args.nseed
start_seed = args.start_seed
output_path = args.output_path
num_workers = args.num_workers
prefactor = args.prefactor
data_path = args.data_path
ori_shape = args.ori_shape
stats_mode = args.stats_mode


if __name__ == '__main__':
    dt_sde = dt_allsde/nt_sde
    cfl = 400*dt_sde/min(45e-3/256,14.6e-3/160)
    print("CFL: ", cfl)

    inert_les = get_flow_stats('../donatella_inert/')
    model = UNet(9,classes,firstC=64)
    state_dict= torch.load(checkpoint_path,map_location=torch.device('cpu'))['state_dict']
    new_state_dict = {}
    for k in state_dict.keys():
        new_state_dict[k[6:]] = state_dict[k]
    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to('cuda')

    test_dict = my_read_csv(test_path)
    test_cases = test_dict['Case #']

    # %%
    if 'init' in data_path:
        test_paths = shifted_paths(ori_shape=ori_shape)
    else:
        test_paths = single_frame_from_cases(test_cases,data_path=data_path)

    validation_transformations = T.Compose([T.Resize((256,160))])
    target_transformations = T.Compose([T.Resize((256,160),interpolation=0)])

    test_ds = ProtoDataset(path_list=test_paths,
                        transform=validation_transformations,mode='prob',
                        target_transform=target_transformations,
                        classes=classes,data_path=data_path,sim_path='../donatella_inert/',stats_mode=stats_mode)

    test_loader = torch.utils.data.DataLoader(test_ds,batch_size=batch_size,
                            shuffle=False,num_workers=num_workers)

    print('Testing...')

    if timeit:
        start_time = time.time()

    count = 0
    test_iterator = iter(test_loader)
    results_list= []
    while True:
        try:
            print("Count: ",count)
            X0,_ = next(test_iterator)
            max_iter= (256-8)//dt 
            seed_list = list(range(start_seed,start_seed+nseed))
            
            
            for seed in seed_list:
                print("Seed: ",seed)
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)

                pred_0d = []
                X = X0.clone()
                centroid = []
                pred_0d.append(get_fraction_ones_batch(X[:,0].detach().numpy()))
                centroid.append(find_centroid_batch(X))
                for i in range(max_iter):
                    #evolve chemistry
                    X1pred = forward_rollout_batch(X,model,'cuda')
                    #evolve velocity
                    X = evolve_batch(X1pred, nt=nt_sde,dt_all=dt_allsde,p=prefactor)
                    pred_0d.append(get_fraction_ones_batch(X[:,0].detach().numpy()))
                    centroid.append(find_centroid_batch(X))
                pred_no_sde = []
                centroid_no_sde = []
                X1 = X0.clone()
                pred_no_sde.append(get_fraction_ones_batch(X1[:,0].detach().numpy()))
                centroid_no_sde.append(find_centroid_batch(X1))
                for i in range(max_iter):
                    X1 = forward_rollout_batch(X1,model,'cuda')
                    pred_no_sde.append(get_fraction_ones_batch(X1[:,0].detach().numpy()))
                    centroid_no_sde.append(find_centroid_batch(X1))
                pred_0d = np.array(pred_0d)
                pred_no_sde = np.array(pred_no_sde)
                #stack list of 2d arrays along batch dimension
                centroid = np.stack(centroid,axis=-1)
                centroid_no_sde = np.stack(centroid_no_sde,axis=-1)
                for ib in range(X0.shape[0]):
                    results_list.append({'id':test_paths[count+ib][0],
                                         'init':test_paths[count+ib][1],
                                         'seed':seed,
                                         'pred':pred_0d[:,ib],
                                         'pred_no_sde':pred_no_sde[:,ib],
                                         'centroid':centroid[ib,:,:],
                                         'centroid_no_sde':centroid_no_sde[ib,:,:]
                    })
            count += X0.shape[0]
        except StopIteration:
            break
    if 'init' in data_path:
        initcase = str(ori_shape)
    else:
        initcase = ''
    results_df = pd.DataFrame(results_list)
    results_df.to_json(output_path+initcase+'ign_prob_seed'+str(nseed)+'_prefactor'+str(prefactor)+'CFL'+str(int(cfl))
                       +'seed_'+str(seed_list[0])+'-'+str(seed_list[-1])
                       +'stats_'+stats_mode+'.json')

    if timeit:
        end_time = time.time()
 
    print('Test time: ' + str((end_time-start_time)/60))







