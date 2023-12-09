import torch
import numpy as np

def shift_torch_tensor_down(t:torch.Tensor,shift:int=1):
    return torch.cat((t[...,shift:,:],-1*torch.ones_like(t[...,:shift,:])),dim=-2)

def shift_torch_tensor_up(t:torch.Tensor,shift:int=1):
    return torch.cat((-1*torch.ones_like(t[...,:shift,:]),t[...,:-shift,:]),dim=-2)

def shift_torch_tensor_left(t:torch.Tensor,shift:int=1):
    return torch.cat((t[...,:,shift:],-1*torch.ones_like(t[...,:,:shift])),dim=-1)

def shift_torch_tensor_right(t:torch.Tensor,shift:int=1):
    return torch.cat((-1*torch.ones_like(t[...,:,:shift]),t[...,:,:-shift]),dim=-1)


def find_centroid(t:torch.Tensor):
    xcoord = np.linspace(0,45,t.shape[0])
    ycoord = np.linspace(-14.6,14.6,t.shape[1])
    indices = torch.nonzero(t == 1)
    mean_centroid =  torch.tensor(indices.float().mean(dim=0).int())
    if any(mean_centroid < 0):
        return (None,None)
    else:
        return (xcoord[mean_centroid[0].item()],ycoord[mean_centroid[1].item()])

def find_centroid_batch_index(t:torch.Tensor):
    centroid_list = []
    for i in range(t.shape[0]):
        indices = torch.nonzero(t[i,0] == 1)   
        mean_centroid = torch.tensor(indices.float().mean(dim=0).int())
        if any(mean_centroid < 0):
            centroid_list.append((torch.nan,torch.nan))
        else:
            centroid_list.append((mean_centroid[0].item(),mean_centroid[1].item()))
    #axial, radial
    return centroid_list

def find_centroid_batch(t:torch.Tensor):
    centroid_list = []
    xcoord = np.linspace(0,45,t[0,0].shape[0])
    ycoord = np.linspace(-14.6,14.6,t[0,0].shape[1])
    for i in range(t.shape[0]):
        indices = torch.nonzero(t[i,0] == 1)   
        mean_centroid = torch.tensor(indices.float().mean(dim=0).int())
        if any(mean_centroid < 0):
            centroid_list.append((torch.nan,torch.nan))
        else:
            centroid_list.append((xcoord[mean_centroid[0].item()],ycoord[mean_centroid[1].item()]))
    #axial, radial
    return centroid_list

def get_mean_rms(t:torch.Tensor):
    mask = t[0] == 1
    return t[2][mask].mean(),t[4][mask].mean()

def get_mean_mean(t:torch.Tensor):
    mask = t[0] == 1
    return t[1][mask].mean(),t[3][mask].mean()  

def get_mean_rms_batch(t:torch.Tensor):
    u_list = []
    for ib in range(t.shape[0]):
        u_list.append(get_mean_rms(t[ib]))
    return torch.tensor(u_list)[:,0],torch.tensor(u_list)[:,1]


def get_mean_mean_batch(t:torch.Tensor):
    u_list = []
    for ib in range(t.shape[0]):
        u_list.append(get_mean_mean(t[ib]))
    return torch.tensor(u_list)[:,0],torch.tensor(u_list)[:,1]

def evolve(X0,dt=4,p=50,mode='mean'):
    X1 = X0.clone()
    dx = 45.00000178813934*1e-3/X1[0].shape[0]
    dy = 14.600000344216824*1e-3*2/X1[0].shape[1] # 2x the radius of the channel
    # centroid = find_centroid(X1[0])
    if mode == 'mean':
        urms,vrms = get_mean_rms(X1)
    else:   
        print("Error: mode not implemented")
        return
    # urms = (max_urms*0.063796625 + 0.06696225)*279.2
    # vrms = (max_vrms*0.021022914 +  0.041860554)*279.2
    urms = urms*17.8+ 16.9
    vrms = vrms*13.3+ 13.4
    if urms is torch.nan:
        urms = 0.0
    if vrms is torch.nan:
        vrms = 0.0
    du = urms*torch.randn(1)*p
    dv = vrms*torch.randn(1)*p

    t= dt/500.0e3
    shift_right = int(dv*t/dx)
    shift_up = int(du*t/dy)
    if shift_right > 0:
        X1[0] = shift_torch_tensor_right(X1[0],shift_right)
    else:
        X1[0] = shift_torch_tensor_left(X1[0],-1*shift_right)
    if shift_up > 0:
        X1[0] = shift_torch_tensor_up(X1[0],shift_up)
    else:
        X1[0] = shift_torch_tensor_down(X1[0],-1*shift_up)

    return X1

def evolve_batch(X0,nt=1,dt_all=8e-6,p=0.125): 
    X1 = X0.clone()
    #TODO check axis
    dt_sde = dt_all/nt
    dx = 45.00000178813934*1e-3/X1.shape[2]
    dy = 14.600000344216824*1e-3*2/X1.shape[3] # 2x the radius of the channel
    du_all = 0
    dv_all = 0
    dt_all = dt_sde
    centroids = torch.tensor(find_centroid_batch_index(X1)) #shape nb,2, order axial, radial,index

    urms,vrms = get_mean_rms_batch(X1) #todo test this
    
    #unscale nn inputs
    urms = urms*17.8+ 16.9
    vrms = vrms*13.3+ 13.4
    

    umean,vmean = get_mean_mean_batch(X1)
    umean = umean*99.5*2
    vmean = vmean*6.21*2
    
    urms = torch.nan_to_num(urms)
    vrms = torch.nan_to_num(vrms)


    for _ in range(nt):
        du = umean + (urms*torch.randn(X1.shape[0])/p)
        dv = vmean + (vrms*torch.randn(X1.shape[0])/p)
        du_all += du
        dv_all += dv
        dt_all+= dt_sde
        #update centroids
        dxp = du*dt_sde
        dyp  = dv*dt_sde
        dcentoidx = dxp/dx 
        dcentoidy = dyp/dy 
        centroids[:,0] += dcentoidx.round().int()
        centroids[:,1] += dcentoidy.round().int()
        #cast to int
        centroids = centroids.int()
        #get new velocity
        umean = []
        vmean = []
        urms = []
        vrms = []
        for ib in range(X1.shape[0]):
            if centroids[ib,0] < 0 or centroids[ib,0] >= X1.shape[2] or centroids[ib,1] < 0 or centroids[ib,1] >= X1.shape[3]:
                umean.append(0.0)
                vmean.append(0.0)
                urms.append(0.0)
                vrms.append(0.0)
            else:
                umean.append(X1[ib,1,centroids[ib,0],centroids[ib,1]])
                vmean.append(X1[ib,3,centroids[ib,0],centroids[ib,1]])
                urms.append(X1[ib,2,centroids[ib,0],centroids[ib,1]])
                vrms.append(X1[ib,4,centroids[ib,0],centroids[ib,1]])
        umean = torch.tensor(umean)*99.5*2
        vmean = torch.tensor(vmean)*6.21*2
        urms = torch.tensor(urms)*17.8+ 16.9
        vrms = torch.tensor(vrms)*13.3+ 13.4
        umean = torch.nan_to_num(umean)
        vmean = torch.nan_to_num(vmean)
        urms = torch.nan_to_num(urms)
        vrms = torch.nan_to_num(vrms)

    shift_right = dv_all*dt_sde/dy
    shift_up = du_all*dt_sde/dx
    shift_right = shift_right.round().int()
    shift_up = shift_up.round().int()
    for i in range(X1.shape[0]):
        try:
            if shift_right[i] is torch.nan or shift_right[i]+centroids[i,1] < 0 or shift_right[i]+centroids[i,1] >= X1.shape[3]:
                shift_right[i] = 0
            if shift_up[i] is torch.nan or shift_up[i]+centroids[i,0] < 0 or shift_up[i]+centroids[i,0] >= X1.shape[2]:
                shift_up[i] = 0
            if shift_right[i] > 0:
                X1[i,0] = shift_torch_tensor_right(X1[i,0],shift_right[i].abs())
            else:
                X1[i,0] = shift_torch_tensor_left(X1[i,0],shift_right[i].abs())
            if shift_up[i] > 0:
                X1[i,0] = shift_torch_tensor_up(X1[i,0],shift_up[i].abs())
            else:
                X1[i,0] = shift_torch_tensor_down(X1[i,0],shift_up[i].abs())
        except:
            print("Shift variables: ",shift_right[i],shift_up[i])
    return X1


def forward_rollout_batch(X0,model,device='mps',correct=False):
    old_centroid = find_centroid_batch_index(X0)
    Yhat0 = model(X0.to(device))
    new_mask = Yhat0.argmax(1).float().clone()
    new_mask = (new_mask-0.5)/0.5
    new_mask = torch.clamp(new_mask,-1,1)
    X1 = X0.clone()
    X1[:,0] = new_mask
    new_centroid = find_centroid_batch_index(X1)
    #shift based on difference between old and new centroid
    if correct:
        for i in range(X1.shape[0]):
            if old_centroid[i][0] is not torch.nan and new_centroid[i][0] is not torch.nan:
                shift_up =  old_centroid[i][0] - new_centroid[i][0]
                shift_right =  old_centroid[i][1] - new_centroid[i][1]
                if shift_up > 0:
                    X1[i,0] = shift_torch_tensor_up(X1[i,0],shift_up)
                else:
                    X1[i,0] = shift_torch_tensor_down(X1[i,0],-1*shift_up)
                if shift_right > 0:
                    X1[i,0] = shift_torch_tensor_right(X1[i,0],shift_right)
                else:
                    X1[i,0] = shift_torch_tensor_left(X1[i,0],-1*shift_right)
    return X1


def get_fraction_ones_batch(X1):
    return np.clip(X1[:],0,1).sum(axis=(1,2))/(X1.shape[1]*X1.shape[2])