import sched
from timm.scheduler.cosine_lr import CosineLRScheduler
from read_npy import create_dataloaders
import os
import sys

from TimeSformer.timesformer.models.vit_enc_dec import TimeSformerEnDe,TimeSformerEnDeConv
from pathlib import Path
import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from skimage.transform import resize

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def embed_dim_by_img(img,num_heads,emb_mult):
    emb_dim = img*emb_mult
    head_det = emb_dim%num_heads
    if head_det!=0:
        emb_dim=emb_dim-head_det+num_heads
    return emb_dim
def count_patch_size(imgsize):
    patch = imgsize**0.5
    if imgsize%patch==0:
        return patch
    else:
        while imgsize%patch!=0:
            patch = int(patch)-1
    return patch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
accumulation_steps = 1
lr_max = 0.0005
lr_min = 0.00001
epochs = 2
predict_period = 2
in_period = 2
batch_size = 1
num_heads=8
emb_mult=8
place='kara'
from_ymd=[1979, 1, 1]
to_ymd=[1980, 1,1]#[2012, 1, 1]
stride = 7
depth = 2
latent_dim = 3
resize_img = [35,30]
if resize_img is not None:
    mask = np.load(fr'D://Projects//test_cond//AAAI_code//Ice//coastline_masks//{place}_mask.npy')
    mask = resize(mask, (resize_img[0], resize_img[1]), anti_aliasing=False)
else:
    mask = np.load(fr'D://Projects//test_cond//AAAI_code//Ice//coastline_masks//{place}_mask.npy')
dataloaders, img_sizes = create_dataloaders(path_to_dir=f'D://Projects//test_cond//AAAI_code//Ice/{place}',
                                            batch_size=batch_size,
                                            in_period=in_period,
                                            predict_period=predict_period,
                                            stride=stride,
                                            test_end=None,
                                            from_ymd=from_ymd,
                                            to_ymd=to_ymd,
                                            pad=False,
                                            resize_img=resize_img)

train_len = dataloaders[0].__len__()
test_len = dataloaders[1].__len__()

if img_sizes[1]>img_sizes[0]:
    img_sizes=(img_sizes[1],img_sizes[0])
if img_sizes[1]!=img_sizes[0]:
    patch_size1 = count_patch_size(img_sizes[0]) #int(img_sizes[0]/(img_sizes[0]*2)**0.5)
    patch_size2 = count_patch_size(img_sizes[1])
    patch_size=[patch_size1,patch_size2]#int(img_sizes[1]/(img_sizes[1]*2)**0.5)
else:
    patch_size = int(img_sizes[0]/(img_sizes[0]*2)**0.5)
#shape_end = (img_sizes[0]//patch_size)**2*in_period
embed_dim = embed_dim_by_img(img_sizes[1],num_heads,emb_mult)
#Loss f-n
loss_l1 = torch.nn.L1Loss()
loss_sim = SSIM(data_range=1, size_average=True, channel=predict_period)

####################
NAME = f'predict_forword_DEC_ENC_noconv_num_depth_{latent_dim}_resize_img_{resize_img}_{depth}_heads_{num_heads}_emb_dim_{embed_dim}'#last num_heads=6
####################
# def loss_fn(x,y):
#     out = loss_l1(x,y)
#     return out
def loss_fn(x,y):
    out = loss_l1(x,y)
    return out
#Model
# model = TimeSformerEnDe(batch_size=batch_size, output_size=[img_sizes[0], img_sizes[1]], img_size=img_sizes[0],embed_dim=embed_dim,
#                     num_frames=3, attention_type='divided_space_time', pretrained_model=False, in_chans=1, out_chans=predict_period,
#                     patch_size=patch_size,num_heads=num_heads,in_periods=in_period,place=place,emb_mult=emb_mult,mask=True,depth=depth,
#                     ).to(device)
model = TimeSformerEnDeConv(batch_size=batch_size, output_size=[img_sizes[0], img_sizes[1]], img_size=img_sizes[0],embed_dim=embed_dim,
                    num_frames=3, attention_type='divided_space_time', pretrained_model=False, in_chans=1, out_chans=predict_period,
                    patch_size=patch_size,num_heads=num_heads,in_periods=in_period,place=place,emb_mult=emb_mult,mask=True,depth=depth,
                    latent_dim=latent_dim).to(device)
# #Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.98), eps=1e-9)

# optimizer_sch = CosineLRScheduler(optimizer, t_initial=train_len*epochs//2, lr_min=lr_min,
#                   warmup_t=train_len*1,cycle_limit=1.0, warmup_lr_init=lr_min, warmup_prefix=False, t_in_epochs=True,
#                   noise_range_t=None, noise_pct=0.67, noise_std=1.0,
#                   noise_seed=42, initialize=True)

optimizer_sch = CosineLRScheduler(optimizer, t_initial=25, lr_min=lr_min*10,
                  warmup_t=1,cycle_limit=1.0, warmup_lr_init=lr_min*10, warmup_prefix=False, t_in_epochs=True,
                  noise_range_t=None, noise_pct=0.67, noise_std=1.0,
                  noise_seed=42, initialize=True)
# (batch x channels x frames x height x width)
#dummy_video = torch.randn(1, 4, 452, 452)
writer = SummaryWriter(fr'writer/{NAME}_{in_period}_pred_per_{predict_period}_bs_{batch_size}_dates_{from_ymd[0]}to_{to_ymd[0]}_stride_{stride}_sigmoid_simm')
if not os.path.isdir(fr'writer/{NAME}'):
    os.mkdir(fr'writer/{NAME}')
img=0
step = 0
ep = 0
test_step=0
current_step = 0
for epoch in tqdm(range(epochs)):
    writer.add_scalar('Lr',optimizer.param_groups[0]['lr'],ep)
    ep+=1
    
    model.train()
    optimizer.zero_grad()
    #TRAIN


    for i,batch in enumerate(dataloaders[0]):
        print(i)
        X,y,x_d,y_d=batch
        step+=1
        #current_step +=1
        X = X.to(device)
        y = y.to(device)
        z = torch.ones(batch_size,1,1,img_sizes[0],img_sizes[1]).to('cuda')
        #outputs = model.predict(X,z)
        outputs = model(X,z)
        y = y.squeeze(1)
        
        loss = loss_fn(outputs,y.unsqueeze(1))/accumulation_steps
        writer.add_scalar('Loss_train',loss.item()*accumulation_steps,step)
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    optimizer_sch.step(ep)
    optimizer.zero_grad()
    # del outputs,X,y
    # torch.cuda.empty_cache()

    #TEST
    with torch.no_grad(): 
        model.eval()
        for X,y,x_d,y_d in dataloaders[1]:
            test_step+=1
            #current_step +=1
            X = X.to(device)
            y = y.to(device)
            z = torch.ones(batch_size,1,1,img_sizes[0],img_sizes[1]).to('cuda')
            #outputs = model(X,torch.cat((z,y),dim=2))
            outputs = model(X,z)
            y = y.to(device)
            
            loss =loss_fn(outputs,y)
            loss_masked =loss_fn(outputs*torch.tensor(np.float32(mask)).to(device),y)
            imgs=np.absolute(outputs.detach().cpu().numpy()-y.detach().cpu().numpy())
            #for i,b in enumerate(imgs):#every slice per batch
            #writer.add_images('Loss_masks', np.expand_dims(imgs[0],axis=1), ep)
            writer.add_images('Ground_truth', np.expand_dims(y.squeeze(1).detach().cpu().numpy()[0],axis=1), ep//40)
            writer.add_images('Predicts', np.expand_dims(outputs.squeeze(1).detach().cpu().numpy()[0],axis=1), ep//40)
            
            writer.add_scalar('Loss_test',loss.item(),test_step)
            # writer.add_text('Dates_exist','_-_'.join(['_'.join(i) for i in x_d]),step)
            # writer.add_text('Dates_predict','_-_'.join(['_'.join(i) for i in y_d]),step)
            writer.add_scalar('Masked_loss_test',loss_masked.item(),test_step)
        #writer.add_figure('matplotlib',)
    # del outputs,X,y
    # torch.cuda.empty_cache()
    # if ep%3==0:
    #     #Validation
    #     with torch.no_grad(): 
    #         model.eval()
    #         for X,y,x_d,y_d in dataloaders[1]:
    #             test_step+=1
    #             #current_step +=1
    #             X = X.to(device)
    #             y = y.to(device)
    #             z = torch.ones(batch_size,1,1,img_sizes[0],img_sizes[1]).to('cuda')
    #             outputs = model.predict(X,z)
    #             y = y.to(device)
                
    #             loss =loss_fn(outputs,y)
    #             loss_masked =loss_fn(outputs*torch.tensor(np.float32(mask)).to(device),y)
    #             imgs=np.absolute(outputs.detach().cpu().numpy()-y.detach().cpu().numpy())
    #             #for i,b in enumerate(imgs):#every slice per batch
    #             #writer.add_images('Loss_masks', np.expand_dims(imgs[0],axis=1), ep)
    #             writer.add_images('Ground_truth_val', np.expand_dims(y.squeeze(1).detach().cpu().numpy()[0],axis=1), ep)
    #             writer.add_images('Predicts_val', np.expand_dims(outputs.squeeze(1).detach().cpu().numpy()[0],axis=1), ep)
                
    #             writer.add_scalar('Loss_test_val',loss.item(),test_step)
    #             # writer.add_text('Dates_exist','_-_'.join(['_'.join(i) for i in x_d]),step)
    #             # writer.add_text('Dates_predict','_-_'.join(['_'.join(i) for i in y_d]),step)
    #             writer.add_scalar('Masked_loss_test_val',loss_masked.item(),test_step)
    #             #writer.add_figure('matplotlib',)
        torch.save(model.state_dict(), f'writer/{NAME}/ep{ep}_model_weights_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}__dates_{from_ymd[0]}to_{to_ymd[0]}_stride{stride}_sigmoid')     

torch.save(model.state_dict(), f'writer/{NAME}/ep{ep}_model_weights_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}__dates_{from_ymd[0]}to_{to_ymd[0]}_stride{stride}_sigmoid')     

# for train in dataloaders[0]:
#     predd = model(train[0].to('cuda'))
#     print(train[0].shape, train[1].shape)
