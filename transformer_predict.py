from multiprocessing import reduction
import sched
from timm.scheduler.cosine_lr import CosineLRScheduler
from read_npy import create_dataloaders
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/TimeSformer')
from timesformer.models.vit import VisionTransformer_conv_aug,TimeSformer
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


device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
accumulation_steps = 1
lr_max = 0.0005
lr_min = 0.000001
epochs = 1
predict_period = 52
in_period = 104
batch_size = 1
num_heads=12
emb_mult=5
place='kara'
PARALLEL = False
from_ymd_train=[1979, 1, 1]
to_ymd_train=[2012,1,1]
from_ymd_test=[2012,1,2]
to_ymd_test=[2024, 1, 1]
load_predtrain = True
depth = 11
LOSS = 'MAE'
predtreain_path = r'D:\Projects\test_cond\AAAI_code\Ice\OLD_CONV(TimeSformer)_aug_simg_predtrain_True_LOSS_MAE_depth_11_num_heads_12_emb_dim_600\model_weights_in_per_104_pred_per_52_bs_1__dates_1979to_2012_stride7_sigmoid_ep80'



#[2012,1,1] to [2012,1,1] [2020,1,1]
stride = 7
resize_img = None
if resize_img is not None:
    mask = np.load(fr'Ice\coastline_masks\{place}_mask.npy')
    mask = resize(mask, (resize_img[0], resize_img[1]), anti_aliasing=False)
else:
    mask = np.load(fr'Ice\coastline_masks\{place}_mask.npy')
# dataloader_train, img_sizes = create_dataloaders(path_to_dir=f'Ice/{place}',
#                                             batch_size=batch_size,
#                                             in_period=in_period,
#                                             predict_period=predict_period,
#                                             stride=stride,
#                                             test_end=None,
#                                             from_ymd=from_ymd_train,
#                                             to_ymd=to_ymd_train,
#                                             pad=False,
#                                             train_test_split=None,
#                                              resize_img=resize_img)
dataloader_test, img_sizes = create_dataloaders(path_to_dir=f'Ice/{place}',
                                            batch_size=1,
                                            in_period=in_period,
                                            predict_period=predict_period,
                                            stride=stride,
                                            test_end=None,
                                            from_ymd=from_ymd_test,
                                            to_ymd=to_ymd_test,
                                            pad=False,
                                            train_test_split=None,
                                             resize_img=resize_img)

#train_len = dataloader_train.__len__()
test_len = dataloader_test.__len__()

if img_sizes[1]>img_sizes[0]:
    img_sizes=(img_sizes[1],img_sizes[0])
if img_sizes[1]!=img_sizes[0]:
    patch_size1 = count_patch_size(img_sizes[0]) #int(img_sizes[0]/(img_sizes[0]*2)**0.5)
    patch_size2 = count_patch_size(img_sizes[1])
    patch_size=[patch_size1,patch_size2]#int(img_sizes[1]/(img_sizes[1]*2)**0.5)
else:
    patch_size = int(img_sizes[0]/(img_sizes[0]*2)**0.5)
embed_dim = embed_dim_by_img(img_sizes[1],num_heads,emb_mult)
dropout=0.1
attn_drop_rate=0.1
#Loss f-n
####################
NAME = f'test52_droptrue__ep80_{load_predtrain}_LOSS_{LOSS}dropout{dropout}_depth_{depth}attn_drop_rate{attn_drop_rate}_num_heads_{num_heads}_emb_dim_{embed_dim}'#last num_heads=6
####################

if LOSS=="MAE":
    loss_l1 = torch.nn.L1Loss(reduction='none')
loss_sim = SSIM(data_range=1, size_average=True, channel=predict_period)

# def loss_fn(x,y):
#     out = loss_l1(x,y)
#     return out
def loss_fn(x,y):
    out = loss_l1(x,y)# + 0.05*(1-loss_sim(x,y))
    return out
#Model

#Optimizer
if PARALLEL:
    model = TimeSformer(batch_size=batch_size, output_size=[img_sizes[0], img_sizes[1]], img_size=img_sizes[0],embed_dim=embed_dim,
                    num_frames=4, attention_type='divided_space_time', pretrained_model=False, in_chans=1, out_chans=predict_period,
                    patch_size=patch_size,num_heads=num_heads,in_periods=in_period,place=place,depth=depth,emb_mult=emb_mult)
    if load_predtrain:
        model_dict_pred_train = torch.load(predtreain_path,map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        dict_matched = [i for i,k in zip(model_dict_pred_train,model_dict) if model_dict_pred_train[i].shape==model_dict[k].shape]
        test_dict = {i:model_dict_pred_train[i] for i in dict_matched}
        model_dict.update(test_dict)
        model.load_state_dict(model_dict)
        # pretrained_dict = {k: v for k, v in model_dict_pred_train.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(pretrained_dict)
    model = torch.nn.DataParallel(model)
    model.to(device)
else:
    model = TimeSformer(batch_size=batch_size, output_size=[img_sizes[0], img_sizes[1]], img_size=img_sizes[0],embed_dim=embed_dim,
                    num_frames=4, attention_type='divided_space_time', pretrained_model=False, in_chans=1, out_chans=predict_period,
                    patch_size=patch_size,num_heads=num_heads,in_periods=in_period,place=place,depth=depth,emb_mult=emb_mult
                    ).to(device)
    if load_predtrain:
        model_dict_pred_train = torch.load(predtreain_path,map_location=torch.device(device))
        # model_dict = model.state_dict()
        # dict_matched = [i for i,k in zip(model_dict_pred_train,model_dict) if model_dict_pred_train[i].shape==model_dict[k].shape]
        # test_dict = {i:model_dict_pred_train[i] for i in dict_matched}
        # model_dict.update(test_dict)
        model.load_state_dict(model_dict_pred_train)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.98), eps=1e-9)

weight_decay = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.98), eps=1e-9)#Weight decay 
# optimizer_sch = CosineLRScheduler(optimizer, t_initial=train_len*epochs//2, lr_min=lr_min,
#                   warmup_t=train_len*1,cycle_limit=1.0, warmup_lr_init=lr_min, warmup_prefix=False, t_in_epochs=True,
#                   noise_range_t=None, noise_pct=0.67, noise_std=1.0,
#                   noise_seed=42, initialize=True)
optimizer_sch = CosineLRScheduler(optimizer, t_initial=120, lr_min=lr_min*10,
                  warmup_t=5,cycle_limit=1.0, warmup_lr_init=lr_min, warmup_prefix=False, t_in_epochs=True,
                  noise_range_t=None, noise_pct=0.67, noise_std=1.0,
                  noise_seed=42, initialize=True)
# (batch x channels x frames x height x width)
#dummy_video = torch.randn(1, 4, 452, 452)

writer = SummaryWriter(f'Ice/writer_test/{NAME}_weight_decay{weight_decay}_lr{lr_min}_{lr_max}_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}_dates_{from_ymd_train[0]}to_{to_ymd_train[0]}_stride_{stride}sigmoid_accum_gr{accumulation_steps}')
params_dict = {
    'attn_drop_rate':attn_drop_rate,
    'dropout':dropout,
    'device' :  device,
    'accumulation_steps' : accumulation_steps,
    'lr_max' : lr_max,
    'lr_min' :lr_min,
    'epochs' :epochs,
    'predict_period' :predict_period,
    'in_period' : in_period,
    'batch_size' : batch_size,
    'num_heads':num_heads,
    'emb_mult':emb_mult,
    'place':place,
    'PARALLEL' : PARALLEL,
    'from_ymd_train':from_ymd_train,
    'to_ymd_train':to_ymd_train,
    'from_ymd_test':from_ymd_test,
    'to_ymd_test':to_ymd_test,
    'depth' : depth,
    'load_predtrain' : load_predtrain,
    'predtreain_path' :predtreain_path,
    'LOSS':LOSS,
    'stride':stride,
    'weight_decay':weight_decay
}
[writer.add_text(k,str(params_dict[k])) for k in params_dict.keys()]
#writer.add_hparams(params_dict,metric_d)
img=0
step = 0
ep = 0
test_step=0
current_step = 0
if not os.path.isdir(f'Ice//{NAME}'):
    os.mkdir(f'Ice//{NAME}')
for epoch in tqdm(range(epochs)):
    # writer.add_scalar('Lr',optimizer.param_groups[0]['lr'],ep)
    # ep+=1
    
    # model.train()
    # #TRAIN
    # for i,batch in enumerate(dataloader_train):
    #     print(i)
    #     X,y,x_d,y_d=batch
    #     step+=1
    #     #current_step +=1
    #     X = X.to(device)
    #     y = y.squeeze(1).to(device)
    #     outputs = model(X)
        
    #     loss = loss_fn(outputs,y)/ accumulation_steps
    #     writer.add_scalar('Loss_train',loss.item()*accumulation_steps,step)
    #     loss.backward()
    #     if (i + 1) % accumulation_steps == 0:
    #         optimizer.step()
    #         optimizer.zero_grad()
    # optimizer_sch.step(ep)
    # optimizer.zero_grad()
    # del outputs,X
    # torch.cuda.empty_cache()
    #writer.add_scalar('GPU',torch.cuda.memory_summary(device),ep)
    
    #TEST
    model.eval()
    with torch.no_grad():
        tt = 1
        for i,batch in enumerate(dataloader_test):
            X,y,x_d,y_d = batch
            print(y_d)
            test_step+=1
            #current_step +=1
            X = X.to(device)
            y = y.squeeze(1).to(device)
            outputs = model(X)
            loss =loss_fn(outputs,y)
            loss_masked =loss_fn(outputs*torch.tensor(np.float32(mask)).to(device),y)
            imgs=np.absolute(outputs.detach().cpu().numpy()-y.detach().cpu().numpy())
            writer.add_images('Loss_masks', np.expand_dims(imgs[0],axis=1), i)
            writer.add_images('Ground_truth', np.expand_dims(y.detach().cpu().numpy()[0],axis=1), i)
            writer.add_images('Predicts', np.expand_dims(outputs.detach().cpu().numpy()[0],axis=1), i)
            #img+=0.005
            lossses = [[i,n.item()] for i,n in enumerate(loss.mean(dim=-1).mean(dim=-1)[0])]
            [writer.add_scalar('Loss_test',n,(time+tt)) for time,n in lossses]
            tt+=lossses[-1][0]
            writer.add_scalar('Masked_loss_test',loss_masked.mean(dim=-1).mean(dim=-1).mean(dim=-1).item(),test_step)
#         del outputs
#         torch.cuda.empty_cache()
#     if ep%30==0:
#         if PARALLEL:
#             torch.save(model.module.state_dict(), f'Ice//{NAME}/Module_model_weights_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}__dates_{from_ymd_train[0]}to_{to_ymd_train[0]}_stride{stride}_sigmoid')
#         else:
#             torch.save(model.state_dict(), f'Ice//{NAME}/model_weights_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}__dates_{from_ymd_train[0]}to_{to_ymd_train[0]}_stride{stride}_sigmoid')     

# if PARALLEL:
#     torch.save(model.module.state_dict(), f'Ice//{NAME}/Module_model_weights_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}__dates_{from_ymd_train[0]}to_{to_ymd_train[0]}_stride{stride}_sigmoid')
# else:
#     torch.save(model.state_dict(), f'Ice//{NAME}/model_weights_in_per_{in_period}_pred_per_{predict_period}_bs_{batch_size}__dates_{from_ymd_train[0]}to_{to_ymd_train[0]}_stride{stride}_sigmoid')     

# for train in dataloaders[0]:
#     predd = model(train[0].to('cuda'))
#     print(train[0].shape, train[1].shape)
