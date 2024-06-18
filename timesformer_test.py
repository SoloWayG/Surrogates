from read_npy import create_dataloaders
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/TimeSformer')
from timesformer.models.vit import TimeSformer
from pathlib import Path
import torch
from torchsummary import summary

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def embed_dim_by_img(img,num_heads):
    emb_dim = img*2
    head_det = emb_dim%num_heads
    if head_det!=0:
        emb_dim=emb_dim-head_det+num_heads
    return emb_dim


predict_period = 3
in_period = 3
batch_size = 3
num_heads=12
place='kara'
dataloaders, img_sizes = create_dataloaders(path_to_dir=f'Ice/{place}',
                                            batch_size=batch_size,
                                            in_period=in_period,
                                            predict_period=predict_period,
                                            stride=7,
                                            test_end=None,
                                            from_ymd=[1999, 2, 4],
                                            to_ymd=[2000, 4, 4],
                                            pad=True,
                                            shuffle_dataset=False)
if img_sizes[1]>img_sizes[0]:
    img_sizes=(img_sizes[1],img_sizes[0])

patch_size = int(img_sizes[0]/(img_sizes[0]*2)**0.5)
shape_end = (img_sizes[0]//patch_size)**2*in_period
embed_dim = embed_dim_by_img(img_sizes[1],num_heads)

# model = TimeSformer(batch_size=batch_size, output_size=[img_sizes[0], img_sizes[1]], img_size=img_sizes[0],embed_dim=embed_dim,
#                     num_frames=4, attention_type='divided_space_time', pretrained_model=False, in_chans=1, out_chans=predict_period,
#                     patch_size=patch_size,num_heads=num_heads,in_periods=in_period,place=place).to('cuda')

# # (batch x channels x frames x height x width)
# dummy_video = torch.randn(1, 4, 452, 452)

for train in dataloaders[0]:
    print(train[0].sum())
    # print(train[0].shape, train[1].shape)
    # #predd = model(train[0].to('cuda'))
    # print(train[0].shape, train[1].shape)
