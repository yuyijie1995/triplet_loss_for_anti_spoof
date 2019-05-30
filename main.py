#-*-coding:utf-8-*-
import argparse,time,logging,math
import numpy as np
import mxnet as mx
from mxnet import gluon,nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import pdb
from net import OutputNet
import data_loader as loader

from gluoncv.model_zoo import get_model

train_path_living = '/mnt/data-3/data/yijie.yu/train_rect2.0_living'
train_path_spoof = '/mnt/data-3/data/yijie.yu/train_rect2.0_spoof'
val_path_living = '/mnt/data-3/data/yijie.yu/val_rect2.0_living'
val_path_spoof = '/mnt/data-3/data/yijie.yu/val_rect2.0_spoof'
test_path_living = '/mnt/data-3/data/yijie.yu/test_rect2.0_living'
test_path_spoof = '/mnt/data-3/data/yijie.yu/test_rect2.0_spoof'

batch_size = 32
num_seq = 3
color = 'GRAY'
random = True

def train():
    ctx=mx.gpu(0)
    net=get_model('resnet18_v2',pretrained=True)
    backbone=net.features[:]
    out_net=OutputNet()
    out_net.initialize(mx.initializer.Xavier(factor_type='in',magnitude=2))
    params=backbone.collect_params()
    params.update(out_net.collect_params())
    optimizer=mx.optimizer.SGD(learning_rate=0.001,wd=5e-4,momentum=0.9)
    trainer=gluon.Trainer(params=params,optimizer=optimizer)
    triplet_loss = gluon.loss.TripletLoss()  # TripletLoss损失函数

    num_step=0
    for epoch in range(30):
        print('epoch:{}'.format(epoch))
        tic=time.time()
        train_data=loader.loadData(list_path_living=train_path_living,list_path_spoof=train_path_spoof,batch_size=batch_size,num_seq=num_seq,color=color,
                                   is_train=True,random=random)
        val_data=loader.loadData(list_path_living=val_path_living,list_path_spoof=val_path_spoof,batch_size=batch_size,num_seq=num_seq,color=color,
                                   is_train=False,random=random)

        train_loss,correct,total=[0,0,0]
        for data,label in train_data:
            num_step+=1
            data=data.copyto(ctx)
            label=label.copyto(ctx)

            data_anchor=data[:,0,:,:,:]
            data_positive=data[:,1,:,:,:]
            data_negative=data[:,2,:,:,:]
            # data_a=data_anchor.reshape(data_anchor.shape[0]*data_anchor.shape[1],data_anchor.shape[2])
            with ag.record():
                feat_anchor=backbone(data_anchor)
                y_anchor=out_net(feat_anchor)
                feat_positive=backbone(data_positive)
                y_positve=out_net(feat_positive)
                feat_negative=backbone(data_negative)
                y_negative=out_net(feat_negative)
                loss=triplet_loss(y_anchor,y_positve,y_negative)
                loss=loss.sum()
            loss.backward()
            trainer.step(batch_size,ignore_stale_grad=True)
            train_loss+=loss.asscalar()
        print('Epoch:{}. loss: {}'.format(epoch,train_loss))



if __name__=='__main__':
    # x=nd.random_uniform(shape=(1,3,128,128))
    # net=get_model('resnet18_v2',pretrained=True)
    # for layer in net:
    #     x=layer(x)
    #     print(layer.name,'output shape:\t',x.shape)
    #
    # backbone=net.features[:]
    train()



