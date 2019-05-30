#-*-coding:utf-8-*-
from mxnet import gluon
from mxnet.gluon import nn
import mxnet as mx
from mxnet import gluon,nd
from gluoncv.model_zoo import get_model



#
# class Backbone(gluon.HybridBlock):
#     def __init__(self):
#         super(Backbone,self).__init__()
#         with self.name_scope():
























class OutputNet(gluon.HybridBlock):
    def __init__(self):
        super(OutputNet,self).__init__()

        with self.name_scope():
            self.feat_layer=nn.HybridSequential()
            self.feat_layer.add(
                nn.Activation(activation='relu'),
                nn.Dropout(rate=0.5),
                nn.Dense(128)
            )
            self.embed_layer=nn.HybridSequential()
            self.embed_layer.add(
                nn.Activation(activation='relu'),
                nn.Dense(64)
            )

    def hybrid_forward(self, F, x):
        feat=self.feat_layer(x)
        embedding=self.embed_layer(feat)
        return feat,embedding

class OutputNet_nd(gluon.Block):
    def __init__(self):
        super(OutputNet_nd,self).__init__()

        with self.name_scope():
            self.feat_layer=nn.HybridSequential()
            self.feat_layer.add(
                nn.Activation(activation='relu'),
                nn.Dropout(rate=0.5),
                nn.Dense(128)
            )
            self.embed_layer=nn.HybridSequential()
            self.embed_layer.add(
                nn.Activation(activation='relu'),
                nn.Dense(64)
            )

    def forward(self, x):
        feat=self.feat_layer(x)
        embedding=self.embed_layer(feat)
        return embedding



if __name__=='__main__':
    x = nd.random_uniform(shape=(1, 3, 128, 128))
    net = get_model('resnet18_v2', pretrained=True)
    # for layer in net:
    #     x = layer(x)
    #     print(layer.name, 'output shape:\t', x.shape)

    backbone = net.features[:]
    out_net=OutputNet_nd()
    out_net.initialize(mx.initializer.Xavier(factor_type='in',magnitude=2))
    feat=backbone(x)
    embedding=out_net(feat)
    print(embedding)