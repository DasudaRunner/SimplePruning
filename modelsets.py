#coding:utf-8

'''
author:Wang Haibo
at: Pingan Tec.
email: haibo.david@qq.com

!!!
代码中会有少量中文注释，无需在意

'''

from pruner import Pruner

__all__ = ["CifarModelZoo"]

class CifarModelZoo():

    def __init__(self):

        # resnet config
        self.model_config = {"resnet18":[2,2,2,2],"resnet34":[3,4,6,3],"resnet50":[3,4,6,3]}
        self.init_channels = 64
        self.block_version = 2
        self.support_model = ["simpleNet","DenseNet40","MobileNetV1","vgg19","resnet18","resnet34"]
        # end

    @classmethod
    def getModel(cls,model_name=None,params=None):

        if not isinstance(model_name,str) and not isinstance(params,dict):
            raise TypeError("model_name must be str, params must be dict.")
        if len(params)!=4:
            raise ValueError("params error.")
        if model_name not in cls().support_model:
            raise ValueError(model_name+" not implemented in modelsets.")

        return_func = None

        if model_name=="simpleNet":
            return_func = cls().simpleNet

        elif model_name=="DenseNet40":
            return_func = cls().DenseNet40

        elif model_name=="MobileNetV1":
            return_func = cls().MobileNetV1

        elif model_name=="vgg19":
            return_func = cls().vgg19

        elif model_name=="resnet18":
            return_func = cls().ResNet18

        elif model_name=="resnet34":
            return_func = cls().ResNet34

        return return_func(inputs=params["inputs"],
                           is_train=params["is_train"],
                           reload_w=params["reload_w"],
                           num_classes=params["num_classes"])

    def simpleNet(self,inputs=None,is_train=True,reload_w=None,num_classes=None):

        model = Pruner(reload_file=reload_w)

        k_size = 5
        x = model._add_layer(inputs, mode='conv', out_c=32, k_size=k_size, strides=1, is_train=is_train)
        x = model._add_layer(x, mode='conv', out_c=64, k_size=k_size, strides=2, is_train=is_train)
        x1 = model._add_layer(x, mode='conv', out_c=64, k_size=k_size, strides=1, is_train=is_train)
        x2 = model._add_layer(x, mode='conv', out_c=64, k_size=k_size, strides=1, is_train=is_train)

        x = model.Add_layer(x1,x2)

        x = model._add_layer(x, mode='conv', out_c=64, k_size=k_size, strides=1, is_train=is_train)
        x = model._add_layer(x, mode='conv', out_c=96, k_size=k_size, strides=2, is_train=is_train)

        x = model._add_layer(x, mode='conv', out_c=96, k_size=k_size, strides=1, is_train=is_train)
        x = model._add_layer(x, mode='conv', out_c=128, k_size=k_size, strides=2, is_train=is_train)

        x = model.gap_layer(x)

        x = model._add_layer(x, mode="fc", out_c=num_classes, with_bn=False, act=None)

        return x,model

    def __resnet_block_v2(self,model, inputs, out_c, strides, is_train):

        output = model.bn_act_layer(inputs,is_train=is_train)

        if strides==2:
            shortcut = model._add_layer(output, mode="conv", out_c=out_c, k_size=3, strides=2, with_bn=False,
                                 act=None)
        else:
            shortcut = inputs

        output = model._add_layer(output, mode="conv", out_c=out_c, k_size=3, strides=strides, with_bn=True,
                                  is_train=is_train)

        output = model._add_layer(output, mode="conv", out_c=out_c, k_size=3, strides=1, with_bn=False,
                                    act=None)

        output = model.Add_layer(shortcut, output)

        return output

    def __resnet_block_v1(self,model, inputs, out_c, strides, is_train):

        output = model._add_layer(inputs, mode="conv", out_c=out_c, k_size=3, strides=strides, with_bn=True,
                                  is_train=is_train)
        output = model._add_layer(output, mode="conv", out_c=out_c, k_size=3, strides=1, with_bn=True,
                                  is_train=is_train)

        if strides==2:
            shortcut = model._add_layer(inputs, mode="conv", out_c=out_c, k_size=3, strides=strides, with_bn=True,
                                        is_train=is_train)
        else:
            shortcut = inputs

        output = model.Add_layer(shortcut,output)

        return output

    def ResNet34(self,inputs=None,is_train=True,reload_w=None,num_classes=None):

        model = Pruner(reload_file=reload_w)

        num_block = self.model_config["resnet34"]

        block_func = self.__resnet_block_v1 if self.block_version==1 else self.__resnet_block_v2

        if self.block_version==1:
            x = model._add_layer(inputs,mode="conv",out_c=self.init_channels,k_size=3,strides=1,with_bn=True,is_train=is_train)
        else:
            x = model._add_layer(inputs,mode="conv",out_c=self.init_channels,k_size=3,strides=1,with_bn=False,act=None)

        # stage 1 out size = 32
        for _ in range(num_block[0]):
            x = block_func(model,inputs=x,out_c=self.init_channels,strides=1,is_train=is_train)

        # stage 2 out_size = 16
        x = block_func(model, inputs=x, out_c=self.init_channels*2, strides=2, is_train=is_train)
        for _ in range(num_block[1]-1):
            x = block_func(model, inputs=x, out_c=self.init_channels*2, strides=1, is_train=is_train)

        # stage 3 out_size = 8
        x = block_func(model, inputs=x, out_c=self.init_channels*4, strides=2, is_train=is_train)
        for _ in range(num_block[2]-1):
            x = block_func(model, inputs=x, out_c=self.init_channels*4, strides=1, is_train=is_train)

        # stage 4 out_size = 4
        x = block_func(model, inputs=x, out_c=self.init_channels*8, strides=2, is_train=is_train)
        for _ in range(num_block[3]-1):
            x = block_func(model, inputs=x, out_c=self.init_channels*8, strides=1, is_train=is_train)

        if self.block_version==2:
            x = model.bn_act_layer(x,is_train=is_train)

        x = model.gap_layer(x)
        x = model._add_layer(x,mode="fc",out_c=num_classes,act=None,with_bn=False)

        return x,model

    def ResNet18(self,inputs=None,is_train=True,reload_w=None,num_classes=None):

        model = Pruner(reload_file=reload_w)

        num_block = self.model_config["resnet18"]

        block_func = self.__resnet_block_v1 if self.block_version==1 else self.__resnet_block_v2

        if self.block_version==1:
            x = model._add_layer(inputs,mode="conv",out_c=self.init_channels,k_size=3,strides=1,with_bn=True,is_train=is_train)
        else:
            x = model._add_layer(inputs,mode="conv",out_c=self.init_channels,k_size=3,strides=1,with_bn=False,act=None)

        # stage 1 out size = 32
        for _ in range(num_block[0]):
            x = block_func(model,inputs=x,out_c=self.init_channels,strides=1,is_train=is_train)

        # stage 2 out_size = 16
        x = block_func(model, inputs=x, out_c=self.init_channels*2, strides=2, is_train=is_train)
        for _ in range(num_block[1]-1):
            x = block_func(model, inputs=x, out_c=self.init_channels*2, strides=1, is_train=is_train)

        # stage 3 out_size = 8
        x = block_func(model, inputs=x, out_c=self.init_channels*4, strides=2, is_train=is_train)
        for _ in range(num_block[2]-1):
            x = block_func(model, inputs=x, out_c=self.init_channels*4, strides=1, is_train=is_train)

        # stage 4 out_size = 4
        x = block_func(model, inputs=x, out_c=self.init_channels*8, strides=2, is_train=is_train)
        for _ in range(num_block[3]-1):
            x = block_func(model, inputs=x, out_c=self.init_channels*8, strides=1, is_train=is_train)

        if self.block_version==2:
            x = model.bn_act_layer(x,is_train=is_train)

        x = model.gap_layer(x)
        x = model._add_layer(x,mode="fc",out_c=num_classes,act=None,with_bn=False)

        return x,model

    def densenet_block(self,model,inputs=None,is_train=True):

        def _add_dense_layer(in_):
            in_c = in_.get_shape().as_list()[-1]
            c = model.bn_act_layer(in_,is_train=is_train)
            c = model._add_layer(c, mode="conv", out_c=12, k_size=3, strides=1, with_bn=False,act=None)
            out = model.concat_layer([in_,c],3)
            return out

        for i in range(12):
            inputs = _add_dense_layer(inputs)

        return inputs

    def densenet_trans(self,model,inputs=None,is_train=True):
        in_c = inputs.get_shape().as_list()[-1]
        c = model.bn_act_layer(inputs,is_train=is_train)
        c = model._add_layer(c, mode="conv", out_c=in_c, k_size=1, strides=1, with_bn=False,act=None)
        c = model.pool_layer(c,"avg",pool_size=2,strides=2)
        return c

    def DenseNet40(self,inputs=None,is_train=True,reload_w=None,num_classes=None):

        model = Pruner(reload_file=reload_w)
        # net header
        x = model._add_layer(inputs, mode="conv", out_c=32, k_size=3, strides=1, with_bn=False,act=None)

        # stage 1
        x = self.densenet_block(model,x,is_train=is_train)
        x = self.densenet_trans(model,x,is_train=is_train) #16
        # stage 2
        x = self.densenet_block(model,x,is_train=is_train)
        x = self.densenet_trans(model,x,is_train=is_train) #8
        # stage 3
        x = self.densenet_block(model,x,is_train=is_train)

        x = model.bn_act_layer(x,is_train=is_train)
        x = model.gap_layer(x)
        x = model._add_layer(x,mode="fc",out_c=num_classes,act=None,with_bn=False)

        return x,model

    def vgg19(self,inputs=None,is_train=True,reload_w=None,num_classes=None):

        model = Pruner(reload_file=reload_w)

        x = inputs

        init_size = 64

        for i in range(4):
            x = model._add_layer(x, mode="conv", out_c=init_size*(2**i), k_size=3, strides=1, with_bn=False)
            x = model._add_layer(x, mode="conv", out_c=init_size*(2**i), k_size=3, strides=1, with_bn=False)
            x = model.pool_layer(x,"max",pool_size=2,strides=2)

        x = model._add_layer(x, mode="conv", out_c=512, k_size=3, strides=1, with_bn=False)
        x = model._add_layer(x, mode="conv", out_c=512, k_size=3, strides=1, with_bn=False)
        x = model._add_layer(x, mode="conv", out_c=512, k_size=3, strides=1, with_bn=False)
        x = model.gmp_layer(x)

        x = model._add_layer(x, mode="fc", out_c=1024, with_bn=False)
        x = model._add_layer(x, mode="fc", out_c=1024, with_bn=False)
        x = model._add_layer(x, mode="fc", out_c=num_classes, with_bn=False,act=None)

        return x,model

    def MobileNetV1(self,inputs=None,is_train=None,reload_w=None,num_classes=None):

        model = Pruner(reload_file=reload_w)

        x = model._add_layer(inputs,mode="conv",out_c=32,k_size=3,strides=1)

        x = model._add_layer(x, mode="dconv", out_c=64, k_size=3, strides=1,with_bn=True,is_train=is_train)
        x = model._add_layer(x, mode="dconv", out_c=128, k_size=3, strides=2,with_bn=True,is_train=is_train)

        x = model._add_layer(x, mode="dconv", out_c=128, k_size=3, strides=1,with_bn=True,is_train=is_train)
        x = model._add_layer(x, mode="dconv", out_c=256, k_size=3, strides=2,with_bn=True,is_train=is_train)

        x = model._add_layer(x, mode="dconv", out_c=256, k_size=3, strides=1,with_bn=True,is_train=is_train)
        x = model._add_layer(x, mode="dconv", out_c=512, k_size=3, strides=2,with_bn=True,is_train=is_train)

        x = model._add_layer(x, mode="dconv", out_c=512, k_size=3, strides=1, with_bn=True,is_train=is_train)
        x = model._add_layer(x, mode="dconv", out_c=512, k_size=3, strides=1, with_bn=True,is_train=is_train)
        x = model._add_layer(x, mode="dconv", out_c=512, k_size=3, strides=1, with_bn=True,is_train=is_train)
        x = model._add_layer(x, mode="dconv", out_c=512, k_size=3, strides=1, with_bn=True,is_train=is_train)
        x = model._add_layer(x, mode="dconv", out_c=512, k_size=3, strides=1, with_bn=True,is_train=is_train)

        x = model.gap_layer(x)

        x = model._add_layer(x,mode="fc",out_c=num_classes,act=None)

        return x,model