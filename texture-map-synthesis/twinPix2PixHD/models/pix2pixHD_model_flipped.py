### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_functions as networks
# from . import networks

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'FlippedPix2PixHDModel'
        # --label_nc 0
        # --no_instance

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else 3

        ##### define networks
        
        # Generator networks 1 & 2
        netG_input_nc = input_nc # 3
        if not opt.no_instance: netG_input_nc += 1
        if self.use_features: netG_input_nc += opt.feat_num # False # 3
        self.netG1 = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.netG2 = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc # 3 + 3
            if not opt.no_instance: netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
                                          
        # Classifier network
        if self.isTrain:
            # opt.class_loss, opt.class_nc, opt.n_layers_C, opt.num_C = True, 3, 3, 1
            # use_sigmoid = opt.class_loss # selects between BCE and MSE loss
            netC_input_nc = opt.class_nc # number of channels for the class.network/densenet
            # if not opt.no_instance: netD_input_nc += 1
            self.netC = networks.define_C(netC_input_nc, opt.ndf, opt.n_layers_C, opt.norm, use_sigmoid,
                                          opt.num_C, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network ?????
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain # if test mode '' else continue @ opt.load_pretrain
            self.load_network(self.netG1, 'G1', opt.which_epoch, pretrained_path)
            self.load_network(self.netG2, 'G2', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size) # check this
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss() # adding L1 G loss
            self.criterionL2 = torch.nn.L1Loss() # adding L2 G loss
            self.criterionCos1 = networks.CosineLoss1() # adding cosine G loss
            self.criterionCos2 = networks.CosineLoss2() # adding cosine G loss            
            self.criterionCEL = torch.nn.CosineEmbeddingLoss() # adding cosine G loss by torch.nn
           
            criterion = 'param'; KL = 'qp'
            if criterion == 'param':
                # print('Using parametric criterion KL_%s' % KL)
                # KL_minimizer = losses.KLN01Loss(direction=opt.KL, minimize=True)
                # KL_maximizer = losses.KLN01Loss(direction=opt.KL, minimize=False)
            
                self.criterionKL_min = networks.KLN01Loss(direction=KL, minimize=True)
                self.criterionKL_max = networks.KLN01Loss(direction=KL, minimize=False)
            
            if not opt.no_vgg_loss: 
                    self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = ['G1_L1_z', 'G2_L1_z', 'G1_cos_z', 'G2_cos_z', 
                               # 'G_cos1_z', 'G_cos2_z', 'G_cos1', 'G_cos2',
                               # 'E_KL_real', 'E_KL_fake', 'G_KL_fake', 'G_L1',
                               'G1_GAN', 'G2_GAN', 'G1_GAN_Feat', 'G2_GAN_Feat', 'G1_VGG', 'G2_VGG',
                               'D_real', 'D_fake'] 
                               # added G_cos1, G_cos2, G_L1, G_KL, E_KL (E_KL_real, E_KL_fake)
            
            self.loss_weights = [opt.lambda_Gs_L1_z, 
                                 opt.lambda_Gs_L1_z, 
                                 opt.lambda_Gs_cos_z, 
                                 opt.lambda_Gs_cos_z, 
                                 # opt.lambda_G_cos1_z,
                                 # opt.lambda_G_cos2_z,
                                 # opt.lambda_G_cos1,
                                 # opt.lambda_G_cos2,
                                 # opt.lambda_E_KL_real,
                                 # opt.lambda_E_KL_fake,
                                 # opt.lambda_G_KL_fake,
                                 # opt.lambda_L1,
                                 1.0, 1.0,
                                 0 if opt.no_ganFeat_loss else opt.lambda_feat, 
                                 0 if opt.no_ganFeat_loss else opt.lambda_feat, 
                                 0 if opt.no_vgg_loss else opt.lambda_feat,  
                                 0 if opt.no_vgg_loss else opt.lambda_feat, 
                                 0.5, 
                                 0.5 ]
                                 
            print('===================== LOSSES =====================')
            [print ('%s: %.2f' %(i, j)) for i, j in zip(self.loss_names, self.loss_weights)]
            print('==================================================')
            
            # initialize optimizers
            # optimizer G ???
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]
            else:
                params = list(self.netG1.parameters()) 
                params += list(self.netG2.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G1 = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
         

    def encode_input(self, label_map_1, label_map_2, inst_map=None, real_image_1=None, real_image_2=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            # input_label = label_map.data.cuda()
            input_label_1, input_label_2 = label_map_1.data.cuda(), label_map_2.data.cuda()
        else: # def. false / not updated
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])

            if self.gpu_ids == '-1': #with CPU:
                input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            else:
                input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            
        # get edges from instance map
        if not self.opt.no_instance: # def false / not updated
            # inst_map = inst_map.data.cuda()
            if self.gpu_ids == '-1': inst_map = inst_map.data
            else: inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
            
        input_label_1, input_label_2 = Variable(input_label_1, volatile=infer), Variable(input_label_2, volatile=infer)

        # real images for training
        if real_image_1 is not None: real_image_1 = Variable(real_image_1.data.cuda())
        if real_image_2 is not None: real_image_2 = Variable(real_image_2.data.cuda())

        # instance map for feature encoding
        if self.use_features: # def. false
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())

        return input_label_1, input_label_2, inst_map, real_image_1, real_image_2, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label_1, label_2, inst, image_1, image_2, feat, infer=False):
        # Encode Inputs
        input_label_1, input_label_2, inst_map, real_image_1, real_image_2, feat_map = \
        self.encode_input(label_1, label_2, inst, image_1, image_2, feat)
        # print(input_label.shape, inst_map.shape, real_image.shape, feat_map.shape)
        # torch.Size([1, 3, 256, 256]) torch.Size([1]) torch.Size([1, 3, 256, 256]) torch.Size([1])
        
        # Fake Generation
        if self.use_features: # def. false
            if not self.opt.load_features: # def. false / not updated
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1) 
        else:
            input_concat_1, input_concat_2 = input_label_1, input_label_2
            
        # print(input_concat.shape)
        fake_image_1 = self.netG1.forward(input_concat_1)
        fake_image_2 = self.netG2.forward(input_concat_2)
        # fake_image_detached = self.netG.forward(input_concat).detach()

        # Fake Detection and Loss
        # in discriminate() we detach() as we dont want G to backward D_fake loss, but G_GAN loss instead (1)
        pred_fake_1_pool = self.discriminate(input_label_1, fake_image_1, use_pool=True)
        pred_fake_2_pool = self.discriminate(input_label_2, fake_image_2, use_pool=True)
        
        loss_D_fake_1 = self.criterionGAN(pred_fake_1_pool, False)
        loss_D_fake_2 = self.criterionGAN(pred_fake_2_pool, False)
        loss_D_fake = loss_D_fake_1 + loss_D_fake_2
        
        # Real Detection and Loss
        pred_real_1 = self.discriminate(input_label_1, real_image_1)
        pred_real_2 = self.discriminate(input_label_2, real_image_2)
        
        loss_D_real_1 = self.criterionGAN(pred_real_1, True)
        loss_D_real_2 = self.criterionGAN(pred_real_2, True)
        loss_D_real = loss_D_real_1 + loss_D_real_2
        
        # GAN loss (Fake Passability Loss) (1)
        pred_fake_1 = self.netD.forward(torch.cat((input_label_1, fake_image_1), dim=1))
        pred_fake_2 = self.netD.forward(torch.cat((input_label_2, fake_image_2), dim=1))
        loss_G1_GAN = self.criterionGAN(pred_fake_1, True)
        loss_G2_GAN = self.criterionGAN(pred_fake_2, True)

        # GAN feature matching loss  
        loss_G1_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_1[i])-1):
                    loss_G1_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake_1[i][j], pred_real_1[i][j].detach()) * self.opt.lambda_feat
        
        loss_G2_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_2[i])-1):
                    loss_G2_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake_2[i][j], pred_real_2[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G1_VGG, loss_G2_VGG = 0, 0
        if not self.opt.no_vgg_loss: loss_G1_VGG = self.criterionVGG(fake_image_1, real_image_1) * self.opt.lambda_feat
        if not self.opt.no_vgg_loss: loss_G2_VGG = self.criterionVGG(fake_image_2, real_image_2) * self.opt.lambda_feat

        # adding L1
        # fake_image = self.netG.forward(input_concat)
        loss_G1_L1 = self.criterionL1(fake_image_1, real_image_1) * self.opt.lambda_L1 
        loss_G2_L1 = self.criterionL1(fake_image_2, real_image_2) * self.opt.lambda_L1 
        
        # calculate Z
        self.netGE1 = self.netG1.model_down
        self.netGE2 = self.netG2.model_down
        
        # # adding KL
        # loss_E_KL_real = self.criterionKL_min(self.netGE(real_image)) * self.opt.lambda_E_KL_real
        # loss_E_KL_fake = self.criterionKL_max(self.netGE(fake_image.detach())) * self.opt.lambda_E_KL_fake
        # loss_G_KL_fake = self.criterionKL_min(self.netGE(fake_image)) * self.opt.lambda_G_KL_fake;
        #
        # # adding Cosine loss
        # loss_G_cos1 = self.criterionCos1(fake_image, real_image) * self.opt.lambda_G_cos1
        # loss_G_cos2 = self.criterionCos2(fake_image, real_image) * self.opt.lambda_G_cos2
        # loss_G_cos1_z = self.criterionCos1(self.netGE(fake_image), self.netGE(real_image)) * self.opt.lambda_G_cos1_z
        # loss_G_cos2_z = self.criterionCos2(self.netGE(fake_image), self.netGE(real_image)) * self.opt.lambda_G_cos2_z 
        
        # adding input-mirror Z loss
        loss_G1_L1_z = self.criterionL1(self.netGE1(input_label_1), self.netGE2(input_label_2).detach()) * self.opt.lambda_Gs_L1_z
        loss_G2_L1_z = self.criterionL2(self.netGE2(input_label_2), self.netGE1(input_label_1).detach()) * self.opt.lambda_Gs_L1_z
        loss_G1_cos_z = self.criterionCos2(self.netGE1(input_label_1), self.netGE2(input_label_2).detach()) * self.opt.lambda_Gs_cos_z
        loss_G2_cos_z = self.criterionCos2(self.netGE2(input_label_2), self.netGE1(input_label_1).detach()) * self.opt.lambda_Gs_cos_z
        
        # Only return the fake_B image if necessary to save BW
        return [[loss_G1_L1_z, loss_G2_L1_z, loss_G1_cos_z, loss_G2_cos_z,
                 # loss_G_cos1_z, loss_G_cos2_z, loss_G_cos1, loss_G_cos2,
                 # loss_E_KL_real, loss_E_KL_fake, loss_G_KL_fake, loss_G_L1,
                 loss_G1_GAN, loss_G2_GAN, 
                 loss_G1_GAN_Feat, loss_G2_GAN_Feat, 
                 loss_G1_VGG, loss_G2_VGG, 
                 loss_D_real, loss_D_fake], [None, None] if not infer else [fake_image_1, fake_image_2]]

    def inference(self, label_1, label_2, inst):
        # Encode Inputs
        input_label_1, input_label_2, inst_map, _, _, _ = self.encode_input(Variable(label_1), 
                                                                            Variable(label_2), 
                                                                            Variable(inst), infer=True)

        # Fake Generation
        if self.use_features:
            # sample clusters from precomputed features
            feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else: input_concat_1, input_concat_2 = input_label_1, input_label_2
        fake_image_1 = self.netG1.forward(input_concat_1)
        fake_image_2 = self.netG2.forward(input_concat_2)
        return fake_image_1, fake_image_2

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = torch.cuda.FloatTensor(1, self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])
                idx = (inst == i).nonzero()
                for k in range(self.opt.feat_num): feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc): feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == i).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        # edge = torch.cuda.ByteTensor(t.size()).zero_()
        if self.gpu_ids == '-1': edge = torch.ByteTensor(t.size()).zero_()
        else: edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG1, 'G1', which_epoch, self.gpu_ids)
        self.save_network(self.netG2, 'G2', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features: self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params_1 = list(self.netG1.parameters())
        params_2 = list(self.netG2.parameters())
        if self.gen_features:
            params_1 += list(self.netE.parameters())
        self.optimizer_G1 = torch.optim.Adam(params_1, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_G2 = torch.optim.Adam(params_2, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd; print(lr, lrd)
        for param_group in self.optimizer_D.param_groups: param_group['lr'] = lr
        for param_group in self.optimizer_G1.param_groups: param_group['lr'] = lr
        for param_group in self.optimizer_G2.param_groups: param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr