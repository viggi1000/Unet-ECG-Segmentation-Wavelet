import torch.nn as nn
import torch


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


class UNet (nn.Module):
    
    def __init__(self, in_shape, num_classes):
        super(UNet, self).__init__()
        in_channels, height, width = in_shape
        #
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=2, stride=1,padding=1))
            
        
        #
        self.e2 = nn.Sequential(nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(128))

        #
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(256))
        #

        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256,512, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(512)) 

        #
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,512, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(512))
        #
        #
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,512, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(512)) 
        #
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,512, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(512)) 
        #
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,512, kernel_size=2, stride=1,padding=1))
            #nn.BatchNorm2d(512))
        ##
        ##
        ##
        self.d1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5))
        #
        self.d2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5))
        #
        self.d3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5))
        #
        self.d4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(512))

        #
        self.d5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(256))
        #
        self.d6 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(128))
        #
        self.d7 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(64))
        #
        self.d8 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.out_l = nn.Sequential(
            nn.Conv2d(64,num_classes,kernel_size=1,stride=1),
            nn.Sigmoid()
            #nn.Conv2d(64,num_classes,kernel_size=1,stride=1))
        )
           #nn.ReLU(inplace=True)) # -------------------------- Might have to change this activation

 
           
    def forward(self, x):

        #### Encoder #####
        #print "Encoder"
        en1 = self.e1(x)
        #print "E1:",en1.size()
        en2 = self.e2(en1)
        #print "E2:",en2.size()
        en3 = self.e3(en2)
        #print "E3:",en3.size()
        en4 = self.e4(en3)
        #print "E4:",en4.size()
        en5 = self.e5(en4)
        # print ("E5:",en5.size())
        en6 = self.e6(en5)
        #print "E6:",en6.size()
        en7 = self.e7(en6)
        #print "E7:",en7.size()
        en8 = self.e8(en7)
        #print "E8:",en8.size()
        

        #### Decoder ####
        #print "Decoder"
        de1_ = self.d1(en8)
        #print de1_.size()
        de1 = torch.cat([en7,de1_],1)
        #print de1.size()

        de2_ = self.d2(de1)
        #print de2_.size()
        de2 = torch.cat([en6,de2_],1)
        #print de2.size()
        
        
        de3_ = self.d3(de2)
        #print de3_.size()
        de3 = torch.cat([en5,de3_],1)
        #print de3.size()
        
        de4_ = self.d4(de3)
        #print de4_.size()
        de4 = torch.cat([en4,de4_],1)
        #print de4.size()
        
        
        de5_ = self.d5(de4)
        #print de5_.size()
        de5 = torch.cat([en3,de5_],1)
        #print de5.size()
        
        de6_ = self.d6(de5)
        #print de6_.size()
        de6 = torch.cat([en2,de6_],1)
        #print de6.size()
        
        
        de7_ = self.d7(de6)
        #print de7_.size()
        de7 = torch.cat([en1,de7_],1)
        #print de7.size()
        de8 = self.d8(de7)
        #print de8.size()

        out_l_mask = self.out_l(de8)
        # print(out_l_mask.size())
        #out_l_mask  = nn.functional.log_softmax(out_l_mask)#.squeeze(0)
        #print out_l.size()


        return out_l_mask 

#####################################
#


#### Inception Redidual Net

class IncResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(IncResBlock, self).__init__()
        self.Inputconv1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        #
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(inplanes, planes/4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes/4))
            #nn.ReLU())
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(inplanes, planes/4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes/4),
            nn.ReLU(),
            nn.Conv2d(planes/4, planes/4, kernel_size=3, stride=stride, dilation=2 ,  padding=2, bias=False),
            nn.BatchNorm2d(planes/4))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(inplanes, planes/4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes/4),
            nn.ReLU(),
            nn.Conv2d(planes/4, planes/4, kernel_size=3, stride=stride, dilation=4 ,  padding=4, bias=False),
            nn.BatchNorm2d(planes/4))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(inplanes, planes/4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes/4),
            nn.ReLU(),
            nn.Conv2d(planes/4, planes/4, kernel_size=3, stride=stride, dilation=8 ,  padding=8, bias=False),
            nn.BatchNorm2d(planes/4))
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        residual = self.Inputconv1x1(x)

        c1 = self.conv1_1(x)
        c2 = self.conv1_2(x)
        c3 = self.conv1_3(x)
        c4 = self.conv1_4(x)

        out = torch.cat([c1,c2,c3,c4],1)
        
        #adding the skip connection
        out += residual
        out = self.relu(out)

        return out


###############
class IncUNet (nn.Module):

    def __init__(self, in_shape,  num_classes):
        super(IncUNet, self).__init__()
        in_channels, height, width = in_shape
        #
        #self.L1 = IncResBlock(in_channels,64)

        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2,padding=1),
            IncResBlock(64,64))
            
        
        #
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(64, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            IncResBlock(128,128))
        #
        self.e2add = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128))
        #
        ##
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(128,256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            IncResBlock(256,256))
        #
        
        ##
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(256,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            IncResBlock(512,512))
        #
        self.e4add = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512)) 
        #
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            IncResBlock(512,512))
        #
        #
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512), 
            IncResBlock(512,512))
        #
        self.e6add = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512)) 
        #
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            IncResBlock(512,512))
        #
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(512))
        ##
        ##
        ##

        #
        # self.e8add = nn.Sequential(
        #     nn.LeakyReLU(0.2,inplace=True),
        #     nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1))
        #####################################################################
   
        #########################
        self.d1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            IncResBlock(512,512))
        #
        self.d2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            IncResBlock(512,512))
        #
        self.d3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            IncResBlock(512,512))
        #
        self.d4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            IncResBlock(512,512))

        #
        self.d5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            IncResBlock(256,256))
        #
        self.d6 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            IncResBlock(128,128))
        #
        self.d7 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(64),
            IncResBlock(64,64))
        #
        self.d8 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(64),
            #nn.ReLU())

        self.out_l = nn.Sequential(
            nn.Conv2d(64,num_classes,kernel_size=1,stride=1))
            #nn.ReLU()) # -------------------------- Might have to change this activation

    # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)
           
   
    def forward(self, x):

     
     

        #Image Encoder

        #### Encoder #####
        #print "Encoder"
        en1 = self.e1(x)
        #print "E1:",en1.size()
        en2 = self.e2(en1)
        en2add = self.e2add(en2)
        #print "E2:",en2.size()
        en3 = self.e3(en2add)
        #print "E3:",en3.size()
        en4 = self.e4(en3)
        en4add = self.e4add(en4)
        #print "E4:",en4.size()
        en5 = self.e5(en4add)
        #print "E5:",en5.size()
        en6 = self.e6(en5)
        en6add = self.e6add(en6)
        #print "E6:",en6.size()
        en7 = self.e7(en6add)
        #print "E7:",en7.size()
        en8 = self.e8(en7)
        #en8add = self.e8add(en8)


        #### Decoder ####
        #print "Decoder"
        de1_ = self.d1(en8)
        #print de1_.size()
        de1 = torch.cat([en7,de1_],1)
        #print de1.size()

        de2_ = self.d2(de1)
        #print de2_.size()
        de2 = torch.cat([en6add,de2_],1)
        #print de2.size()
        
        
        de3_ = self.d3(de2)
        #print de3_.size()
        de3 = torch.cat([en5,de3_],1)
        #print de3.size()
        
        de4_ = self.d4(de3)
        #print de4_.size()
        de4 = torch.cat([en4add,de4_],1)
        #print de4.size()
        
        
        de5_ = self.d5(de4)
        #print de5_.size()
        de5 = torch.cat([en3,de5_],1)
        #print de5.size()
        
        de6_ = self.d6(de5)
        #print de6_.size()
        de6 = torch.cat([en2add,de6_],1)
        #print de6.size()
        
        
        de7_ = self.d7(de6)
        #print de7_.size()
        de7 = torch.cat([en1,de7_],1)
        #print de7.size()
        de8 = self.d8(de7)
        #print de8.size()

        out_l_mask = self.out_l(de8)
        out_l_mask  = nn.functional.log_softmax(out_l_mask )

        return out_l_mask 

######################################################################################################

### Redidual Net

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #batch normalization
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = self.conv1x1(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        #adding the skip connection
        out += residual
        out = self.relu(out)

        return out

###############

###############
class ResUnet (nn.Module):

    def __init__(self, in_shape,  num_classes):
        super(ResUnet, self).__init__()
        in_channels, height, width = in_shape
        #
        #self.L1 = IncResBlock(in_channels,64)
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2,padding=1),
            ResBlock(64,64))
            
        
        #
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(64, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            ResBlock(128,128))
        #
        self.e2add = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128))
        #
        ##
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(128,256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            ResBlock(256,256))
        #
        
        ##
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(256,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        self.e4add = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512)) 
        #
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        #
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512), 
            ResBlock(512,512))
        #
        self.e6add = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512)) 
        #
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(512))
        ##
        ##
        ##

        #
        # self.e8add = nn.Sequential(
        #     nn.LeakyReLU(0.2,inplace=True),
        #     nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1))
        #####################################################################
   
        #########################
        self.d1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))

        #
        self.d5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            ResBlock(256,256))
        #
        self.d6 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            ResBlock(128,128))
        #
        self.d7 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(64),
            ResBlock(64,64))
        #
        self.d8 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(64),
            #nn.ReLU())

        self.out_l = nn.Sequential(
            nn.Conv2d(64,num_classes,kernel_size=1,stride=1))
            #nn.ReLU()) # -------------------------- Might have to change this activation

    # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)
           
   
    def forward(self, x):

     
     

        #Image Encoder

        #### Encoder #####
        #print "Encoder"
        en1 = self.e1(x)
        #print "E1:",en1.size()
        en2 = self.e2(en1)
        en2add = self.e2add(en2)
        #print "E2:",en2.size()
        en3 = self.e3(en2add)
        #print "E3:",en3.size()
        en4 = self.e4(en3)
        en4add = self.e4add(en4)
        #print "E4:",en4.size()
        en5 = self.e5(en4add)
        #print "E5:",en5.size()
        en6 = self.e6(en5)
        en6add = self.e6add(en6)
        #print "E6:",en6.size()
        en7 = self.e7(en6add)
        #print "E7:",en7.size()
        en8 = self.e8(en7)
        #en8add = self.e8add(en8)


        #### Decoder ####
        #print "Decoder"
        de1_ = self.d1(en8)
        #print de1_.size()
        de1 = torch.cat([en7,de1_],1)
        #print de1.size()

        de2_ = self.d2(de1)
        #print de2_.size()
        de2 = torch.cat([en6add,de2_],1)
        #print de2.size()
        
        
        de3_ = self.d3(de2)
        #print de3_.size()
        de3 = torch.cat([en5,de3_],1)
        #print de3.size()
        
        de4_ = self.d4(de3)
        #print de4_.size()
        de4 = torch.cat([en4add,de4_],1)
        #print de4.size()
        
        
        de5_ = self.d5(de4)
        #print de5_.size()
        de5 = torch.cat([en3,de5_],1)
        #print de5.size()
        
        de6_ = self.d6(de5)
        #print de6_.size()
        de6 = torch.cat([en2add,de6_],1)
        #print de6.size()
        
        
        de7_ = self.d7(de6)
        #print de7_.size()
        de7 = torch.cat([en1,de7_],1)
        #print de7.size()
        de8 = self.d8(de7)
        #print de8.size()

        out_l_mask = self.out_l(de8)
        
        out_l_mask  = nn.functional.softmax(out_l_mask)
    
        return out_l_mask 

############################################
######################################################################################################


#######################################33

### Redidual Net

class ResBlock2(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock2, self).__init__()
        #self.conv1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #batch normalization
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x #self.conv1x1(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        #adding the skip connection
        out += residual
        #out = self.relu(out)

        return out
############### Refine Net

class FuseBlock(nn.Module):
    """docstring for FuseBlock"""
    def __init__(self,  in_channels, out_channels): #, arg):
        super(FuseBlock, self).__init__()
        #in_channels, height, width = in_shape        #self.arg = arg
        
        self.image1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,padding=0),
            nn.ReLU(),
            IncResBlock(out_channels,out_channels), #IncResBlock2(out_channels,out_channels),
            nn.ReLU(),
            IncResBlock(out_channels,out_channels), #IncResBlock2(out_channels,out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1))
        self.fuse = nn.Sequential(
            nn.ReLU(),
            ResBlock2(out_channels,out_channels))

    def forward(self, x):

        img1 = self.image1(x)
        
        fusion = self.fuse(img1)

        return fusion


class RefineNet (nn.Module):

    def __init__(self, in_shape,  num_classes):
        super(RefineNet, self).__init__()
        in_channels, height, width = in_shape
        fuse_channels=8
        #
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels+fuse_channels, 64, kernel_size=4, stride=2,padding=1),
            ResBlock(64,64))
            
        
        #
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(64+fuse_channels, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            ResBlock(128,128))
        #

        #
        ##
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128+fuse_channels, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(128,256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            ResBlock(256,256))
        #
        
        ##
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(256+fuse_channels,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        #
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512+fuse_channels,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        #
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512+fuse_channels,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512), 
            ResBlock(512,512))
        #

        #
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512+fuse_channels,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(512))
        ##
        ##
        ##

        #
        # self.e8add = nn.Sequential(
        #     nn.LeakyReLU(0.2,inplace=True),
        #     nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1))
        #####################################################################
   
        #########################
        self.d1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))

        #
        self.d5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            ResBlock(256,256))
        #
        self.d6 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            ResBlock(128,128))
        #
        self.d7 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(64),
            ResBlock(64,64))
        #
        self.d8 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(64),
            #nn.ReLU())

        self.out_l = nn.Sequential(
            nn.Conv2d(64,num_classes,kernel_size=1,stride=1))
            #nn.ReLU()) # -------------------------- Might have to change this activation

    # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)

        self.fuseblock = FuseBlock(in_shape,8)

        self.DownSample =  nn.AvgPool2d(kernel_size=2, stride=2)

           
   
    def forward(self, x):

     
     

        #Image Encoder

        #### Encoder #####
        #print "Encoder"
        x1 = x
       
        fu1 = self.fuseblock(x1)
        en1 = self.e1(torch.cat([fu1,x],1))
        #print "E1:",en1.size()
        x2 = self.DownSample(x1)
        # y2 = self.DownSample(y1)
        fu2 = self.fuseblock(x2)
        en2 = self.e2(torch.cat([fu2,en1],1))
        #en2add = self.e2add(en2)
        #print "E2:",en2.size()
        x3 = self.DownSample(x2)
        # y3 = self.DownSample(y2)
        fu3 = self.fuseblock(x3)
        en3 = self.e3(torch.cat([fu3,en2],1))
        #print "E3:",en3.size()
        x4 = self.DownSample(x3)
        # y4 = self.DownSample(y3)
        fu4 = self.fuseblock(x4)
        en4 = self.e4(torch.cat([fu4,en3],1))
        #en4add = self.e4add(en4)
        #print "E4:",en4.size()
        x5 = self.DownSample(x4)
        # y5 = self.DownSample(y4)
        fu5 = self.fuseblock(x5)
        en5 = self.e5(torch.cat([fu5,en4],1))
        #print "E5:",en5.size()
        x6 = self.DownSample(x5)
        # y6 = self.DownSample(y5)
        fu6 = self.fuseblock(x6)
        en6 = self.e6(torch.cat([fu6,en5],1))
        #en6add = self.e6add(en6)
        #print "E6:",en6.size()
        x7 = self.DownSample(x6)
        # y7 = self.DownSample(y6)
        fu7 = self.fuseblock(x7)
        en7 = self.e7(torch.cat([fu7,en6],1))
        #print "E7:",en7.size()
        en8 = self.e8(en7)
        #en8add = self.e8add(en8)


        #### Decoder ####
        #print "Decoder"
        de1_ = self.d1(en8)
        #print de1_.size()
        de1 = torch.cat([en7,de1_],1)
        #print de1.size()

        de2_ = self.d2(de1)
        #print de2_.size()
        de2 = torch.cat([en6,de2_],1)
        #print de2.size()
        
        
        de3_ = self.d3(de2)
        #print de3_.size()
        de3 = torch.cat([en5,de3_],1)
        #print de3.size()
        
        de4_ = self.d4(de3)
        #print de4_.size()
        de4 = torch.cat([en4,de4_],1)
        #print de4.size()
        
        
        de5_ = self.d5(de4)
        #print de5_.size()
        de5 = torch.cat([en3,de5_],1)
        #print de5.size()
        
        de6_ = self.d6(de5)
        #print de6_.size()
        de6 = torch.cat([en2,de6_],1)
        #print de6.size()
        
        
        de7_ = self.d7(de6)
        #print de7_.size()
        de7 = torch.cat([en1,de7_],1)
        #print de7.size()
        de8 = self.d8(de7)
        #print de8.size()

        out_l_mask = self.out_l(de8)
        out_l_mask  = nn.functional.log_softmax(out_l_mask )

        return out_l_mask 

############################################


########################33333333333333


###############
class RefineNet2 (nn.Module):

    def __init__(self, in_shape,  num_classes):
        super(RefineNet2, self).__init__()
        in_channels, height, width = in_shape
        #
        #self.L1 = IncResBlock(in_channels,64)
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2,padding=1),
            ResBlock(64,64))
            
        
        #
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(64, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            ResBlock(128,128))
        #
        self.e2add = FuseBlock(128,128)
        #
        ##
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(128,256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            ResBlock(256,256))
        #
        
        ##
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(256,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        self.e4add =  FuseBlock(512,512)
        #
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        #
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512), 
            ResBlock(512,512))
        #
        self.e6add = FuseBlock(512,512) 
        #
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))
        #
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(512))
        ##
        ##
        ##

        #
        # self.e8add = nn.Sequential(
        #     nn.LeakyReLU(0.2,inplace=True),
        #     nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1))
        #####################################################################
   
        #########################
        self.d1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            ResBlock(512,512))
        #
        self.d4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            ResBlock(512,512))

        #
        self.d5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            ResBlock(256,256))
        #
        self.d6 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            ResBlock(128,128))
        #
        self.d7 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(64),
            ResBlock(64,64))
        #
        self.d8 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(64),
            #nn.ReLU())

        self.out_l = nn.Sequential(
            nn.Conv2d(64,num_classes,kernel_size=1,stride=1))
            #nn.ReLU()) # -------------------------- Might have to change this activation

    # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)
           
   
    def forward(self, x):

     
     

        #Image Encoder

        #### Encoder #####
        #print "Encoder"
        en1 = self.e1(x)
        #print "E1:",en1.size()
        en2 = self.e2(en1)
        en2add = self.e2add(en2)
        #print "E2:",en2.size()
        en3 = self.e3(en2add)
        #print "E3:",en3.size()
        en4 = self.e4(en3)
        en4add = self.e4add(en4)
        #print "E4:",en4.size()
        en5 = self.e5(en4add)
        #print "E5:",en5.size()
        en6 = self.e6(en5)
        en6add = self.e6add(en6)
        #print "E6:",en6.size()
        en7 = self.e7(en6add)
        #print "E7:",en7.size()
        en8 = self.e8(en7)
        #en8add = self.e8add(en8)


        #### Decoder ####
        #print "Decoder"
        de1_ = self.d1(en8)
        #print de1_.size()
        de1 = torch.cat([en7,de1_],1)
        #print de1.size()

        de2_ = self.d2(de1)
        #print de2_.size()
        de2 = torch.cat([en6,de2_],1)
        #print de2.size()
        
        
        de3_ = self.d3(de2)
        #print de3_.size()
        de3 = torch.cat([en5,de3_],1)
        #print de3.size()
        
        de4_ = self.d4(de3)
        #print de4_.size()
        de4 = torch.cat([en4,de4_],1)
        #print de4.size()
        
        
        de5_ = self.d5(de4)
        #print de5_.size()
        de5 = torch.cat([en3,de5_],1)
        #print de5.size()
        
        de6_ = self.d6(de5)
        #print de6_.size()
        de6 = torch.cat([en2,de6_],1)
        #print de6.size()
        
        
        de7_ = self.d7(de6)
        #print de7_.size()
        de7 = torch.cat([en1,de7_],1)
        #print de7.size()
        de8 = self.d8(de7)
        #print de8.size()

        out_l_mask = self.out_l(de8)
        out_l_mask  = nn.functional.log_softmax(out_l_mask )

        return out_l_mask 

############################################
######################################################################################################


#############################################################################################

class IncRefineNet2 (nn.Module):

    def __init__(self, in_shape,  num_classes):
        super(IncRefineNet2, self).__init__()
        in_channels, height, width = in_shape
        #
        #self.L1 = IncResBlock(in_channels,64)
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2,padding=1),
            IncResBlock(64,64))
            
        
        #
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(64, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            IncResBlock(128,128))
        #
        self.e2add = FuseBlock(128,128)
        #
        ##
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(128,256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            IncResBlock(256,256))
        #
        
        ##
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(256,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            IncResBlock(512,512))
        #
        self.e4add =  FuseBlock(512,512)
        #
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            IncResBlock(512,512))
        #
        #
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512), 
            IncResBlock(512,512))
        #
        self.e6add = FuseBlock(512,512) 
        #
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            IncResBlock(512,512))
        #
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv2d(512,512, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(512))
        ##
        ##
        ##

        #
        # self.e8add = nn.Sequential(
        #     nn.LeakyReLU(0.2,inplace=True),
        #     nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1))
        #####################################################################
   
        #########################
        self.d1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            IncResBlock(512,512))
        #
        self.d2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            IncResBlock(512,512))
        #
        self.d3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5),
            IncResBlock(512,512))
        #
        self.d4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(512),
            IncResBlock(512,512))

        #
        self.d5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            IncResBlock(256,256))
        #
        self.d6 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            IncResBlock(128,128))
        #
        self.d7 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(64),
            IncResBlock(64,64))
        #
        self.d8 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1))
            #nn.BatchNorm2d(64),
            #nn.ReLU())

        self.out_l = nn.Sequential(
            nn.Conv2d(64,num_classes,kernel_size=1,stride=1))
            #nn.ReLU()) # -------------------------- Might have to change this activation

    # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)
           
   
    def forward(self, x):

     
     

        #Image Encoder

        #### Encoder #####
        #print "Encoder"
        en1 = self.e1(x)
        #print "E1:",en1.size()
        en2 = self.e2(en1)
        en2add = self.e2add(en2)
        #print "E2:",en2.size()
        en3 = self.e3(en2add)
        #print "E3:",en3.size()
        en4 = self.e4(en3)
        en4add = self.e4add(en4)
        #print "E4:",en4.size()
        en5 = self.e5(en4add)
        #print "E5:",en5.size()
        en6 = self.e6(en5)
        en6add = self.e6add(en6)
        #print "E6:",en6.size()
        en7 = self.e7(en6add)
        #print "E7:",en7.size()
        en8 = self.e8(en7)
        #en8add = self.e8add(en8)


        #### Decoder ####
        #print "Decoder"
        de1_ = self.d1(en8)
        #print de1_.size()
        de1 = torch.cat([en7,de1_],1)
        #print de1.size()

        de2_ = self.d2(de1)
        #print de2_.size()
        de2 = torch.cat([en6,de2_],1)
        #print de2.size()
        
        
        de3_ = self.d3(de2)
        #print de3_.size()
        de3 = torch.cat([en5,de3_],1)
        #print de3.size()
        
        de4_ = self.d4(de3)
        #print de4_.size()
        de4 = torch.cat([en4,de4_],1)
        #print de4.size()
        
        
        de5_ = self.d5(de4)
        #print de5_.size()
        de5 = torch.cat([en3,de5_],1)
        #print de5.size()
        
        de6_ = self.d6(de5)
        #print de6_.size()
        de6 = torch.cat([en2,de6_],1)
        #print de6.size()
        
        
        de7_ = self.d7(de6)
        #print de7_.size()
        de7 = torch.cat([en1,de7_],1)
        #print de7.size()
        de8 = self.d8(de7)
        #print de8.size()

        out_l_mask = self.out_l(de8)
        out_l_mask  = nn.functional.log_softmax(out_l_mask )

        return out_l_mask 

############################################
######################################################################################################
