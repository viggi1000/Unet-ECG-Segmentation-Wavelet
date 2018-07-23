
#Architecture

import torch.nn as nn
import torch
from torch.autograd import Variable



class Sig2Sig(nn.Module):
    
    def __init__(self):
        super(Sig2Sig,self).__init__()
       
        filter_size = 3
        stride_size = 1
        
        self.block1d = nn.Sequential(nn.Conv2d(in_channels=47,
                                     out_channels=16,
                                     kernel_size=filter_size,
                                     stride=stride_size,
                                     padding=0
                                     ),nn.ReLU())
        
        self.block2d = nn.Sequential(nn.Conv2d(in_channels=16,
                                     out_channels=32,
                                     kernel_size=filter_size,
                                     stride=stride_size,
                                     padding=0
                                     ),
                        nn.BatchNorm1d(32),
                        nn.ReLU())
        
        self.block3d = nn.Sequential(nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=filter_size,
                                     stride=stride_size,
                                     padding=0
                                     ),
                        nn.BatchNorm1d(64),
                        nn.ReLU())
        
        self.block4d = nn.Sequential(nn.Conv2d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=filter_size,
                                     stride=stride_size,
                                     padding=0
                                     ),
                        nn.BatchNorm1d(128),
                        nn.ReLU())
        
    
        self.block4u = nn.Sequential(nn.ConvTranspose2d(in_channels=128,
                                                        out_channels=64,
                                                        kernel_size=filter_size,
                                                        stride=stride_size,
                                                        padding=0),
                                    nn.BatchNorm1d(64),
                                    nn.ReLU())
        
        self.block3u = nn.Sequential(nn.ConvTranspose2d(in_channels=128,
                                                        out_channels=32,
                                                        kernel_size=filter_size,
                                                        stride=stride_size,
                                                        padding=0),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU())
        
        
        self.block2u = nn.Sequential(nn.ConvTranspose2d(in_channels=64,
                                                        out_channels=16,
                                                        kernel_size=filter_size,
                                                        stride=stride_size,
                                                        padding=0),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU())
        
        
        self.block1u = nn.Sequential(nn.ConvTranspose2d(in_channels=32,
                                                        out_channels=4,
                                                        kernel_size=filter_size,
                                                        stride=stride_size,
                                                        padding=0))
        
    def forward(self,x):
        
        ## Downsample
#        print x.size()
        block1d_out = self.block1d(x)
#        print block1d_out.size()
        block2d_out = self.block2d(block1d_out)
#        print block2d_out.size()
        block3d_out = self.block3d(block2d_out)
#        print block3d_out.size()
        block4d_out = self.block4d(block3d_out)
#        print block4d_out.size()
        
        ## Upsample and skip connnection
        block4u_out = self.block4u(block4d_out)
#        print block4u_out.size()
        
        block3u_out_skip = torch.cat([block4u_out,block3d_out],dim=1)
        block3u_out = self.block3u(block3u_out_skip)
#        print block3u_out.size()
        
        block2u_out_skip = torch.cat([block3u_out,block2d_out],dim=1)
        block2u_out = self.block2u(block2u_out_skip)
#        print block2u_out.size()
        
        block1u_out_skip = torch.cat([block2u_out,block1d_out],dim=1)
        block1u_out = self.block1u(block1u_out_skip)
        #print(block1u_out.size())
        block1u_out = nn.functional.log_softmax(block1u_out,dim=0)
#        print block1u_out.size()
        
        return block1u_out
        
        
        
net = Sig2Sig()
#print net
#net = net.cuda()
#x   = Variable(torch.rand([1,1,150]).cuda())
#out = net(x)
##print out
        
        
