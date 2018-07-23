import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
from wavelets_pytorch_2.alltorch.wavelets import Morlet, Ricker, DOG, Paul
from wavelets_pytorch_2.alltorch.transform import WaveletTransformTorch #WaveletTransform, 


from tqdm import tqdm
import argparse
import moment
import os
import sys

#from loader import CustomDataset,Rescale,ToTensor
from unet_network import UNet #ResUnet #RefineNet2 #IncUNet # UNetResAdd_2 #UNetResMul ResUNet#   #Unet UNetRes


def visual(label_g):#,cwt):
    fps = 150
    dt  = 1.0/fps
    dt1 = 1.0/fps
    dj  = 0.125
    unbias = False
    batch_size = 32
    #wavelet = Morlet(w0=2)
    wavelet = Paul(m=8)
    x = np.linspace(0, 1, num=150)
    dt = 1
    scales = np.load('./wavelet_dataset/scales.npy')
    print(label_g.shape)
    scales_label_torch = torch.from_numpy(scales).type(torch.FloatTensor)
    power_label_torch = torch.from_numpy(label_g.data.cpu().numpy()).type(torch.FloatTensor)
    # print(power_label_torch.size(),cwt.size())
    wa_label_torch = WaveletTransformTorch(dt, dj, wavelet, unbias=unbias, cuda=True)
    label_recon = wa_label_torch.reconstruction(power_label_torch,scales_label_torch,cwt_n=power_label_torch[0])

    return label_recon

if __name__ == "__main__":

    '''
    parser = argparse.ArgumentParser('Tool to perform segmentation with input and mask images')
    parser.add_argument(
        '--img_dir',
        required = True,
        type = str,
        help = 'Provide the path which contains image files'
    )

    parser.add_argument(
        '--mask_dir',
        required = True,
        type = str,
        help = 'Provide the path which contains mask files'
    )
    '''
    # parser.add_argument(
    #     '--batch_size',
    #     default = 1,
    #     type = int 
    # )

    # parser.add_argument(
    #     '--no_epoch',
    #     default = 25,
    #     type = int
    # )

    #img_dir = 'A/train'

    #mask_dir = 'B/train'
    #device = torch.device("cuda:0") 
    #torch.cuda.set_device(0)
    BATCH_SIZE = 5
    EPOCH = 70
    #C,H,W = 3,256,256
    C,H,W = 2,47,150
    C_depth = 2
    learn_rate = 0.001
    pretrained = True
   
    start_time =  moment.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = './logfiles/' + start_time + '.txt'
    log_data = open(log_file,'w+')
    model_path = './models/' + start_time + '/'
    os.mkdir(model_path)
    Exp_name = 'ECG segmentation' #'DristiData_Disc_NoDepth_RefineNet2_1' #'OrigaData_Disc_NoDepth_RefineNet2_1' #'OrigaData_Disc_NoDepth_RefineNet2_1' #'OrigaData_Disc_NoDepth_IncRefineNet2_1'
    SETTINGS = 'Epoch %d, LR %f, BATCH_SIZE %d \n'%(EPOCH,learn_rate,BATCH_SIZE)
    log_data.write(SETTINGS)


    if (pretrained == True):
        model = UNet(in_shape=(C,H,W),num_classes=2)
        # model = model.load_state_dict(torch.load('./models/2018-07-05_14-26-27/Epoch17.pt'))
        model = torch.load('./models/2018-07-06_13-48-32/Epoch3.pt')
    else:
        model = UNet(in_shape=(C,H,W),num_classes=2)
   
    #--------- load pretrained model if necessary
    # model.load_state_dict(torch.load('./Checkpoints/RimoneCkpt/RimOneData_Disc_NoDepth_RefineNet2_1_200_epch.pth'))

    #model = model.to(device)
    model = model.cuda()
    train_dat = torch.load('wavelet_dataset/train_dat_cwt.pt')
    test_dat = torch.load('wavelet_dataset/test_dat_cwt.pt')
    #transformed_dataset_train = CustomDataset(img_dir=img_dir,mask_dir=mask_dir,transform=transforms.Compose([Rescale(256), ToTensor()]))     
    train_x = train_dat[:,:2]#torch.stack([train_dat[:,0],train_dat[:,1]],1)#train_dat[:,:1] 
    train_y = train_dat[:,2:]
    #cwt_n =  #torch.load('wavelet_dataset/train_cwt.pt')
    print("Train X:", train_x.shape)
    print("Train Y:", train_y.shape)
    
    train_set = data_utils.TensorDataset(train_x, train_y)
    
    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # momentum parameter for Adam
    momentum = 0.9

    # weight decay
    weightDecay = 0.005

    optimizer = optim.SGD(model.parameters(), lr = learn_rate,momentum = momentum) 
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)
    criterion = nn.L1Loss()

   # criterion = nn.CrossEntropyLoss()
   # criterion = nn.NLLLoss2d()
    num_epochs = 70
    y_train = {}

    for epoch in tqdm(range(num_epochs)):
        
        print ('Epoch {}/{}'.format(epoch,num_epochs-1))
        print ('-'*10)

        model.train()

        running_loss = 0.0
        step = 0
        
        for  data,label in trainloader:
            step+=1
            inputs = data
            labels = label#.squeeze(1)
            cwt = data[:,1]
            if (type(criterion)==type(nn.MSELoss()) or type(criterion)==type(nn.L1Loss())): 
                labels = labels.type(torch.FloatTensor)
            else: 
                labels = labels.type(torch.LongTensor)
            # inputs = inputs.to(device)#Variable(inputs.cuda())
            # labels = labels.to(device)#Variable(labels.cuda())
            inputs,labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)#.squeeze(1)
            # print('Data:', inputs.size())
            # print('Inputs:', inputs.size())
            # print('Labels:', labels.size())
            # print('cwt:', cwt.size())
            # print('Output:', outputs.size())             
            # break
            loss = criterion(outputs,labels) 
            
            # print('Image:', inputs.size())
            # print('Labels:', labels.size())
            optimizer.zero_grad()
            # with torch.set_grad_enable(True):       
            #print(lo)
            loss.backward() 
            optimizer.step()
            if step % 10 == 0:
                #print(outputs.data.cpu().numpy().shape)
                y_train['pred'] = np.argmax(outputs.data.cpu().numpy().T,axis=1)
                y_train['true'] = labels.data.cpu().numpy()
               # print (y_train['true'].shape)
               # print (y_train['pred'].shape)
                y_train['inp']  = inputs.data.cpu().numpy().T

                #print(labels[0].size())#,labels.size())
               #.squeeze(0)
                #cw = cwt[0].squeeze(0)
                # print(lab.shape)
                # print(labels[0].shape)#,labels[0].shape)
                # label_new = visual(lab)#,inputs)
                # label_orig = visual(labels[0])#[0],cw)
                # print('Label_new:',label_new.shape)
                # print('Label_orig:',label_orig.shape)
                update = 'Epoch: ', epoch, '| step : %d' %step,  ' | train loss : %.6f' % loss.data[0] # , '| test accuracy: %.4f' % acc
                update_str = [str(i) for i in update]
                update_str = ''.join(update_str)	
                print (update_str)
                if (epoch>30 and pretrained ):
                    lab = outputs[0]
                    ecg_orig =  visual(inputs[0])
                    label_new = visual(lab)#,inputs)
                    label_orig = visual(labels[0])#[0],cw)
                    print('Label_new:',label_new.shape)
                    print('Label_orig:',label_orig.shape)
                    fig, ax = plt.subplots(3, 1, figsize=(12,6))
                    ax = ax.flatten()
                    ax[0].plot(ecg_orig)
                    ax[0].set_title(r'ECG')
                    ax[0].set_xlabel('Samples')
                    ax[1].plot(label_orig)
                    ax[1].set_title(r'Label source')
                    ax[1].set_xlabel('Samples')
                    ax[2].plot(label_new)
                    ax[2].set_title(r'Label U-net')
                    plt.tight_layout()
                    plt.show()
                    ax[2].set_xlabel('Samples')
                    # plt.plot(label_new)
                    # plt.show()
                log_data.write(update_str + '\n')
        print ("Saving the model" )
        torch.save(model,model_path+'Epoch'+str(epoch)+'.pt')
 