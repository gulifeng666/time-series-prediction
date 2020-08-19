
import argparse
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import LSTNet,MHA_Net,CNN,RNN
import importlib
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
from utils import *
from train_eval import predict
from train_eval import train, evaluate, makeOptimizer,muti_loss
from tensorboardX import SummaryWriter
#torch.cuda.set_device(0)
modelname = 'MHA_Net'
#modelname = 'LSTNet'
#modelname='RNN'
#params = [2,4,3,1]#rnn
params = [4,8]#mha
#params = [3,6,9,16]#cnn
#params = [512,256,32,64,128]#lstnet
writer = SummaryWriter(logsavepath)
windowsize = 800
classify_flag = True
reg_flag =False
point_flag = True
for i in range(len(params)):
        parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
        parser.add_argument('--data', type=str,default = '/root/code/data_analysis/lstnet/data/er.npz' ,help='location of the data file')
        parser.add_argument('--model', type=str, default=modelname, help='')
        parser.add_argument('--window', type=int, default=windowsize,help='window size')#input length
        parser.add_argument('--horizon', type=int, default=300)#predict length

        parser.add_argument('--hidRNN', type=int, default=300, help='number of RNN hidden units each layer')
        parser.add_argument('--rnn_layers', type=int, default=params[i], help='number of RNN hidden layers')

        parser.add_argument('--hidCNN', type=int, default=300, help='number of CNN hidden units (channels)')
        parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
        parser.add_argument('--highway_window', type=int, default=24, help='The window size of the highway component')
        parser.add_argument('-n_head', type=int, default=params[i])
        parser.add_argument('-d_k', type=int, default=64)
        parser.add_argument('-d_v', type=int, default=64)

        parser.add_argument('--clip', type=float, default=10.,help='gradient clipping')
        parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
        parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
        parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
        parser.add_argument('--seed', type=int, default=54321,help='random seed')
        parser.add_argument('--log_interval', type=int, default=2000, metavar='N',help='report interval')

        parser.add_argument('--cuda', type=str, default=True)
        parser.add_argument('--optim', type=str, default='adam')
        parser.add_argument('--amsgrad', type=str, default=True)
        parser.add_argument('--lr', type=float, default=0.00001)
        parser.add_argument('--skip', type=float, default=24)
        parser.add_argument('--hidSkip', type=int, default=params[i])
        parser.add_argument('--L1Loss', type=bool, default=True)
        parser.add_argument('--normalize', type=int, default=1)
        parser.add_argument('--output_fun', type=str, default='sigmoid')
        parser.add_argument('--classify_flag', type=bool, default=classify_flag)#控制是否分类
        parser.add_argument('--reg_flag', type=bool, default = reg_flag)#控制是否回归
        parser.add_argument('--save', type=str, default=os.path.join('model',
                                                                     modelname + '-' + str(params[i]) + '-' + str(
                                                                         windowsize) + str(classify_flag)+str(reg_flag)+'.pt'),
                            help='path to save the final model')
        parser.add_argument('--point_flag', type=bool, default=reg_flag)#点预测还是序列预测
        args = parser.parse_args()



        # Choose device: cpu or gpu
        #args.cuda = torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")

        # Reproducibility.
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        # Load data
        Data = Data_utility(args.data, 0.6, 0.2, device, args)
        # loss function
        if args.L1Loss:
            criterion = nn.L1Loss(size_average=False)
        else:
            criterion = nn.MSELoss(size_average=False)
        evaluateL2 = nn.MSELoss(size_average=False)
        evaluateL1 = nn.L1Loss(size_average=False)
        evaluateclassify = nn.BCELoss(size_average=False)


        if args.cuda:
            criterion = criterion.cuda()
            evaluateL1 = evaluateL1.cuda()
            evaluateL2 = evaluateL2.cuda()

        # Select model
        model = eval(args.model).Model(args, Data)
        model = nn.DataParallel(model, device_ids=[0,1,2])
        train_method = train
        eval_method = evaluate
        nParams = sum([p.nelement() for p in model.parameters()])
        print('number of parameters: %d' % nParams)
        if args.cuda:
            #model = nn.DataParallel(model)
            model = model.cuda()


        best_val = 10000000
        best_f1 = 0
        optim = makeOptimizer(model.parameters(), args)
        print_shape(Data)
        # While training you can press Ctrl + C to stop it.
        try:
            print('Training start')
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train_loss = train_method(Data, Data.train[0], Data.train[1], model, criterion,evaluateclassify,optim, args)
                val_loss, val_rae, val_corr,val_f1 = eval_method(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args)
                print('| end of epoch {:3d} | time used: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | f1  {:5.4f}'.
                        format( epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr,val_f1))
                writer.add_scalar(args.save+'/trainloss',train_loss,epoch)
                writer.add_scalar(args.save + '/valloss', val_loss, epoch)
                writer.add_scalar(args.save + '/valrae', val_rae, epoch)
                writer.add_scalar(args.save+'/valcorr',val_corr,epoch)
                writer.add_scalar(args.save + '/valf1', val_f1, epoch)
                if(args.classify_flag):
                  if val_f1>best_f1 :
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_f1 = val_f1
                else:
                    if val_loss > best_val:
                        with open(args.save, 'wb') as f:
                            torch.save(model, f)
                        best_val = val_loss
                if epoch % 10 == 0:
                    test_loss, test_rae, test_corr,test_f1 = eval_method(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,args)
                    writer.add_scalar(args.save + '/testloss', test_loss, epoch)
                    writer.add_scalar(args.save + '/testrae', test_rae, epoch)
                    writer.add_scalar(args.save + '/testcorr', test_corr, epoch)
                    writer.add_scalar(args.save + '/testf1', test_f1, epoch)
                    print("| test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | f1 {:5.4f}\n".format(test_loss, test_rae, test_corr,test_f1))
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # Load the best saved model.
        with open(args.save, 'rb') as f:
            model = torch.load(f,)
        test_acc, test_rae, test_corr,test_f1 = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,args)
        writer.add_scalar(args.save + '/testf1', test_f1, args.epochs + 1)
        print('Best model performance：')
        print("| test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test f1 {:5.4f}".format(test_acc, test_rae, test_corr,test_f1))
        predict(Data, Data.test[0], Data.test[1], model,args)
print(modelname)