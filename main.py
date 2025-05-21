import argparse
import random
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.w2v2_aasist import Model as w2v2_aasist
from networks.aasist import Model as aasist
from networks.beats_aasist import Model as beats_aasist 
from data_utils import genSpoof_list, ADD_Dataset, eval_to_score_file
import config

from tensorboardX import SummaryWriter

torch.autograd.set_detect_anomaly(True)

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    """
    Produces evaluation scores for a dataset and saves them to a file.
    """
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)
    model.eval()
    model.to(device)
    with open(save_path, 'w') as fh:  # Open file once in write mode
        with torch.no_grad():
            for batch_x, utt_id in tqdm(data_loader, total=len(data_loader)):
                batch_x = batch_x.to(device)
                batch_out = model(batch_x)
                batch_score = batch_out[:, 1].data.cpu().numpy().ravel()  # Extract spoof confidence scores
                
                for f, cm in zip(utt_id, batch_score):
                    fh.write(f'{f}|{cm}\n')  # Save file name and score
    print(f'Scores saved to {save_path}')


def train_epoch(train_loader, model, lr,optim, device, accumulation_steps=1):
    running_loss = 0
    
    num_total = 0.0
    
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    model.to(device)
    step_nums = 0
    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Training'):
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        batch_loss = batch_loss / accumulation_steps
        
        running_loss += (batch_loss.item() * batch_size)
       
        # optimizer.zero_grad()
        batch_loss.backward()
        # optimizer.step()
        # 累加到指定的 steps 后再更新参数
        if (step_nums+1) % accumulation_steps == 0:     
            optimizer.step()                    # 更新参数
            optimizer.zero_grad()               # 梯度清零

        step_nums += 1
       
    running_loss /= num_total
    
    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='baseline system')
    # Dataset
    parser.add_argument('--train_meta_json', type=str, 
                        default='tta/dev/train.json', 
                        help='metadata of training data.') 
    parser.add_argument('--dev_meta_json', type=str, 
                        default='tta/dev/valid.json', 
                        help='metadata of validation data.')
    parser.add_argument('--test_meta_json', type=str, 
                        default='tta/test/test_01.json', 
                        help='metadata of test data.')
    parser.add_argument('--protocols_path', type=str, 
                        default='./', 
                        help='Change with path to user\'s database protocols directory address.')
    # Hyperparameters
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--num_workers', type=int, default=8)
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--model', type=str, default='aasist',)
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--exp_id', type=str, default=0, 
                        help='Experiment id.')
    
    print('====== Begin ======')
    args = parser.parse_args()
    output_folder = args.protocols_path + f'/exps/exp_{args.exp_id}'
    if not os.path.exists(f'{output_folder}/ckpts'):
        os.makedirs(f'{output_folder}/ckpts')

    #make experiment reproducible
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    #define model saving path
    model_tag = '{}_{}_{}_{}_{}_{}'.format(
        args.model, args.loss, args.num_epochs, 
        args.batch_size, args.accumulation_steps, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join(f'{output_folder}/ckpts', model_tag)
    if args.eval_output is None:
        args.eval_output = os.path.join(model_save_path, 'eval_scores.txt')
    #set model save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
        
    #GPU device
    device = args.device                  
    print('Device: {}'.format(device))
    
    if args.model == 'w2v2_aasist':
        model = w2v2_aasist(args,device)
    elif args.model == 'aasist':
        aasist_config = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
        }
        model = aasist(aasist_config)
    elif args.model == 'beats_aasist':
        model = beats_aasist(args,device)
    else:
        print('Model not found')
        sys.exit()

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print('nb_params:',nb_params)
    # sys.exit()
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    #evaluation 
    if args.eval:
        print('====== Evaluation ======')
        d_label, file_eval = genSpoof_list( dir_meta = f'{config.metadata_json_file}/{args.test_meta_json}',is_train=True,is_eval=False)
        print('test data path: ', args.test_meta_json)
        print('no. of test trials',len(file_eval))
        eval_set=ADD_Dataset(args, list_IDs = file_eval, labels=d_label, is_eval=True)
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        eer = eval_to_score_file(args.eval_output, f'{config.metadata_json_file}/{args.test_meta_json}' )
        sys.exit()
   
    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta = f'{config.metadata_json_file}/{args.train_meta_json}',is_train=True,is_eval=False)
    # file_train = file_train[:256]
    print('train data path: ', args.train_meta_json)
    print('no. of training trials',len(file_train))
    
    train_set=ADD_Dataset(args,list_IDs = file_train,labels = d_label_trn)
    
    train_loader = DataLoader(train_set, batch_size = args.batch_size,
                              num_workers = args.num_workers, 
                              shuffle = True, drop_last = True)
    
    del train_set,d_label_trn

    # define dev (validation) dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  f'{config.metadata_json_file}/{args.dev_meta_json}',is_train=False,is_eval=False)
    # file_dev = file_dev[:256]
    print('validation data path: ', args.dev_meta_json)
    print('no. of validation trials',len(file_dev))
    
    dev_set = ADD_Dataset(args,list_IDs = file_dev,labels = d_label_dev)

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)

    del dev_set,d_label_dev
    
    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/exp_{}'.format(args.exp_id))
    
    print('====== Train ======')
    val_not_decrease_epochs = 0
    min_val_loss = 1e3
    for epoch in range(num_epochs):
        ## early stop
        if val_not_decrease_epochs == args.patience:
            break
        running_loss = train_epoch(train_loader, model, args.lr, 
                                   optimizer, device, args.accumulation_steps)
        val_loss = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('train_loss', running_loss, epoch)
        print('\n{0:.4f} - train_loss:{1:.4f} - val_loss:{2:.4f} '.format(epoch,
                                                   running_loss,val_loss))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
        if val_loss < min_val_loss:
            val_not_decrease_epochs = 0
            min_val_loss = val_loss
        else:
            val_not_decrease_epochs += 1