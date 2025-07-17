from argparse import ArgumentParser
import numpy as np
import torch
import random
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from unet import UNet
from threshold_select import ThresholdNumberSelector
from torch.utils.data import DataLoader
import os 
from dataload import NormalDataset, AnomalDataset, AnomalDataset_Test
from evaluate import evaluate_BLOGRR
import wandb
import kornia
from medpy.metric.binary import dc
import copy 

def get_config():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_model_path_lower', type=str, default='Save_model/lower')
    parser.add_argument('--save_model_path_upper', type=str, default='Save_model/upper')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--lr_schedule', type=str, default=False, help='Use learning rate schedule.')    
    parser.add_argument('--model_weight', type=str, default='')
    parser.add_argument('--eval', type=str, default=True, help='Evaluation mode')    
    parser.add_argument('--train_HCP_dir', type=str, default='Dataset/HCP_train_data/img')
    parser.add_argument('--train_in_house_img_dir', type=str, default='Dataset/In_house_data/img')
    parser.add_argument('--train_in_house_seg_dir', type=str, default='Dataset/In_house_data/seg')
    parser.add_argument('--test_img_dir', type=str, default='Dataset/final_test_data/img')
    parser.add_argument('--test_seg_dir', type=str, default='Dataset/final_test_data/seg')
    parser.add_argument('--sample_test_img', type=str, default='Dataset/sample_test_data/img')
    parser.add_argument('--sample_test_seg', type=str, default='Dataset/sample_test_data/seg')
    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
    parser.add_argument('--step', type=int, default=0, help='Record the training steps')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=10)
    # Model Hyperparameters
    parser.add_argument("--noise_res", type=float, default=16, help="noise resolution.")
    parser.add_argument("--noise_std", type=float, default=0.2, help="noise magnitude.")
    parser.add_argument('--img_channels', type=int, default=1, help='Image channels')
    parser.add_argument('--modality', type=str, default='MRI', help='the modality of the date')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--sigma', '-s', type=float, default=4, help='sigma of gaussian filter')
    return parser.parse_args()
config = get_config()

# set seed
random.seed(config.seed)
os.environ['PYTHONHASHSEED'] = str(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# set wandb
if config.eval == False:
    wandb.init(project="BLOGRR")

# load dataset
if config.eval == True:
    dataset_test = AnomalDataset_Test(config.test_img_dir, config.test_seg_dir)
    config.shuffle = False
    data_loader_test = DataLoader(dataset_test, batch_size=120, shuffle=False, num_workers=config.num_workers) 
else:
    dataset_HCP = NormalDataset(config.train_HCP_dir)
    dataset_in_house = AnomalDataset(config.train_in_house_img_dir, config.train_in_house_seg_dir)
    dataset_BraTS2021_test = AnomalDataset_Test(config.sample_test_img, config.sample_test_seg)
    config.shuffle = True
    data_loader_HCP = DataLoader(dataset_HCP, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers) 
    data_loader_in_house = DataLoader(dataset_in_house, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers) 
    data_loader_BraTS2021_test = DataLoader(dataset_BraTS2021_test, batch_size=120, shuffle=False, num_workers=config.num_workers) 

# load model
print("Initializing model...")
DAE = UNet(in_channels=3, n_classes=config.img_channels).to(config.device)
threshold_network = ThresholdNumberSelector(config).to(config.device)

if config.eval:
    DAE.load_state_dict(torch.load(config.model_weight, map_location=f'cuda:{config.device}')['lower_model_state_dict'])
    threshold_network.load_state_dict(torch.load(config.model_weight, map_location=f'cuda:{config.device}')['upper_model_state_dict'])
    print('Saved model loaded.')

# set lr and optimizer
optimizer_lower = torch.optim.Adam(DAE.parameters(), lr=config.lr, amsgrad=True, weight_decay=config.weight_decay)
optimizer_upper = torch.optim.Adam(threshold_network.parameters(), lr=config.lr*0.1, amsgrad=True, weight_decay=config.weight_decay)
lr_scheduler_lower = CosineAnnealingLR(optimizer=optimizer_lower, T_max=100)
lr_scheduler_upper = CosineAnnealingLR(optimizer=optimizer_upper, T_max=100)


def add_noise(input):
    ns = torch.normal(mean=torch.zeros(input.shape[0], input.shape[1], config.noise_res, config.noise_res),
                      std=config.noise_std).to(config.device)
    ns = F.interpolate(ns, size=config.image_size, mode='bilinear', align_corners=True)
    roll_x = random.choice(range(config.image_size))
    roll_y = random.choice(range(config.image_size))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
    if config.modality == 'MRI':
        mask = torch.cat([inp > inp.min() for inp in input]).unsqueeze(1)
        ns *= mask
    res = input + ns
    return res, ns


def train():
    epoch = 0
    best_dice_loss = 1000
    while True:
        epoch = epoch + 1

        # start lower loop training
        for input in data_loader_HCP:  
            DAE.train()
            threshold_network.eval()
            config.step += 1
            input = input.to(config.device)
            if epoch > 1:
                threshold = threshold_network(input)
            else:
                threshold = torch.full((input.shape[0], 1), -100.0)
            noisy_input, noise_tensor = add_noise(input)
            optimizer_lower.zero_grad()
            reconstruction = DAE(noisy_input)
            anomaly_map = torch.pow(reconstruction - input, 2).mean(1, keepdim=True)
            loss_mse = anomaly_map.mean()
            
            if epoch <= 1:
                threshold = threshold.unsqueeze(2).unsqueeze(3)
                threshold = threshold.expand(-1, -1, config.image_size, config.image_size)
            
            threshold = threshold.to(anomaly_map.device)
            anomaly_map = torch.where(anomaly_map < threshold, torch.tensor(0.0).to(anomaly_map.device), anomaly_map)
            loss_threshold = anomaly_map.mean()
            loss_sum = loss_mse + 0.5*loss_threshold   # We finally find that this loss function is good than that proposed in paper "loss_sum = loss_threshold"
            loss_sum.backward()
            optimizer_lower.step()     
            wandb.log({"lower_loss": loss_sum.item()})   

        # start upper loop training
        dice_loss = []
        for input, seg in data_loader_in_house: 
            input = input.to(config.device) 
            seg = seg.to(config.device)   
            DAE.eval()
            threshold_network.train()
            threshold = threshold_network(input)
            input_recon = DAE(input)
            anomaly_map = (input - input_recon).abs().mean(1, keepdim=True)
            anomaly_map = kornia.filters.gaussian_blur2d(anomaly_map, kernel_size=(5,5), sigma=(config.sigma, config.sigma))
            threshold = threshold.unsqueeze(2).unsqueeze(3)    
            threshold = threshold.expand(-1, -1, config.image_size, config.image_size)  
            X_output = anomaly_map - threshold
            X_output = torch.reciprocal(1 + torch.exp(-50 * X_output))
            optimizer_upper.zero_grad()
            loss_abnormal = threshold_network.Dice_loss_function(X_output, seg)
            loss_sum = loss_abnormal
            loss_sum.backward()
            optimizer_upper.step()
            dice_loss.append(loss_abnormal.item())
            wandb.log({"upper_loss": loss_sum.item()})
        average_dice_loss = sum(dice_loss) / len(dice_loss)
        wandb.log({"average_dice_loss": average_dice_loss})
        
        if config.lr_schedule:
            lr_scheduler_lower.step()
            lr_scheduler_upper.step()
        
        wandb.log({"upper_lr": lr_scheduler_upper.get_last_lr()[0]})
        wandb.log({"lower_lr": lr_scheduler_lower.get_last_lr()[0]})
        if epoch > 100:   # train 100 epochs
            return
        if (epoch + 1) % 2 == 0:
            DAE.eval()
            threshold_network.eval()
            dice = []
            with torch.no_grad():
                for input, seg in data_loader_BraTS2021_test:
                    input = input.to(config.device) 
                    seg = seg.to(config.device)
                    threshold = threshold_network(input)
                    input_recon = DAE(input)
                    anomaly_map = (input - input_recon).abs().mean(1, keepdim=True)
                    anomaly_map = kornia.filters.gaussian_blur2d(anomaly_map, kernel_size=(5,5), sigma=(config.sigma, config.sigma))
                    threshold = threshold.unsqueeze(2).unsqueeze(3)    
                    threshold = threshold.expand(-1, -1, config.image_size, config.image_size)
                    pred = (anomaly_map > threshold).float()
                    dice.append(dc(pred.cpu().numpy(), seg.cpu().numpy()))
            avg_dice = sum(dice) / len(dice)
            print(avg_dice)
            wandb.log({"test_avg_dice": avg_dice})

        if epoch > 3:
            if best_dice_loss > average_dice_loss:
                os.makedirs(config.save_model_path_lower, exist_ok=True)
                os.makedirs(config.save_model_path_upper, exist_ok=True)
                best_dice_loss = average_dice_loss
                model_name_lower = f"best_{epoch}_{best_dice_loss:.4f}.pth" 
                model_name_upper = f"best_{epoch}_{best_dice_loss:.4f}.pth" 
                save_path_lower = os.path.join(config.save_model_path_lower, model_name_lower)
                save_path_upper = os.path.join(config.save_model_path_upper, model_name_upper)
                torch.save(DAE.state_dict(), save_path_lower)  
                torch.save(threshold_network.state_dict(), save_path_upper) 


if __name__ == '__main__':
    if config.eval:
        print('Evaluating model...')
        evaluate = evaluate_BLOGRR(config, DAE, threshold_network)
        ## Visualize the results
        for image, seg in data_loader_test:
            image = image.to(config.device) 
            seg = seg.to(config.device) 
            evaluate.visual_result(image, seg)
    else:
        train()
