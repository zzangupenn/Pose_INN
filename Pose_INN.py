import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import os

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
from efficientnet_pytorch import EfficientNet
import torchvision
import time
from torch_utils import Conv2DLayer, TransConv2DLayer, PositionalEncoding, PositionalEncoding_torch
from rotation_utils import euler_2_matrix_sincos, GeodesicLoss, get_orient_err, get_posit_err
from pytorch3d import transforms

EXP_NAME = 'pose_inn_kings'
SCENE = 'KingsCollege/'
DATA_DIR = 'data/'
DATAFILE = '50k_train_w_render.npz'

MOVE_DATA_TO_DEVICE = 1
INSTABILITY_RECOVER = 1
USE_MIX_PRECISION_TRAINING = 0
CONTINUE_TRAINING = 0
TRANSFER_TRAINING = 0
TRANSFER_EXP_NAME = ''
RE_PROCESS_DATA = 1

BATCHSIZE = int(200)
N_DIM = int(66)
COND_DIM = 6
COND_OUT_DIM = 6
LR = 5e-4

    
class Encoder(nn.Module):
    def __init__(self, device, model_name, pretrained=True, latent_dim=60):
        super(Encoder, self).__init__()
        self.backbone = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(1280, latent_dim)
        self.fc_var = nn.Linear(1280, latent_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
    
    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        sigma = torch.exp(log_var)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        z = mu + sigma * self.N.sample(mu.shape)
        return z
    

class Decoder(nn.Module):
    def __init__(self, latent_dim=60):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 1280)
        self.bn = nn.BatchNorm1d(1280)
        self.relu = nn.ReLU()
        self.trans_conv1 = TransConv2DLayer(20, 40, kernel_size=4, stride=2, padding=1, add_layer=1, add_layer_kernel_size=3)
        self.trans_conv2 = TransConv2DLayer(40, 40, kernel_size=4, stride=2, padding=1, add_layer=1, add_layer_kernel_size=3)
        self.trans_conv3 = Conv2DLayer(40, 40, kernel_size=3, stride=1, padding=1, add_layer=1, add_layer_kernel_size=3)
        self.trans_conv4 = Conv2DLayer(40, 40, kernel_size=3, stride=1, padding=1, add_layer=1, add_layer_kernel_size=3)
        self.trans_conv5 = TransConv2DLayer(40, 20, kernel_size=4, stride=2, padding=1, add_layer=1, add_layer_kernel_size=3)
        self.trans_conv6 = TransConv2DLayer(20, 3, kernel_size=4, stride=2, padding=1, add_layer=1, add_layer_kernel_size=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(-1, 20, 8, 8)
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        x = self.trans_conv3(x)
        x = self.trans_conv4(x)
        x = self.trans_conv5(x)
        x = self.trans_conv6(x)
        x = self.sigmoid(x)
        return x


class VAE(nn.Module):
    def __init__(self, device):
        super(VAE, self).__init__()
        self.encoder = Encoder(device, model_name='efficientnet-b0', pretrained=True, latent_dim=60).to(device)
        self.decoder = Decoder(latent_dim=60).to(device)
    

class Local_INN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = self.build_inn(device)
        
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for param in self.trainable_parameters:
            param.data = 0.05 * torch.randn_like(param)
        self.vae = VAE(device)
        
    def build_inn(self, device):

        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, 1024), nn.ReLU(),
                                    nn.Linear(1024, c_out))

        nodes = [InputNode(N_DIM, name='input')]
        for k in range(4):
            nodes.append(Node(nodes[-1],
                                GLOWCouplingBlock,
                                {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                                name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                                PermuteRandom,
                                {'seed': k},
                                name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))
            
        return ReversibleGraphNet(nodes, verbose=False).to(device)
    
    def forward(self, x):
        return self.model(x)
    
    def reverse(self, y_rev):
        return self.model(y_rev, rev=True)
    

class torchDataset(torch.utils.data.Dataset):
    def __init__(self, data, device):
        self.pose_array = torch.from_numpy(data['pose_array']).type('torch.FloatTensor').to(device)
        print('pose_array.shape', self.pose_array.shape)
        self.matrix_array = euler_2_matrix_sincos(self.pose_array[:, 6:12], 'ZXY').to(device)
        print('matrix_array', self.matrix_array.shape)
        
        self.img_array = np.transpose(data['img_array'], (0, 3, 1, 2))
        self.img_array = torch.from_numpy(self.img_array).type('torch.FloatTensor')
        self.img_array = torchvision.transforms.Resize((128, 128)).forward(self.img_array).to(device)
        self.img_array = self.img_array / 255
        print('img_array.shape', self.img_array.shape)
        
        # self.img_render = np.transpose(data['imgs_render'], (0, 3, 1, 2))
        # self.img_render = torch.from_numpy(self.img_render).type('torch.FloatTensor')
        # self.img_render = torchvision.transforms.Resize((512, 512)).forward(self.img_render)
        # self.img_render = self.img_render / 255
        # print('img_render.shape', self.img_render.shape)

    def __len__(self):
        return self.pose_array.shape[0]
    
    def __getitem__(self, index):
        # if index >= self.img_render.shape[0]:
        return self.img_array[index], self.pose_array[index], self.img_array[index], self.matrix_array[index]
        # else:
        #     return self.img_array[index], self.pose_array[index], self.img_render[index], self.matrix_array[index]



def main():
    from trainer import Trainer
    from tqdm import trange
    from torch.utils.tensorboard import SummaryWriter
    from utils import ConfigJSON, DataProcessor
    writer = SummaryWriter('results/tensorboard/' + EXP_NAME)
    device = torch.device('cuda:1')
    
    print("EXP_NAME", EXP_NAME)
    if not os.path.exists('results/' + EXP_NAME + '/'):
            os.mkdir('results/' + EXP_NAME + '/')
    with open(__file__, "r") as src:
        with open('results/' + EXP_NAME + '/' + EXP_NAME + '.py', "w") as tgt:
            tgt.write(src.read())
    with open('results/' + EXP_NAME + '/' + EXP_NAME + '.txt', "a") as tgt:
        tgt.writelines(EXP_NAME + '\n')
            
    if os.path.exists('train_data.npz') and not RE_PROCESS_DATA:
        train_data = np.load('train_data.npz')
        c = ConfigJSON()
        c.load_file('train_data.json')
        c.save_file('results/' + EXP_NAME + '/' + EXP_NAME + '.json')
    else:
        data_dir = DATA_DIR + SCENE
        dataset = np.load(data_dir + '50k_train_w_render.npz')
        train_imgs = dataset['train_imgs']
        # train_imgs_render = dataset['train_imgs_render']
        # test_imgs_render = dataset['test_imgs_render']
        train_poses = dataset['train_poses']
        test_imgs = dataset['test_imgs']
        test_poses = dataset['test_poses']
        
        print('Data loaded.')
        dp = DataProcessor()
        c = ConfigJSON()
        pose_array = train_poses
        pose_array[:, 0], c.d['normalization_tx'] = dp.data_normalize(pose_array[:, 0])
        pose_array[:, 1], c.d['normalization_ty'] = dp.data_normalize(pose_array[:, 1])
        pose_array[:, 2], c.d['normalization_tz'] = dp.data_normalize(pose_array[:, 2])
        c.d['normalization_rz'] = [np.pi * 2, 0]
        c.d['normalization_rx'] = [np.pi * 2, 0]
        c.d['normalization_ry'] = [np.pi * 2, 0]
        pose_array[:, 3] = dp.runtime_normalize(dp.two_pi_warp(pose_array[:, 3]), c.d['normalization_rz'])
        pose_array[:, 4] = dp.runtime_normalize(dp.two_pi_warp(pose_array[:, 4]), c.d['normalization_rx'])
        pose_array[:, 5] = dp.runtime_normalize(dp.two_pi_warp(pose_array[:, 5]), c.d['normalization_ry'])
        c.save_file('results/' + EXP_NAME + '/' + EXP_NAME + '.json')
        c.save_file('train_data.json')
        
        encoded_pos = []
        matrix_array = []
        positional_encoding = PositionalEncoding(L = int(N_DIM / 12))
        p_encoding_t = PositionalEncoding_torch(L = int(N_DIM / 12), device=device)
        for ind_0 in trange(pose_array.shape[0]):
            encoded_data = []
            for k in range(6):
                if k >= 3:
                    sine_part, cosine_part = positional_encoding.encode_even(pose_array[ind_0, k])
                else:
                    sine_part, cosine_part = positional_encoding.encode(pose_array[ind_0, k])
                encoded_data.append(sine_part)
                encoded_data.append(cosine_part)
            encoded_data = np.array(encoded_data)
            encoded_data = encoded_data.flatten('F')
            new_data = np.concatenate([encoded_data, pose_array[ind_0, :]])
            encoded_pos.append(new_data)
        pose_array = np.array(encoded_pos)
        train_data = {'img_array':train_imgs, 'pose_array':pose_array}
        
        pose_array = test_poses
        pose_array[:, 0] = dp.runtime_normalize(pose_array[:, 0], c.d['normalization_tx'])
        pose_array[:, 1] = dp.runtime_normalize(pose_array[:, 1], c.d['normalization_ty'])
        pose_array[:, 2] = dp.runtime_normalize(pose_array[:, 2], c.d['normalization_tz'])
        pose_array[:, 3] = dp.runtime_normalize(dp.two_pi_warp(pose_array[:, 3]), c.d['normalization_rz'])
        pose_array[:, 4] = dp.runtime_normalize(dp.two_pi_warp(pose_array[:, 4]), c.d['normalization_rx'])
        pose_array[:, 5] = dp.runtime_normalize(dp.two_pi_warp(pose_array[:, 5]), c.d['normalization_ry'])
        
        encoded_pos = []
        positional_encoding = PositionalEncoding(L = int(N_DIM / 12))
        for ind_0 in trange(pose_array.shape[0]):
            encoded_data = []
            for k in range(6):
                if k >= 3:
                    sine_part, cosine_part = positional_encoding.encode_even(pose_array[ind_0, k])
                else:
                    sine_part, cosine_part = positional_encoding.encode(pose_array[ind_0, k])
                encoded_data.append(sine_part)
                encoded_data.append(cosine_part)
            encoded_data = np.array(encoded_data)
            encoded_data = encoded_data.flatten('F')
            new_data = np.concatenate([encoded_data, pose_array[ind_0, :]])
            encoded_pos.append(new_data)
        pose_array = np.array(encoded_pos)
        test_data = {'img_array':test_imgs, 'pose_array':pose_array}
        
        
    
    train_set = torchDataset(train_data, device)
    test_set = torchDataset(test_data, device)
    if MOVE_DATA_TO_DEVICE:
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCHSIZE, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCHSIZE, shuffle=True, drop_last=False)
    else: 
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCHSIZE, shuffle=True, pin_memory=True, num_workers=5, drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCHSIZE, shuffle=True, pin_memory=True, num_workers=5)
    
    
    
    # cond_noise = np.array(COND_NOISE) # m, rad
    # cond_noise[0] /= c.d['normalization_x'][0]
    # cond_noise[1] /= c.d['normalization_y'][0]
    # cond_noise[2] /= np.pi * 2
    # cond_noise = torch.from_numpy(cond_noise).type('torch.FloatTensor').to(device)
    
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    geodesic_loss = GeodesicLoss()
    p_encoding_t = PositionalEncoding_torch(1, device)
    
    ### Training
    trainer = Trainer(EXP_NAME, 500, 0.0001, device,
                      LR, [300], 0.1, 'exponential',
                      INSTABILITY_RECOVER, 3, 0.99, USE_MIX_PRECISION_TRAINING)

    model = Local_INN(device)
    model.to(device)
    if CONTINUE_TRAINING:
        ret = trainer.continue_train_load(model, path='results/' + EXP_NAME + '/', transfer=TRANSFER_TRAINING)
        if ret is not None:
            print('Continue Training')
            model = ret
            
    if TRANSFER_TRAINING:
        ret = trainer.continue_train_load(model, path='results/' + TRANSFER_EXP_NAME + '/', transfer=TRANSFER_TRAINING, transfer_exp_name=TRANSFER_EXP_NAME)
        if ret is not None:
            print('Transfer Training')
            model = ret
    
    current_lr = LR
    optimizer = torch.optim.Adam(model.trainable_parameters, lr=current_lr)
    # optimizer.add_param_group({"params": model.cond_net.parameters(), "lr": current_lr})
    optimizer.add_param_group({"params": model.vae.encoder.parameters(), "lr": current_lr})
    optimizer.add_param_group({"params": model.vae.decoder.parameters(), "lr": current_lr})
    n_hypo = 20
    epoch_time = 0
    mix_precision = False
    scaler = GradScaler(enabled=mix_precision)
    
    while(not trainer.is_done()):
        epoch = trainer.epoch
        epoch_info = np.zeros(7)
        epoch_info_extra = np.zeros(9)
        epoch_info[3] = epoch
        epoch_time_start = time.time()
        model.train()
        model.vae.encoder.train()
        model.vae.decoder.train()

        if USE_MIX_PRECISION_TRAINING:
            scaler = GradScaler(enabled=mix_precision)
        
        trainer_lr = trainer.get_lr()
        if trainer_lr != current_lr:
            current_lr = trainer_lr
            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.param_groups[1]['lr'] = current_lr
            optimizer.param_groups[2]['lr'] = current_lr
            # optimizer.param_groups[3]['lr'] = current_lr
        for img, pose, img2, rot_matrix in train_loader:
            if not MOVE_DATA_TO_DEVICE:
                img = img.to(device)
                img2 = img2.to(device)
                pose = pose.to(device)
                rot_matrix = rot_matrix.to(device)
            optimizer.zero_grad()
            with autocast(enabled=mix_precision):
                
                x_hat_gt = pose[:, :N_DIM]
                y_gt = img
            
                y_hat_vae = torch.empty_like(x_hat_gt, device=device)
                y_hat_vae[:, :-6] = model.vae.encoder.forward(y_gt)
                y_hat_inn, _ = model(x_hat_gt)
                y_inn = model.vae.decoder.forward(y_hat_inn[:, :-6])
                y_vae = model.vae.decoder.forward(y_hat_vae[:, :-6])
                vae_kl_loss = model.vae.encoder.kl * 0.00001
                vae_recon_loss = l1_loss(img, y_vae)
                inn_recon_loss = l1_loss(img, y_inn)
                y_hat_inn_loss = l1_loss(y_hat_inn[:, :-6], y_hat_vae[:, :-6])
                loss_forward = vae_recon_loss + inn_recon_loss + vae_kl_loss + y_hat_inn_loss
                epoch_info[0] += loss_forward.item()
            
            scaler.scale(loss_forward).backward(retain_graph=True)

            with autocast(enabled=mix_precision):
                y_hat_vae[:, -6:] = 0
                x_hat_0, _ = model.reverse(y_hat_vae)
                
                result_angles = torch.zeros((x_hat_gt.shape[0], 3)).to(device)
                result_angles[:, 0] = p_encoding_t.batch_decode_even(x_hat_0[:, 6], x_hat_0[:, 7])
                result_angles[:, 1] = p_encoding_t.batch_decode_even(x_hat_0[:, 8], x_hat_0[:, 9])
                result_angles[:, 2] = p_encoding_t.batch_decode_even(x_hat_0[:, 10], x_hat_0[:, 11])
                
                rot_loss = geodesic_loss(transforms.euler_angles_to_matrix(result_angles * 2 * np.pi, 'ZXY'), rot_matrix)
                epoch_info_extra[8] += rot_loss.item()
                
                loss_reverse = l1_loss(x_hat_0[:, :6], x_hat_gt[:, :6])
                epoch_info[2] += loss_reverse.item()
                loss_reverse += rot_loss
                
                batch_size = y_gt.shape[0]
                z_samples = torch.cuda.FloatTensor(n_hypo, batch_size, 6, device=device).normal_(0., 1.)
                y_hat = y_hat_vae[None, :, :-6].repeat(n_hypo, 1, 1)
                y_hat_z_samples = torch.cat((y_hat, z_samples), dim=2).view(-1, N_DIM)
                x_hat_i = model.reverse(y_hat_z_samples)[0].view(n_hypo, batch_size, N_DIM)
                
                x_hat_i_loss = torch.mean(torch.min(torch.mean(torch.abs(x_hat_i[:, :, :6] - x_hat_gt[:, :6]), dim=2), dim=0)[0])
                result_angles = torch.zeros((n_hypo, BATCHSIZE, 3)).to(device)
                result_angles[:, :, 0] = p_encoding_t.batch_decode_even(x_hat_i[:, :, 6], x_hat_i[:, :, 7])
                result_angles[:, :, 1] = p_encoding_t.batch_decode_even(x_hat_i[:, :, 8], x_hat_i[:, :, 9])
                result_angles[:, :, 2] = p_encoding_t.batch_decode_even(x_hat_i[:, :, 10], x_hat_i[:, :, 11])
                x_hat_i_rot_losses = []
                for ind in range(n_hypo):
                    new_loss = geodesic_loss(transforms.euler_angles_to_matrix(result_angles[ind, :, :] * 2 * np.pi, 'ZXY'), rot_matrix)
                    x_hat_i_rot_losses.append(new_loss)
                
                epoch_info_extra[8] += torch.min(torch.stack(x_hat_i_rot_losses)).item()
                loss_reverse += x_hat_i_loss + torch.min(torch.stack(x_hat_i_rot_losses))
                epoch_info[1] += loss_reverse
                
            scaler.scale(loss_reverse).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
            torch.nn.utils.clip_grad_norm_(model.vae.encoder.parameters(), 8)
            torch.nn.utils.clip_grad_norm_(model.vae.decoder.parameters(), 8)

            scaler.step(optimizer)
            scaler.update()
            
            epoch_info_extra[0] += vae_recon_loss.item()
            epoch_info_extra[1] += vae_kl_loss.item()
            epoch_info_extra[2] += y_hat_inn_loss.item()
            epoch_info_extra[3] += inn_recon_loss.item()

            
        epoch_info[:3] /= len(train_loader)
        epoch_info_extra[:4] /= len(train_loader)
        epoch_info_extra[8] /= len(train_loader)
        epoch_time = (time.time() - epoch_time_start)
        remaining_time = (trainer.max_epoch - epoch) * epoch_time / 3600
        
        
        # Testing
        model.eval()
        model.vae.encoder.eval()
        model.vae.decoder.eval()
        epoch_info_5 = []
        epoch_info_6 = []
        for img, pose, img2, _ in test_loader:
            if not MOVE_DATA_TO_DEVICE:
                img = img.to(device)
                img2 = img2.to(device)
                pose = pose.to(device)
                rot_matrix = rot_matrix.to(device)
            with torch.no_grad():
                x_hat_gt = pose[:, :N_DIM]
                # cond = torch.zeros((pose.shape[0], COND_DIM), device=device)
                y_gt = img
                
                y_hat_vae = torch.empty_like(x_hat_gt, device=device)
                y_hat_vae[:, :-6] = model.vae.encoder.forward(y_gt)
                y_hat_inn, _ = model(x_hat_gt)
                y_inn = model.vae.decoder.forward(y_hat_inn[:, :-6])
                y_vae = model.vae.decoder.forward(y_hat_vae[:, :-6])
                vae_kl_loss = model.vae.encoder.kl * 0.00001
                vae_recon_loss = l1_loss(img, y_vae)
                inn_recon_loss = l1_loss(img, y_inn)
                y_hat_inn_loss = l1_loss(y_hat_inn[:, :-6], y_hat_vae[:, :-6])
                loss_forward = vae_recon_loss + inn_recon_loss + vae_kl_loss + y_hat_inn_loss
                epoch_info[4] += loss_forward.item()
                
                y_hat_vae[:, -6:] = 0
                x_hat_0, _ = model.reverse(y_hat_vae)
                
                result_posit = torch.zeros((x_hat_gt.shape[0], 3)).to(device)
                result_posit[:, 0] = dp.de_normalize(p_encoding_t.batch_decode(x_hat_0[:, 0], x_hat_0[:, 1]), c.d['normalization_tx'])
                result_posit[:, 1] = dp.de_normalize(p_encoding_t.batch_decode(x_hat_0[:, 2], x_hat_0[:, 3]), c.d['normalization_ty'])
                result_posit[:, 2] = dp.de_normalize(p_encoding_t.batch_decode(x_hat_0[:, 4], x_hat_0[:, 5]), c.d['normalization_tz'])
                gt_posit = torch.zeros((x_hat_gt.shape[0], 3)).to(device)
                gt_posit[:, 0] = dp.de_normalize(pose[:, -6], c.d['normalization_tx'])
                gt_posit[:, 1] = dp.de_normalize(pose[:, -5], c.d['normalization_ty'])
                gt_posit[:, 2] = dp.de_normalize(pose[:, -4], c.d['normalization_tz'])
                epoch_info_6.append(torch.median(get_posit_err(result_posit, gt_posit)))
                
                result_angles = torch.zeros((x_hat_gt.shape[0], 3)).to(device)
                result_angles[:, 0] = p_encoding_t.batch_decode_even(x_hat_0[:, 6], x_hat_0[:, 7])
                result_angles[:, 1] = p_encoding_t.batch_decode_even(x_hat_0[:, 8], x_hat_0[:, 9])
                result_angles[:, 2] = p_encoding_t.batch_decode_even(x_hat_0[:, 10], x_hat_0[:, 11])
                orient_err = get_orient_err(result_angles * 2 * np.pi, pose[:, -3:] * 2 * np.pi)
                epoch_info_5.append(torch.median(orient_err))
                
                epoch_info_extra[4] += vae_recon_loss.item()
                epoch_info_extra[5] += vae_kl_loss.item()
                epoch_info_extra[6] += y_hat_inn_loss.item()
                epoch_info_extra[7] += inn_recon_loss.item()
        
        epoch_info[4] /= len(test_loader)
        epoch_info[5] = torch.median(torch.stack(epoch_info_5))
        epoch_info[6] = torch.median(torch.stack(epoch_info_6))
        epoch_info_extra[4:8] /= len(test_loader)


        model, return_text, mix_precision = trainer.step(model, epoch_info, mix_precision)
        if return_text == 'instable':
            optimizer = torch.optim.Adam(model.trainable_parameters, lr=current_lr)
            # optimizer.add_param_group({"params": model.cond_net.parameters(), "lr": current_lr})
            optimizer.add_param_group({"params": model.vae.encoder.parameters(), "lr": current_lr})
            optimizer.add_param_group({"params": model.vae.decoder.parameters(), "lr": current_lr})
        
        
        writer.add_scalar("INN_V/0_forward", epoch_info[0], epoch)
        writer.add_scalar("INN_V/1_reverse", epoch_info[1], epoch)
        writer.add_scalar("INN_V/2_pos", epoch_info[2], epoch)
        writer.add_scalar("INN_V/3_rot", epoch_info_extra[8], epoch)
        writer.add_scalar("INN_V/4_LR", current_lr, epoch)
        writer.add_scalar("INN_V/5_y_vae", epoch_info_extra[0], epoch)
        writer.add_scalar("INN_V/6_y_inn", epoch_info_extra[3], epoch)
        writer.add_scalar("INN_V/7_vae_kl", epoch_info_extra[1], epoch)
        writer.add_scalar("INN_V/8_y_hat_inn", epoch_info_extra[2], epoch)
        
        writer.add_scalar("INN_V/9_T_forward", epoch_info[4], epoch)
        writer.add_scalar("INN_V/10_T_orient", epoch_info[5], epoch)
        writer.add_scalar("INN_V/11_T_dist", epoch_info[6], epoch)
        writer.add_scalar("INN_V/12_T_y_vae", epoch_info_extra[4], epoch)
        writer.add_scalar("INN_V/13_T_y_inn", epoch_info_extra[7], epoch)
        writer.add_scalar("INN_V/14_T_vae_kl", epoch_info_extra[5], epoch)
        writer.add_scalar("INN_V/15_T_y_hat_inn", epoch_info_extra[6], epoch)
        writer.add_scalar("INN_V/16_remaining(h)", remaining_time, epoch)
        
        text_print = "Epoch {:d}".format(epoch) + \
            ' |f {:.5f}'.format(epoch_info[0]) + \
            ' |r {:.5f}'.format(epoch_info[1]) + \
            ' |r_pos {:.5f}'.format(epoch_info[2]) + \
            ' |r_rot {:.5f}'.format(epoch_info_extra[8]) + \
            ' |y_vae {:.5f}'.format(epoch_info_extra[0]) + \
            ' |y_inn {:.5f}'.format(epoch_info_extra[3]) + \
            ' |y_hat {:.5f}'.format(epoch_info_extra[2]) + \
            ' |h_left {:.1f}'.format(remaining_time) + \
            ' | ' + return_text
        # '|vae_kl {:.5f}'.format(epoch_info_extra[1]) + \
        print(text_print)
        with open('results/' + EXP_NAME + '/' + EXP_NAME + '.txt', "a") as tgt:
            tgt.writelines(text_print + '\n')
            
        text_print = "Epoch {:d}".format(epoch) + \
            ' |f {:.5f}'.format(epoch_info[4]) + \
            ' |ori {:.5f}'.format(epoch_info[5]) + \
            ' |dis {:.5f}'.format(epoch_info[6]) + \
            ' |y_vae {:.5f}'.format(epoch_info_extra[4]) + \
            ' |y_inn {:.5f}'.format(epoch_info_extra[7]) + \
            ' |y_hat {:.5f}'.format(epoch_info_extra[6]) + \
            ' |h_left {:.1f}'.format(remaining_time) + \
            ' | ' + 'TEST'
        # '|vae_kl {:.5f}'.format(epoch_info_extra[5]) + \
        print(text_print)
        with open('results/' + EXP_NAME + '/' + EXP_NAME + '.txt', "a") as tgt:
            tgt.writelines(text_print + '\n')
    writer.flush()


if __name__ == '__main__':
    main()
