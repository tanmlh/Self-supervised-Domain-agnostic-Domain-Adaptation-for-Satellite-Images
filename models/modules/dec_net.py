import torch
import torch.nn as nn
from .unet.model import Unet
from .fpn.model import FPN
from .transformer.Models import Transformer
import torch.nn.functional as F
import pdb
from .unet.decoder import UnetDecoder
from .utils import check_freeze

def get_dec_net(net_conf):

    if 'name' in net_conf and net_conf['name'] == 'identity':
        dec_net = nn.Identity()
    else:
        dec_net = UnetDecoder(
            encoder_channels=net_conf['enc_channels'],
            decoder_channels=net_conf['dec_channels'],
            n_blocks=net_conf['depth'],
            norm_type=net_conf['norm_type'] if 'norm_type' in net_conf else 'bn',
            center=True if net_conf['enc_name'].startswith("vgg") else False,
            attention_type=net_conf['att_type'],
        )

        check_freeze(dec_net, net_conf)
    return dec_net

class PrivateDecoder(nn.Module):
	def __init__(self, shared_code_channel, private_code_size):
		super(PrivateDecoder, self).__init__()
		num_att = 256
		self.shared_code_channel = shared_code_channel
		self.private_code_size = private_code_size

		self.main = []
		self.upsample = nn.Sequential(
            # input: 1/8 * 1/8
            nn.ConvTranspose2d(256, 256, 4, 2, 2, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
			Conv2dBlock(256, 128, 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), 
			# 1/4 * 1/4
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
			Conv2dBlock(128, 64 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), 
			# 1/2 * 1/2
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
			Conv2dBlock(64 , 32 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), 
			# 1 * 1
			nn.Conv2d(32, 3, 3, 1, 1),
			nn.Tanh())

		self.main += [Conv2dBlock(shared_code_channel+num_att+1, 256, 3, stride=1, padding=1, norm='ln', activation='relu', pad_type='reflect', bias=False)]
		self.main += [ResBlocks(3, 256, 'ln', 'relu', pad_type='zero')]
		self.main += [self.upsample]

		self.main = nn.Sequential(*self.main)
		self.mlp_att   = nn.Sequential(nn.Linear(private_code_size, private_code_size),
						   	 nn.ReLU(),
						   	 nn.Linear(private_code_size, private_code_size),
						   	 nn.ReLU(),
						   	 nn.Linear(private_code_size, private_code_size),
						   	 nn.ReLU(),
						   	 nn.Linear(private_code_size, num_att))
	
	def assign_adain_params(self, adain_params, model):
		# assign the adain_params to the AdaIN layers in model
		for m in model.modules():
			if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
				mean = adain_params[:, :m.num_features]
				std = torch.exp(adain_params[:, m.num_features:2*m.num_features])
				m.bias = mean.contiguous().view(-1)
				m.weight = std.contiguous().view(-1)
				if adain_params.size(1) > 2*m.num_features:
					adain_params = adain_params[:, 2*m.num_features:]

	def get_num_adain_params(self, model):
		# return the number of AdaIN parameters needed by the model
		num_adain_params = 0
		for m in model.modules():
			if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
				num_adain_params += 2*m.num_features
		return num_adain_params

	def forward(self, shared_code, private_code, d):
		d = Variable(torch.FloatTensor(shared_code.shape[0], 1).fill_(d)).cuda()
		d = d.unsqueeze(1)
		d_img = d.view(d.size(0), d.size(1), 1, 1).expand(d.size(0), d.size(1), shared_code.size(2), shared_code.size(3))
		att_params = self.mlp_att(private_code)
		att_img    = att_params.view(att_params.size(0), att_params.size(1), 1, 1).expand(att_params.size(0), att_params.size(1), shared_code.size(2), shared_code.size(3))
		code         = torch.cat([shared_code, att_img, d_img], 1)

		output = self.main(code)
		return output


