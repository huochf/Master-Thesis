import torch
import torch.nn as nn
import torch.nn.functional as F


class Voxel3DEncoder(nn.Module):
	# adapted from https://github.com/JuheonHwang/SMPL-registration/blob/main/smpl_registration/models/ipnet_models.py

	def __init__(self, feat_dim=256, num_parts=7, res=128):
		super(Voxel3DEncoder, self).__init__()
		self.feat_dim = feat_dim
		self.num_parts = num_parts

		self.conv_00 = nn.Conv3d(1, 32, 3, padding=1) # out: 128
		self.conv_01 = nn.Conv3d(32, 32, 3, padding=1) # out: 128
		self.bn_01 = nn.BatchNorm3d(32)

		self.conv_10 = nn.Conv3d(32, 64, 3, padding=1) # out: 128
		self.conv_11 = nn.Conv3d(64, 64, 3, padding=1, stride=2) # out: 64
		self.bn_11 = nn.BatchNorm3d(64)

		self.conv_20 = nn.Conv3d(64, 64, 3, padding=1) # out: 64
		self.conv_21 = nn.Conv3d(64, 64, 3, padding=1, stride=2) # out: 32
		self.bn_21 = nn.BatchNorm3d(64)

		self.conv_30 = nn.Conv3d(64, 128, 3, padding=1) # out: 32
		self.conv_31 = nn.Conv3d(128, 128, 3, padding=1, stride=2) # out: 16
		self.bn_31 = nn.BatchNorm3d(128)

		self.conv_40 = nn.Conv3d(128, 128, 3, padding=1) # out: 16
		self.conv_41 = nn.Conv3d(128, 128, 3, padding=1, stride=2) # out: 8
		self.bn_41 = nn.BatchNorm3d(128)

		self.conv_50 = nn.Conv3d(128, 256, 3, padding=1) # out: 8
		self.conv_51 = nn.Conv3d(256, 256, 3, padding=1, stride=2) # out: 4
		self.bn_51 = nn.BatchNorm3d(256)

		s = res // 32
		self.fc_0 = nn.Conv1d(256 * s * s * s, feat_dim * num_parts, 1)
		self.fc_1 = nn.Conv1d(feat_dim * num_parts, feat_dim * num_parts, 1, groups=num_parts)
		self.fc_2 = nn.Conv1d(feat_dim * num_parts, feat_dim * num_parts, 1, groups=num_parts)

		self.actvn = nn.ReLU()


	def forward(self, voxel):
		# voxel: [b, 1, 128, 128, 128]
		batch_size = voxel.shape[0]

		x = self.actvn(self.conv_00(voxel))
		x = self.actvn(self.conv_01(x))
		x = self.bn_01(x)

		x = self.actvn(self.conv_10(x))
		x = self.actvn(self.conv_11(x))
		x = self.bn_11(x)

		x = self.actvn(self.conv_20(x))
		x = self.actvn(self.conv_21(x))
		x = self.bn_21(x)

		x = self.actvn(self.conv_30(x))
		x = self.actvn(self.conv_31(x))
		x = self.bn_31(x)

		x = self.actvn(self.conv_40(x))
		x = self.actvn(self.conv_41(x))
		x = self.bn_41(x)

		x = self.actvn(self.conv_50(x))
		x = self.actvn(self.conv_51(x))
		x = self.bn_51(x)

		x = x.reshape(batch_size, -1, 1)

		feats = self.actvn(self.fc_0(x))
		feats = self.actvn(self.fc_1(feats))
		feats = self.actvn(self.fc_2(feats))

		return feats.reshape(batch_size, self.num_parts, self.feat_dim)
