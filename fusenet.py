import torch
import torch.nn as nn
import torch.nn.functional as F

class FuseNet(nn.Module):
    def __init__(self, chs=(64, 32, 1)):
        super(FuseNet, self).__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
            )
        self.d_conv = nn.Sequential(
            nn.Conv3d(chs[0], chs[2], kernel_size=3, padding=1),
            nn.Sigmoid())
        self.F_conv = nn.Sequential(
            nn.Conv3d(128, chs[0], kernel_size=3, padding=1),
            nn.GroupNorm(64 // 4, 64),
            nn.ReLU(),
            nn.Conv3d(chs[0], chs[2], kernel_size=3, padding=1),
            nn.Sigmoid()
        )



    def forward(self, fA_g, fB_g, fC_g, fD_g, fA_l, fB_l, fC_l, fD_l):

        fAg = self.up1(fA_g)
        fBg = self.up1(fB_g)
        fCg = self.up1(fC_g)
        fDg = self.up1(fD_g)
        FA = torch.cat((fAg, fA_l), dim=1)
        FB = torch.cat((fBg, fB_l), dim=1)
        FC = torch.cat((fCg, fC_l), dim=1)
        FD = torch.cat((fDg, fD_l), dim=1)
        eps = 1e-7

        WA = self.d_conv(FA)
        WB = self.d_conv(FB)
        WC = self.d_conv(FC)
        WD = self.d_conv(FD)

        # wA = (WA/(WA+WB+WC+WD+eps)).permute(0, 1, 4, 3, 2)
        # wB = (WB/(WA+WB+WC+WD+eps)).permute(0, 1, 4, 3, 2)
        # wC = (WC / (WA + WB + WC + WD + eps)).permute(0, 1, 4, 3, 2)
        # wD = (WD / (WA + WB + WC + WD + eps)).permute(0, 1, 4, 3, 2)
        wA = (WA / (WA + WB + WC + WD + eps))
        wB = (WB / (WA + WB + WC + WD + eps))
        wC = (WC / (WA + WB + WC + WD + eps))
        wD = (WD / (WA + WB + WC + WD + eps))

        # FA = self.d_conv(FA)
        # FB = self.d_conv(FB)
        # FC = self.d_conv(FC)
        # FD = self.d_conv(FD)
        # F = torch.cat((FA, FB, FC, FD), dim=1)
        # fused_image = self.F_conv(F)
        # fused_image = fused_image.permute(0, 1, 4, 3, 2)



        return wA, wB, wC, wD