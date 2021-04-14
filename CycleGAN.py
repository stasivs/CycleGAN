import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import itertools
import numpy as np

import Generators
import Discriminators
import DataPreprocessing


class CycleGAN:
    def __init__(self, coeff=1, gpu_mode=False, const_photo=None,
                 generator=Generators.ImprovedUNetGen, discriminator=Discriminators.PatchGAN):


        self.LAMBDA = coeff
        self.device = torch.device('cpu')
        if gpu_mode:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.const_photo = const_photo.reshape((1, 3, 256, 256)).to(self.device)

        # Defining Generators and Discriminators
        self.netG = generator().to(self.device)  # Making fake Monet work
        self.netF = generator().to(self.device)  # Making fake Real photo

        self.netDM = discriminator().to(self.device)# Checking is it Monet work
        self.netDP = discriminator().to(self.device)  # Checking is it Real photo

        # Defining Losses
        self.ganLoss = nn.BCELoss().to(self.device)
        self.cycleConsistencyLoss = nn.L1Loss().to(self.device)
        self.identityLoss = nn.L1Loss().to(self.device)

        # Defining Optimizers
        self.optimG = optim.Adam(itertools.chain(self.netG.parameters(), self.netF.parameters()),
                                 lr=2e-4, betas=(0.5, 0.999))
        self.optimDM = optim.Adam(self.netDM.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimDP = optim.Adam(self.netDP.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Defining History of losses
        self.DP_loss_hist = []
        self.DM_loss_hist = []
        self.G_loss_hist = []

        # Defining image history
        self.image_hist = []

    def fit(self, photo, monet, epochs=10):
        l = min(len(photo), len(monet))
        for epoch in range(1, epochs + 1):
            epoch_G_loss = 0
            epoch_DM_loss = 0
            epoch_DP_loss = 0
            for batch_num, data in enumerate(zip(photo, monet), 1):
                realP, realM = data
                realP, realM = realP.to(self.device), realM.to(self.device)

                # Init graph to train netG, netF
                fakeM, fakeP = self.netG(realP), self.netF(realM)
                recP, recM = self.netF(fakeM), self.netG(fakeP)

                DM_G = self.netDM(fakeM)
                DP_F = self.netDP(fakeP)

                # Train netG, netF
                self.optimG.zero_grad()
                G_gLoss = self.ganLoss(DM_G, torch.ones_like(DM_G))
                F_gLoss = self.ganLoss(DP_F, torch.ones_like(DP_F))
                G_cLoss = self.cycleConsistencyLoss(realP, recP)
                F_cLoss = self.cycleConsistencyLoss(realM, recM)
                G_iLoss = self.identityLoss(realM, fakeP)
                F_iLoss = self.identityLoss(realP, fakeM)
                Gen_loss = (G_gLoss + F_gLoss) + (G_cLoss + F_cLoss) * self.LAMBDA \
                           + (G_iLoss + F_iLoss) * self.LAMBDA / 2
                Gen_loss.backward()
                self.optimG.step()
                self.G_loss_hist.append(Gen_loss)
                epoch_G_loss += Gen_loss

                # Init Discriminator graph
                DM_G = self.netDM(fakeM.detach())
                DM_M = self.netDM(realM)
                DP_F = self.netDP(fakeP.detach())
                DP_P = self.netDP(realP)

                # Train netDM
                self.optimDM.zero_grad()
                DM_fake_loss = self.ganLoss(DM_G, torch.zeros_like(DM_G))
                DM_real_loss = self.ganLoss(DM_M, torch.ones_like(DM_M))
                DM_loss = (DM_fake_loss + DM_real_loss) / 2
                DM_loss.backward()
                self.optimDM.step()
                self.DM_loss_hist.append(DM_loss)
                epoch_DM_loss += DM_loss

                # Train netDP
                self.optimDP.zero_grad()
                DP_fake_loss = self.ganLoss(DP_F, torch.zeros_like(DP_F))
                DP_real_loss = self.ganLoss(DP_P, torch.ones_like(DP_P))
                DP_loss = (DP_fake_loss + DP_real_loss) / 2
                DP_loss.backward()
                self.optimDP.step()
                self.DP_loss_hist.append(DP_loss)
                epoch_DP_loss += DP_loss

            if self.const_photo != None:
                fakeCP = self.netG(self.const_photo)
                recCP = self.netF(fakeCP)

                cp = DataPreprocessing.return_img_from_tensor(self.const_photo.detach().to("cpu"))
                fakeCP = DataPreprocessing.return_img_from_tensor(fakeCP.detach().to("cpu"))
                recCP = DataPreprocessing.return_img_from_tensor(recCP.detach().to("cpu"))

                self.image_hist.append([cp, fakeCP, recCP])

            epoch_G_loss /= l
            epoch_DM_loss /= l
            epoch_DP_loss /= l
            print("Epoch: {}/{}. Gen_loss: {}.  Disc_monet_loss: {}. Disc_photo_loss: {}.".format(
                epoch, epochs, epoch_G_loss, epoch_DM_loss, epoch_DP_loss
            ))


        return self.netG, (self.G_loss_hist, self.DM_loss_hist, self.DP_loss_hist), self.image_hist

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    monet_loader = DataPreprocessing.return_dataloader("monet_jpg", transform, 2)
    photo_loader = DataPreprocessing.return_dataloader("photo_jpg", transform, 2)
    const_photo = DataPreprocessing.return_const_photo(photo_loader)

    cg = CycleGAN(const_photo=const_photo, gpu_mode=True)
    netG, losses, image_hist = cg.fit(photo_loader, monet_loader)
