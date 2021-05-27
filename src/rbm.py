


import torch
import h5py
import numpy as np

class RBM:
    # NEEDED VAR:
    # * num_visible
    # * num_hidden
    def __init__(self, num_visible, # number of visible nodes
                 num_hidden, # number of hidden nodes
                 device, # CPU or GPU ?
                 gibbs_steps=10, # number of MCMC steps for computing the neg term
                 var_init=1e-4, # variance of the init weights
                 dtype=torch.float,
                 num_pcd = 100, # number of permanent chains
                 lr = 0.01, # learning rate
                 ep_max = 100, # number of epochs
                 mb_s = 50, # size of the minibatch
                 UpdCentered = False, # Update using centered gradients
                 CDLearning = False
                 ): 
        self.Nv = num_visible        
        self.Nh = num_hidden
        self.gibbs_steps = gibbs_steps
        self.device = device
        self.dtype = dtype
        # weight of the RBM
        self.W = torch.randn(size=(self.Nh,self.Nv), device=self.device, dtype=self.dtype)*var_init
        self.var_init = var_init
        # visible and hidden biases
        self.vbias = torch.zeros(self.Nv, device=self.device, dtype=self.dtype)
        self.hbias = torch.zeros(self.Nh, device=self.device, dtype=self.dtype)
        # permanent chain
        self.X_pc = torch.bernoulli(torch.rand((self.Nv,num_pcd), device=self.device, dtype=self.dtype))
        self.lr = lr
        self.ep_max = ep_max
        self.mb_s = mb_s
        self.num_pcd = num_pcd

        self.ep_tot = 0
        self.up_tot = 0
        self.list_save_time = []
        self.list_save_rbm = []
        self.file_stamp = ''
        self.VisDataAv = 0
        self.HidDataAv = 0
        self.UpdCentered = UpdCentered
        self.ResetPermChainBatch = False
        self.CDLearning = CDLearning

    # save RBM's parameters
    def saveRBM(self,fname):
        f = h5py.File(fname,'w')
        f.create_dataset('W',data=self.W.cpu())
        f.create_dataset('hbias',data=self.hbias.cpu())
        f.create_dataset('vbias',data=self.vbias.cpu())
        f.create_dataset('X_pc',data=self.X_pc.cpu())
        f.close()

    # load a RBM from file
    def loadRBM(self,fname,stamp='',PCD=False):
        f = h5py.File(fname,'r')
        self.W = torch.tensor(f['W'+stamp]); 
        self.vbias = torch.tensor(f['vbias'+stamp]); 
        self.hbias = torch.tensor(f['hbias'+stamp]); 
        self.Nv = self.W.shape[1]
        self.Nh = self.W.shape[0]
        if PCD:
            self.x_pc = torch.tensor(f['X_pc']+stamp); 
        if self.device.type != "cpu":
            self.W = self.W.cuda()
            self.vbias = self.vbias.cuda()
            self.hbias = self.hbias.cuda()
            if PCD:
                self.X_pc = self.X_pc.cuda()

    # tile a set of 2d arrays
    def ImConcat(self,X,ncol=10,nrow=5,sx=28,sy=28,ch=1):
        tile_X = []
        for c in range(nrow):
            L = torch.cat((tuple(X[i,:].reshape(sx,sy,ch) for i in np.arange(c*ncol,(c+1)*ncol))))
            tile_X.append(L)
        return torch.cat(tile_X,1)

    
    # X is Nv x Ns
    # return the free energy of all samples -log(p(x))
    def FreeEnergy(self, X):
        vb = torch.sum(X.t() * self.vbias,1) # vb: Ns
        fe_exp = 1 + torch.exp(self.W.mm(X).t() + self.hbias) # fe_exp: Ns x Nh
        Wx_b_log = torch.sum( torch.log(fe_exp),1) # Wx_b_log: Ns
        result = - vb - Wx_b_log # result: Ns
        return result

    # Compute the energy of the RBM
    # V is Nv x Ns x NT
    # H is Nh x Ns x NT
    # E is Ns x NT
    def computeE(self,V,H):
        INT = torch.sum((H * torch.tensordot(self.W,V,dims=1)),0)
        FIELDS = torch.tensordot(self.hbias,H,dims=1) +  torch.tensordot(self.vbias,V,dims=1)
        return -(INT + FIELDS)

    # Compute ratio of log(Z) using AIS
    def ComputeFE(self,nβ=1000,NS=1000):
        FE_RATIO_AIS = self.ComputeFreeEnergyAIS(nβ,NS)
        FE_PRIOR = self.ComputeFreeEnergyPrior()

        return FE_PRIOR,FE_RATIO_AIS

    def ComputeFreeEnergyAIS(self,nβ,nI):

        βlist = torch.arange(0,1.000001,1.0/nβ)
        x = torch.zeros(self.Nv+self.Nh2,nI,device=self.device)
        H = torch.zeros(self.Nh,nI,device=self.device)
        E = torch.zeros(nI,device=self.device)

        # initialize xref
        x = torch.bernoulli(torch.sigmoid(self.vbias_prior).repeat(nI,1).t())
        H = torch.bernoulli(torch.rand((self.Nh,nI),device=self.device))
        E = self.computeE(x,H).double().to(self.device)  - self.computeE_prior(x)
        self.V_STATE = x
        self.H_STATE = H
        for idβ in range(1,nβ+1):
            H, _ = self.SampleHiddens01(x,β=βlist[idβ])
            x, _ = self.SampleVisibles01(H,β=βlist[idβ])
            E += self.computeE(x,H)

        Δ = 0
        # E = self.computeE(x,H) - self.computeE_prior(x)
        Δβ = 1.0/nβ
        Δ = -Δβ*E # torch.sum(E,1)
        #for idβ in range(nβ):
        #    Δβ = 1/nβ
        #    Δ += -Δβ*E[:,idβ]
        Δ = Δ.double()
        Δ0 = torch.mean(Δ)

        AIS = torch.log(torch.mean(torch.exp(Δ-Δ0).double()))+Δ0
        # AIS = torch.log(torch.mean(torch.exp(Δ)))
        return AIS

    # Compute both AATS scores
    def ComputeAATS(self,X,fake_X,s_X):
        CONCAT = torch.cat((X[:,:s_X],fake_X[:,:s_X]),1)
        dAB = torch.cdist(CONCAT.t(),CONCAT.t())    
        torch.diagonal(dAB).fill_(float('inf'))
        dAB = dAB.cpu().numpy()

        # the next line is use to tranform the matrix into
        #  d_TT d_TF   INTO d_TF- d_TT-  where the minus indicate a reverse order of the columns
        #  d_FT d_FF        d_FT  d_FF
        dAB[:int(dAB.shape[0]/2),:] = dAB[:int(dAB.shape[0]/2),::-1] 
        closest = dAB.argmin(axis=1) 
        n = int(closest.shape[0]/2)

        ninv = 1/n
        correctly_classified = closest>=n   #np.concatenate([(closest[:n] < n), (closest[n:] >= n)])
        AAtruth = (closest[:n] >= n).sum()*ninv  # for a true sample, proba that the closest is in the set of true samples
        AAsyn = (closest[n:] >= n).sum()*ninv  # for a fake sample, proba that the closest is in the set of fake samples

        return AAtruth, AAsyn

    # init the visible bias using the empirical frequency of the training dataset
    def SetVisBias(self,X):
        NS = X.shape[1]
        prob1 = torch.sum(X,1)/NS
        prob1 = torch.clamp(prob1,min=1e-5)
        prob1 = torch.clamp(prob1,max=1-1e-5)
        self.vbias = -torch.log(1.0/prob1 - 1.0)

    # define an initial value fo the permanent chain
    def InitXpc(self,V):
        self.X_pc = V

    # Sampling and getting the mean value using Sigmoid
    # using CurrentState
    def SampleHiddens01(self,V,β=1):             
        mh = torch.sigmoid(β*(self.W.mm(V).t() + self.hbias).t())
        h = torch.bernoulli(mh)

        return h,mh

    # H is Nh X M
    # W is Nh x Nv
    # Return Visible sample and average value for binary variable
    def SampleVisibles01(self,H,β=1):
        mv = torch.sigmoid(β*(self.W.t().mm(H).t() + self.vbias).t())
        v = torch.bernoulli(mv)
        return v,mv

    # Compute the negative term for the gradient
    # IF it_mcmc=0 : use the class variable self.gibbs_steps for the number of MCMC steps
    # IF self.anneal_steps>= : perform anealing for the corresponding number of steps
    # FOR ANNEALING: only if the max eigenvalues is above self.ann_threshold
    # βs : effective temperure. Used only if =! -1
    def GetAv(self,it_mcmc=0,βs=-1):
        if it_mcmc==0:
            it_mcmc = self.gibbs_steps

        v = self.X_pc
        mh = 0

        β=1
        h,mh = self.SampleHiddens01(v,β=β)
        v,mv = self.SampleVisibles01(h,β=β)
        
        for t in range(1,it_mcmc):
            h,mh = self.SampleHiddens01(v,β=β)
            v,mv = self.SampleVisibles01(h,β=β)


        return v,mv,h,mh

    # Return samples and averaged values
    # IF it_mcmc=0 : use the class variable self.gibbs_steps for the number of MCMC steps
    # IF self.anneal_steps>= : perform anealing for the corresponding number of steps
    # FOR ANNEALING: only if the max eigenvalues is above self.ann_threshold
    # βs : effective temperure. Used only if =! -1
    def Sampling(self,X,it_mcmc=0):  
        if it_mcmc==0:
            it_mcmc = self.gibbs_steps

        v = X
        β=1

        h,mh = self.SampleHiddens01(v,β=β)
        v,mv = self.SampleVisibles01(h,β=β)
        
        for t in range(it_mcmc-1):
            h,mh = self.SampleHiddens01(v,β=β)
            v,mv = self.SampleVisibles01(h,β=β)
            
        return v,mv,h,mh

    # Update weights and biases
    def updateWeights(self,v_pos,h_pos,v_neg,h_neg_v,h_neg_m):

        lr_p = self.lr/self.mb_s
        lr_n = self.lr/self.num_pcd
        lr_reg = self.lr*self.regL2

        NegTerm_ia = h_neg_v.mm(v_neg.t())

        self.W += h_pos.mm(v_pos.t())*lr_p -  NegTerm_ia*lr_n
        self.vbias += torch.sum(v_pos,1)*lr_p - torch.sum(v_neg,1)*lr_n
        self.hbias += torch.sum(h_pos,1)*lr_p - torch.sum(h_neg_m,1)*lr_n

            

    # Update weights and biases
    def updateWeightsCentered(self,v_pos,h_pos_v,h_pos_m,v_neg,h_neg_v,h_neg_m,ν=0.2,ε=0.01):

        # self.HidDataAv = (1-ν)*self.HidDataAv + ν*torch.mean(h_pos_m,1)
        self.VisDataAv = torch.mean(v_pos,1)
        self.HidDataAv = torch.mean(h_pos_m,1)
        Xc_pos = (v_pos.t() - self.VisDataAv).t()
        Hc_pos = (h_pos_m.t() - self.HidDataAv).t()

        Xc_neg = (v_neg.t() - self.VisDataAv).t()
        Hc_neg = (h_neg_m.t() - self.HidDataAv).t()

        NormPos = 1.0/self.mb_s
        NormNeg = 1.0/self.num_pcd
        # NormL2 = self.regL2

        siτa_neg = Hc_neg.mm(Xc_neg.t())*NormNeg
        si_neg = torch.sum(v_neg,1)*NormNeg
        τa_neg = torch.sum(h_neg_m,1)*NormNeg


        ΔW = Hc_pos.mm(Xc_pos.t())*NormPos -  siτa_neg

        self.W += ΔW*self.lr

        ΔVB = torch.sum(v_pos,1)*NormPos - si_neg - torch.mv(ΔW.t(),self.HidDataAv)
        self.vbias += self.lr*ΔVB

        ΔHB = torch.sum(h_pos_m,1)*NormPos - τa_neg - torch.mv(ΔW,self.VisDataAv)
        self.hbias += self.lr*ΔHB


    # Compute positive and negative term
    def fit_batch(self,X):
        h_pos_v, h_pos_m = self.SampleHiddens01(X)
        if self.CDLearning:
            self.X_pc = X
            self.X_pc,_,h_neg_v,h_neg_m = self.GetAv()
        else:
            self.X_pc,_,h_neg_v,h_neg_m = self.GetAv()

        if self.UpdCentered:
            self.updateWeightsCentered(X,h_pos_v,h_pos_m,self.X_pc,h_neg_v,h_neg_m)
        else:
            self.updateWeights(X,h_pos_m,self.X_pc,h_neg_v,h_neg_m)

   
    def getMiniBatches(self,X,m):
        return X[:,m*self.mb_s:(m+1)*self.mb_s]

    def fit(self,X,ep_max=0):
        if ep_max==0:
            ep_max = self.ep_max

        NB = int(X.shape[1]/self.mb_s)

        if self.ep_tot==0:
            self.VisDataAv = torch.mean(X,1)

        #_,h_av = self.SampleHiddens01(X)
        #self.HidDataAv = torch.mean(h_av,1)

        if (len(self.list_save_time)>0) & (self.up_tot == 0):
            f = h5py.File('models/AllParameters'+self.file_stamp+'.h5','w')
            f.create_dataset('alltime',data=self.list_save_time)
            f.close()

        if (len(self.list_save_rbm)>0) & (self.ep_tot == 0):
            f = h5py.File('models/RBM'+self.file_stamp+'.h5','w')   
            f.create_dataset('lr',data=self.lr)
            f.create_dataset('NGibbs',data=self.gibbs_steps)
            f.create_dataset('UpdByEpoch',data=NB)
            f.create_dataset('miniBatchSize',data=self.mb_s)
            f.create_dataset('numPCD',data=self.num_pcd)
            f.create_dataset('alltime',data=self.list_save_rbm)
            f.close()
        

        for t in range(ep_max):
            print("IT ",self.ep_tot)
            self.ep_tot += 1

            Xp = X[:,torch.randperm(X.size()[1])]
            for m in range(NB):
                # print(m)
                if self.ResetPermChainBatch:
                    self.X_pc = torch.bernoulli(torch.rand((self.Nv,self.num_pcd), device=self.device, dtype=self.dtype))                    

                Xb = self.getMiniBatches(Xp,m)
                self.fit_batch(Xb)                

                if self.up_tot in self.list_save_time:
                    f = h5py.File('models/AllParameters'+self.file_stamp+'.h5','a')
                    print('Saving nb_upd='+str(self.up_tot))
                    f.create_dataset('W'+str(self.up_tot),data=self.W.cpu())
                    f.create_dataset('vbias'+str(self.up_tot),data=self.vbias.cpu())
                    f.create_dataset('hbias'+str(self.up_tot),data=self.hbias.cpu())
                    f.close()

                self.up_tot += 1

            if self.ep_tot in self.list_save_rbm:
                f = h5py.File('models/RBM'+self.file_stamp+'.h5','a')
                f.create_dataset('W'+str(self.ep_tot),data=self.W.cpu())
                f.create_dataset('vbias'+str(self.ep_tot),data=self.vbias.cpu())
                f.create_dataset('hbias'+str(self.ep_tot),data=self.hbias.cpu())
                f.close()

