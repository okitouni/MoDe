import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Function

class MoDeLoss():
    def __init__(self,bins=32,sbins=32,memory=False,background_label=0,background_only=True,power=2,order=0,lambd=None,max_slope=None,monotonic=False,eps=1e-4,dynamicbins=True,normalize=True,sign_func="tanh"):
        """
        Wrapper class for MoDe  Loss. Creates a callable that calculates the MoDe loss between two tensors. 

        Parameters
        ----------
        bins : int
            Number of bins in the biased feature to integrate over.
        sbins : int
            Number of bins of scores values. A large number of bins gives a more accurate loss but does not affect the gradients. 
        memory : bool, default True
            If True, integrate over biased feature locally i.e. on a per batch basis. Otherwise save data (biased feature and scores) from previous batches and perform a global MoDe claculation.  
        background_only : bool, default True
            If True, only apply the loss to the response of background events (label 1.) Otherwise, constrain the response for the whole tensor not just the subset labaled 1 (i.e. both classes at the same time if two classes are provided.)
        power : int, default 2
            Power used to calculate the flat part of the loss. E.g. L2: LegendreLoss=mean((F(s)-F_flat(s))**2)
        order : int={0,1,2}, default 0
            Order up tp which the Legendre expansion is computed.
        lambd : float, optional
            Amount of penalization to high slopes.  
        max_slope : float, optional 
        Specify a maximum slope in the decomposition as a fraction of the minimal slope that corresponds to 0 efficiency near some edge. If max_slope<1 then the response will have nonzero efficiency everywhere, otherwise the slope can ve large enough that efficiency is 0 near one edge. If None, the slope is unconstrained.  
        monotonic : bool, default True
            If True, forces the response to be monotonic for quadratic fits.  
        eps : float, default 1e-4 
            Small number used to prevent divergences in fractions used if monotonic is True. 
        dynamicbins : bool, default True
            If True, the bins of unbiased feature are adaptive i.e. have equal occupancy. Otherwise the bin width is fixed. If the bin width is fixed, the bins are padded up to the bin with highest occupancy which might use more memory.
        normalize : bool, default True
            If True, the values of the biased feature are normalzied between -1 and 1 on a per batch basis.
        """
        self.bins = bins
        self.sbins = sbins
        self.backonly = background_only
        self.background_label = background_label
        self.power = power
        self.order = order
        self.memory = memory
        self.lambd = lambd
        self.monotonic = monotonic
        self.max_slope = max_slope
        self.normalize = normalize
        self.eps = eps
        self.dynamicbins = dynamicbins
        if not dynamicbins:self.boundaries=torch.linspace(-1,1,bins)
        self.m = torch.Tensor()
        self.pred_long = torch.Tensor()
        self.fitter = _LegendreFitter(order=order, power=power,lambd=lambd,max_slope=max_slope,monotonic=monotonic,eps=eps,dynamicbins=dynamicbins) 

    def __call__(self,pred,target,x_biased,weights=None):
        """
        Returns the MoDe loss of pred over x_biased. 

        Parameters
        ----------
        pred : Tensor
            Tensor of predictions.
        target : Tensor
            Tensor of target labels. Must have the same shape as pred.
        x_biased : Tensor
            Tensor of biased feature. Must have the same shape as pred.
        weights : Tensor, Optional
            Tensor of weights for each sample. Must have the same shape as pred.
        """
        if self.backonly:
            mask = target==self.background_label
            x_biased = x_biased[mask]
            pred = pred[mask]
            target = target[mask]
            if weights is not None: weights = weights[mask]
            mod = x_biased.shape[0]%self.bins
            if mod !=0:
                x_biased = x_biased[:-mod]
                pred = pred[:-mod]
                target = target[:-mod] 
                if weights is not None: weights = weights[:-mod] #Not used currently 
        if self.memory:
            self.m = torch.cat([self.m,x_biased])
            self.pred_long = torch.cat([self.pred_long,pred])
            self.pred_long = self.pred_long.detach()
            m,msorted = self.m.sort()
            pred_long = self.pred_long[msorted].view(self.bins,-1)
            self.fitter.initialize(m=m.view(self.bins,-1),overwrite=True)
            m,msorted = x_biased.sort()
            pred = pred[msorted].view(self.bins,-1)
            if weights is not None:weights = weights[msorted].view(self.bins,-1)
            LLoss = _LegendreIntegral.apply(pred,self.fitter,weights,self.sbins,pred_long)
        else:
            if self.normalize:
                x_biased = 2*(x_biased - x_biased.min())/(x_biased.max()-x_biased.min()) -1
            m,msorted = x_biased.sort()
            m = m.view(self.bins,-1)
            pred = pred[msorted].view(self.bins,-1)
            if weights is not None:weights = weights[msorted].view(self.bins,-1)
#            if not self.dynamicbins: #still need to fix nbin normalization in dervatives
#                bin_index = torch.bucketize(x_biased,self.boundaries)     
#                m = torch.index_select(self.boundaries,0,torch.unique(bin_index))
#                m = torch.cat([torch.Tensor([-1]).to(m.device),m])
#                binned = [pred[bin_index==index] for index in torch.unique(bin_index)]
#                pred = pad_sequence(binned,batch_first=True,padding_value=0)
#                if weights is not None:
#                    binned = [weights[bin_index==index] for index in torch.unique(bin_index)]
#                    weights = pad_sequence(binned,batch_first=True,padding_value=0)
            self.fitter.initialize(m=m,overwrite=True)
            LLoss = _LegendreIntegral.apply(pred,self.fitter,weights,self.sbins)
        return LLoss 

    def __repr__(self):
        str1 = "power={}, background_only={}, order={}, bins={}, sbins={}, dynamicbins={}, normalize={}, monotonic={}, max_slope={}, memory={}".format(self.power, self.backonly,self.order,self.bins,self.sbins, self.dynamicbins, self.normalize, self.monotonic, self.max_slope, self.memory)
        str2 = repr(self.mse)
        return "\n".join([str1,str2])

    def reset(self):
        self.pred_long = torch.Tesnor()
        self.m = torch.Tesnor()
        return 

    def to(self,device):
        if not self.dynamicbins:self.boundaries=self.boundaries.to(device)
        return self
    

class _LegendreFitter():
    def __init__(self,order=0,power=1,lambd=None,max_slope=None,monotonic=False,eps=1e-8,dynamicbins=True):
        """
        Fit an array of values F(m) using Legendre polynomials. Must be initialized using the m array.

        Parameters
        ----------
        order : int, default 0
            The highest order of legendre polynomial used in the fit.
        power : int, default 1
            Power used in the norm of the difference between the input and the fit. |fit(input) - input|**power
        lambd : float, optional
            Amount of penalization to high slopes.  
        max_slope : float, optional 
            Specify a maximum slope in the decomposition. If None, the slope is unconstrained.  
        monotonic : bool, default True
            If True, forces the response to be monotonic for quadratic fits.  
        eps : float, default 1e-4 
            Small number used to prevent divergences in fractions used if monotonic is True. 
        dynamicbins : bool, default True
            If True, the bins of unbiased feature are adaptive i.e. have equal occupancy. Otherwise the bin width is fixed. If the bin width is fixed, the bins are padded up to the bin with highest occupancy which might use more memory.
        """
        self.power = power
        self.order = order
        self.lambd = lambd
        self.eps   = eps
        self.max_slope = max_slope
        self.monotonic = monotonic
        self.dynamicbins = dynamicbins
        self.initialized = False
        self.a0 = None
        self.a1 = None
        self.a2 = None
    def __call__(self,F):
        """
        Fit F(m) with Legendre polynomials and return the fit. Must be initialized using tensor of m values.
        
        Parameters
        ----------
        F : torch.Tensor
            Tensor of CDFs F_m(s) has shape (N,mbins) where N is the number of scores
        """
        if self.initialized == False:
            raise Exception("Please run initialize method before calling or use the MoDeLoss wrapper.")
        self.a0 = 1/2 * (F*self.dm).sum(axis=-1).view(-1,1) #integrate over mbins
        fit = self.a0.expand_as(F) # make boradcastable
        if self.order>0:
            self.a1 = 3/2 * (F*self.m*self.dm).sum(axis=-1).view(-1,1)
            if self.max_slope is not None:
                fit = fit + self.max_slope*self.a0*torch.tanh(self.a1/(self.max_slope*self.a0+self.eps))*self.m
            else:
                fit = fit + self.a1*self.m
        if self.order>1:
            p2 = (3*self.m**2-1)*0.5
            self.a2 = 5/2 * (F*p2*self.dm).sum(axis=-1).view(-1,1)
            if self.monotonic:
                fit = fit + self.a1/3.*torch.tanh(self.a2/(self.a1/3.+self.eps))*p2
            else:
                fit = fit+ self.a2*p2
        return fit

    def initialize(self,m,overwrite=True):
        if overwrite or self.initialized==False:
            if type(m) != torch.Tensor:
                m = torch.DoubleTensor(m)
            if self.dynamicbins:
                dm = m.max(axis=1)[0] - m.min(axis=1)[0]  # bin widths for each of the mbins.
                m  = m.mean(axis=1) # bin centers
            else:
                dm = m[:-1]-m[1:]
                m  = (m[:-1]+m[1:]) * 0.5
            self.m = m.view(-1)
            self.dm = dm.view(-1)
            self.mbins = self.m.shape[0]
            self.initialized = True
        return


class _LegendreIntegral(Function):
    @staticmethod
    def forward(ctx, input,fitter,weights=None,sbins=None,extra_input=None):
        """
        Calculate the MoDe loss of input: integral{Norm(F(s)-F_fit(s))} integrating over s. F(s) = CDF_input(s)

        Parameters
        ----------
        input : torch.Tensor
            Scores with shape (mbins,bincontent) where mbins * bincontent = N (or the batch size.)
        fitter : LegendreFitter
            Fitter object used to calculate F_flat(s)
        sbins : int
            Number of s bins to use in the integral.
        extra_input : torch.Tensor
            Additional scores from previous batches if memory=True is used.
        """
        s_edges = torch.linspace(0,1,sbins+1,dtype=input.dtype).to(input.device) #create s edges to integrate over
        s = (s_edges[1:] + s_edges[:-1])*0.5
        s = _expand_dims_as(s,input)
        ds = s_edges[1:] - s_edges[:-1]
        F = s-input
        _Heaviside_(F)
        F = F.sum(axis=-1)/input.shape[-1] # get CDF at s from input values
        if weights is not None:
            weights = weights.sum(axis=-1).true_divide(weights.shape[1])
            ctx.weights = torch.repeat_interleave(weights,input.shape[1]).view(input.shape)
        else:
            weights,ctx.weights=1.,1.
        integral = (ds.matmul((F-fitter(F))**fitter.power)*weights).sum(axis=0)/input.shape[0] # not exactly right with max_slope
        del F,s,ds,s_edges

        # Stuff for backward
        if extra_input is not None:
            input_appended = extra_input
        else:
            input_appended = input
        F_s_i =  _expand_dims_as(input.view(-1),input) #make a flat copy of input and add dimensions for boradcasting
        F_s_i =  F_s_i-input_appended
        _Heaviside_(F_s_i)
        F_s_i =  F_s_i.sum(axis=-1)/F_s_i.shape[-1] #sum over bin content to get CDF
        residual = F_s_i - fitter(F_s_i)
        ctx.fitter = fitter
        ctx.residual = residual
        ctx.shape = input.shape
        return integral

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        shape = ctx.shape
        lambd = ctx.fitter.lambd
        max_slope = ctx.fitter.max_slope
        monotonic = ctx.fitter.monotonic
        eps = ctx.fitter.eps
        power = ctx.fitter.power
        order = ctx.fitter.order
        dm = ctx.fitter.dm
        m = ctx.fitter.m
        a0 = ctx.fitter.a0.view(shape)
        if ctx.needs_input_grad[0]:
            dF = ctx.residual[torch.eye(shape[0],dtype=bool).repeat_interleave(shape[1],axis=0)].view(shape)
            dF0 =  ctx.residual.sum(axis=-1).view(shape) *-.5* dm.view(-1,1)
            summation = dF + dF0
            if order >0:
                a1 = ctx.fitter.a1.view(shape)
                if max_slope is None:
                    dF1  = (ctx.residual*m).sum(axis=-1).view(shape) *-1.5* (dm*m).view(-1,1)
                    summation += dF1
                else:
                    a0 = max_slope*a0
                    dF1   =  (ctx.residual*m).sum(axis=-1).view(shape) *\
                                (-1.5* (dm*m).view(-1,1) *(1/torch.cosh(a1/(a0+eps)))**2+\
                                -a1/(a0+eps)*(1/torch.cosh(a1/(a0+eps)))**2*-.5* dm.view(-1,1)*max_slope+\
                                torch.tanh(a1/(a0+eps))*-.5* dm.view(-1,1)*max_slope)
                                
                    summation += dF1
            if order>1:
                a2 = ctx.fitter.a2.view(shape)
                if not monotonic:
                    dF2   = (ctx.residual*.5*(3*m**2-1)).sum(axis=-1).view(shape) *\
                            -2.5*(dm*0.5*(3*m**2-1)).view(-1,1) #dc_2/ds is this term 
                    summation += dF2
                else:
                    a1 = a1/3.
                    dF2   = (ctx.residual*.5*(3*m**2-1)).sum(axis=-1).view(shape) *\
                            (1/torch.cosh(a2/(a1+eps))**2*(-2.5*dm*0.5*(3*m**2-1)).view(-1,1)+\
                            (1/torch.cosh(a2/(a1+eps))**2* -a2/(a1+eps)*-1.5*(dm*m).view(-1,1)/3. +\
                            -1.5*(dm*m).view(-1,1)/3.*(torch.tanh(a2/(a1+eps)))))
                    summation += dF2

            summation *= (-power)/np.prod(shape)
            if lambd is not None:
                summation += -lambd*2/np.prod(shape) *\
                3/2* ctx.fitter.a1.view(shape)*(m*dm).view(-1,1)

            grad_input  = grad_output * summation * ctx.weights

        return grad_input, None, None, None, None

def _expand_dims_as(t1,t2):
    result = t1[(...,)+(None,)*t2.dim()]
    return result

def _Heaviside_(tensor):
    tensor.masked_fill_(tensor>0, 1)
    tensor.masked_fill_(tensor==0, 0.5)
    tensor.masked_fill_(tensor<0, 0)
    return

