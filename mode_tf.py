import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.ops.math_ops import bucketize

class MoDeLoss():
    def __init__(self,bins=32,sbins=32,memory=False,background_only=True,power=2,order=0,lambd=None,max_slope=None,monotonic=False,eps=1e-4,dynamicbins=True,normalize=True):
        """
        Wrapper for MoDe  Loss.

        Parameters
        ----------
        bins : int
            Number of bins in the biased feature to integrate over.
        sbins : int
            Number of bins of scores values. A large number of bins gives a more accurate loss but does not affect the gradients.
        memory : bool, default True
            If True, integrate over biased feature locally i.e. on a per batch basis. Otherwise save data (biased feature and scores) from previous batches and perform a global MoDe claculation.
        background_only : bool, default True
            If True, only apply the loss to the response of background events (label 1.) Otherwise, constrain the response for both classes at the same time.
        power : int, default 2
            Power used to calculate the flat part of the loss. E.g. L2: LegendreLoss=mean((F(s)-F_flat(s))**2)
        order : int={0,1,2}, default 0
            Order up tp which the Legendre expansion is computed.
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
        normalize : bool, default True
            If True, the values of the biased feature are normalzied between -1 and 1 on a per batch basis.
        """
        self.bins = bins
        self.sbins = sbins
        self.backonly = background_only
        self.power = power
        self.order = order
        self.memory = memory
        self.lambd = lambd
        self.monotonic = monotonic
        self.max_slope = max_slope
        self.normalize = normalize
        self.eps = eps
        self.dynamicbins = dynamicbins
        if not dynamicbins:self.boundaries=tf.linspace(-1.,1.,bins)
        self.m = tf.constant([])
        self.pred_long = tf.constant([])
        self.fitter = _LegendreFitter(order=order, power=power,lambd=lambd,max_slope=max_slope,monotonic=monotonic,eps=eps,dynamicbins=dynamicbins)

    def __call__(self,pred,target,x_biased,weights=None):
        """
        Calculate the total loss (flat and MSE.)


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
            mask = target==1
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
            self.m = tf.concat([self.m,x_biased],axis=0)
            self.pred_long = tf.concat([self.pred_long,pred],axis=0)
            self.pred_long = tf.stop_gradient(self.pred_long)
            m,msorted = self.m.sort()
            pred_long = self.pred_long[msorted].view(self.bins,-1)
            self.fitter.initialize(m=m.view(self.bins,-1),overwrite=True)
            m,msorted = x_biased.sort()
            pred = pred[msorted].view(self.bins,-1)
            if weights is not None:weights = weights[msorted].view(self.bins,-1)
            LLoss = _LegendreIntegral.apply(pred, weights, self.fitter, self.sbins,pred_long)
        else:
            if self.dynamicbins:
                if self.normalize:
                    x_biased = 2*(x_biased - tf.reduce_min(x_biased))/(tf.reduce_max(x_biased)-tf.reduce_min(x_biased)) -1
                msorted = tf.argsort(x_biased)
                m = tf.gather(x_biased,msorted,)
                m = view(m,(self.bins,-1))
                pred = view(tf.gather(pred,msorted),(self.bins,-1))
                if weights is not None:weights = view(tf.gather(weights,msorted),(self.bins,-1))
            else: #still need to fix nbin normalization in dervatives
                bin_index = tf.feature_column.bucketized_column(x_biased,self.boundaries)
                m = tf.gather(self.boundaries,tf.unique(bin_index)[0],batch_dims=0)
                m = tf.concat([tf.constant([-1]),m])
                binned = [pred[bin_index==index] for index in tf.unique(bin_index)[0]]
                pred = pad_sequences(binned,padding='post',value=0)
                if weights is not None:
                    binned = [weights[bin_index==index] for index in tf.unique(bin_index).y]
                    weights = tf.keras.preprocessing.sequence.pad_sequences(binned,padding='post',value=0)
            self.fitter.initialize(m=m,overwrite=True)
            if weights is None:
                weights = tf.ones_like(pred)
            LLoss = _LegendreIntegral(pred,weights, self.fitter, self.sbins)
        return LLoss

    view = lambda x,shape: tf.reshape(x,shape)
def _LegendreIntegral(input,weights=None, fitter=None,sbins=2,extra_input=None):
    @tf.custom_gradient
    def MoDeOP(input):
        s_edges = tf.linspace(0.,1.,sbins+1)#create s edges to integrate over
        s = (s_edges[1:] + s_edges[:-1])*0.5
        s = _expand_dims_as(s,input)
        ds = view(s_edges[1:] - s_edges[:-1],(1,-1))
        w = tf.reduce_sum(weights,axis=-1)/weights.shape[1]
        F = tf.reduce_sum(_Heaviside(s-input),axis=-1)/input.shape[-1] # get CDF at s from input values
        integral = view(tf.matmul(ds,(F-fitter(F))**fitter.power),-1)
        integral = tf.reduce_sum(integral*w,axis=0)/input.shape[0] # not exactly right with max_slope
        del F,s,ds,s_edges

        # Stuff for backward
        if extra_input is not None:
            input_appended = extra_input
        else:
            input_appended = input


        F_s_i =  _expand_dims_as(view(input,-1),input) #make a flat copy of input and add dimensions for boradcasting
        F_s_i =  F_s_i-input_appended
        F_s_i = _Heaviside(F_s_i)
        F_s_i =  tf.reduce_sum(F_s_i,axis=-1)/F_s_i.shape[-1] #sum over bin content to get CDF
        residual = F_s_i - fitter(F_s_i)
        def grad(grad_output):
            shape = input.shape
            lambd = fitter.lambd
            max_slope =fitter.max_slope
            monotonic =fitter.monotonic
            eps = fitter.eps
            power = fitter.power
            order = fitter.order
            dm = fitter.dm
            m = fitter.m
            a0 = view(fitter.a0,shape)
            dF = view(residual[tf.repeat(tf.eye(shape[0],dtype=bool),shape[1],axis=0)],shape)
            dF0 = -.5 * view(tf.reduce_sum(residual,axis=-1),shape) * view(dm,(-1,1))
            summation = dF + dF0
            if order >0:
                a1 = fitter.a1.view(shape)
                if max_slope is None:
                    dF1  = -1.5 * view(tf.reduce_sum((residual*m),axis=-1),shape) * view(dm*m,(-1,1))
                    summation += dF1
                else:
                    dF1   = -1.5 * view(tf.reduce_sum((residual*m),axis=-1),shape) * view(dm*m,(-1,1))*\
                             (1/tf.cosh(a1/max_slope))**2
                    summation += dF1
            if order>1:
                a2 = view(fitter.a2,shape)
                if not monotonic:
                    dF2   = -2.5* view(tf.reduce_sum(residual*.5*(3*m**2-1),axis=-1),shape) *\
                            view(dm*0.5*(3*m**2-1),(-1,1))
                    summation += dF2
                else:
                    dF2   = -2.5* view(tf.reduce_sum(residual*.5*(3*m**2-1),axis=-1),shape) *\
                            view(dm*0.5*(3*m**2-1),(-1,1)) *\
                            (1/tf.cosh(a2/(a1+eps))**2*view(-2.5*dm*0.5*(3*m**2-1),(-1,1))+1.5*a2/(a1+eps)*view(dm*m,(-1,1)) +\
                            -1.5*view(dm*m,(-1,1))*(tf.tanh(a2/(a1+eps))))
                    summation += dF2
            summation *= (-power)/np.prod(shape)
            if lambd is not None:
                summation += -lambd*2/np.prod(shape) *\
                3/2* fitter.a1.view(shape)*view(dm*m,(-1,1))

            grad_input  = grad_output * summation * view(tf.repeat(w,shape[1]),shape)

            return grad_input
        return integral, grad
    return MoDeOP(input)

class _LegendreFitter():
    def __init__(self,order=0,power=1,lambd=None,max_slope=None,monotonic=False,eps=1e-8,dynamicbins=True):
        """
        Object used to fit an array of using Legendre polynomials.

        Parameters
        ----------
        mbins :int or Array[float] (optional)
            Array of bin edges or number of bins in m used in the fit. The fit is integrated along m.
        m : Array[float] (optional)
            Array of all masses. Has shape (mbins,bincontent)
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
        Fit F with Legendre polynomials and return the fit.

        Parameters
        ----------
        F : torch.Tensor
            Tensor of CDFs F_m(s) has shape (N,mbins) where N is the number of scores
        """
        if self.initialized == False:
            raise Exception("Please run initialize method before calling or use the MoDeLoss wrapper.")
        self.a0 = 1/2 * view(tf.reduce_sum(F*self.dm,axis=-1),(-1,1)) #integrate over mbins
        fit = tf.broadcast_to(self.a0,F.shape)# make boradcastable
        if self.order>0:
            self.a1 = 3/2 * view(tf.reduce_sum(F*self.m*self.dm,axis=-1),(-1,1))
            if self.max_slope is not None:
                fit = fit + self.max_slope*tf.tanh(self.a1/self.max_slope)*self.m
            else:
                fit = fit + self.a1*self.m
        if self.order>1:
            p2 = (3*self.m**2-1)*0.5
            self.a2 = 5/2 * view((F*p2*self.dm).sum(axis=-1),(-1,1))
            if self.monotonic:
                fit = fit + self.a1*tf.tanh(self.a2/(self.a1+self.eps))*p2
            else:
                fit = fit+ self.a2*p2
        return fit

    def initialize(self,m,overwrite=True):
        if overwrite or self.initialized==False:
#             if type(m) != tf.Tensor:
#                 m = tf.constant(m)
            if self.dynamicbins:
                dm = tf.math.reduce_max(m,axis=1)[0] - tf.math.reduce_min(m,axis=1)[0]  # bin widths for each of the mbins.
                m  = tf.math.reduce_mean(m,axis=1) # bin centers
            else:
                dm = m[:-1]-m[1:]
                m  = (m[:-1]+m[1:]) * 0.5
            self.m = view(m,-1)
            self.dm = view(dm,-1)
            self.mbins = self.m.shape[0]
            self.initialized = True
        return

def _Heaviside(tensor):
    tensor = (tf.sign(tensor)+1.)*0.5
    return tensor
def _expand_dims_as(t1,t2):
    result = t1[(...,)+(None,)*len(t2.get_shape().as_list())]
    return result