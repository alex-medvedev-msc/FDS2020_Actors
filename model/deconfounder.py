# Imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys

import pyro
from pyro.distributions import Bernoulli
from pyro.distributions import Delta
from pyro.distributions import Normal
from pyro.distributions import Uniform
from pyro.distributions import LogNormal
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.infer import Predictive
from pyro.optim import Adam
import torch.distributions.constraints as constraints
pyro.set_rng_seed(101)


def f_z(params):
    """Samples from P(Z)"""    
    z_mean0 = params['z_mean0']
    z_std0 = params['z_std0']
    z = pyro.sample("z", Normal(loc = z_mean0, scale = z_std0))
    return z

def f_x(z, params):
    """
    Samples from P(X|Z)
    
    P(X|Z) is a Bernoulli with E(X|Z) = logistic(Z * W),
    where W is a parameter (matrix).  In training the W is
    hyperparameters of the W distribution are estimated such
    that in P(X|Z), the elements of the vector of X are
    conditionally independent of one another given Z.
    """
    def sample_W():
        """
        Sample the W matrix
        
        W is a parameter of P(X|Z) that is sampled from a Normal
        with location and scale hyperparameters w_mean0 and w_std0
        """
        w_mean0 = params['w_mean0']
        w_std0 = params['w_std0']
        W = pyro.sample("W", Normal(loc = w_mean0, scale = w_std0))
        return W
    W = sample_W()
    linear_exp = torch.matmul(z, W)
    # sample x using the Bernoulli likelihood
    x = pyro.sample("x", Bernoulli(logits = linear_exp))
    return x

def f_y(x, z, params):
    """
    Samples from P(Y|X, Z)
    
    Y is sampled from a Gaussian where the mean is an
    affine combination of X and Z.  Bayesian linear
    regression is used to estimate the parameters of
    this affine transformation  function.  Use torch.nn.Module to create
    the Bayesian linear regression component of the overall
    model.
    """
    predictors = torch.cat((x, z), 1)

    w = pyro.sample('weight', Normal(params['weight_mean0'], params['weight_std0']))
    b = pyro.sample('bias', Normal(params['bias_mean0'], params['bias_std0']))

    y_hat = (w * predictors).sum(dim=1) + b
    # variance of distribution centered around y
    sigma = pyro.sample('sigma', Normal(params['sigma_mean0'], params['sigma_std0']))
    with pyro.iarange('data', len(predictors)):
        pyro.sample('y', Normal(y_hat, sigma))
        return y_hat
    
    
def f_y2(x, z, params):
    """
    Samples from P(Y|X, Z)
    
    Y is sampled from a Gaussian where the mean is an
    affine combination of X and Z.  Bayesian linear
    regression is used to estimate the parameters of
    this affine transformation  function.  Use torch.nn.Module to create
    the Bayesian linear regression component of the overall
    model.
    """
    predictors = torch.cat((x, z), 1)

    w = pyro.sample('weight', Normal(params['weight_mean0'], params['weight_std0']))
    w2 = pyro.sample('weight2', Normal(params['weight_mean0'], params['weight_std0']))
    
    b = pyro.sample('bias', Normal(params['bias_mean0'], params['bias_std0']))

    y_hat = (w * predictors).sum(dim=1) + b + (w2 * predictors**2).sum(dim=1)
    # variance of distribution centered around y
    sigma = pyro.sample('sigma', Normal(params['sigma_mean0'], params['sigma_std0']))
    with pyro.iarange('data', len(predictors)):
        pyro.sample('y', Normal(y_hat, sigma))
        return y_hat

    
def model(params):
    """The full generative causal model"""
    z = f_z(params)
    x = f_x(z, params)
    y = f_y(x, z, params)
    return {'z': z, 'x': x, 'y': y}

def step1_guide(params):
    """
    Guide function for fitting P(Z) and P(X|Z) from data
    """
    # Infer z hyperparams
    qz_mean = pyro.param("qz_mean", params['z_mean0'])
    qz_stddv = pyro.param("qz_stddv", params['z_std0'],
                         constraint=constraints.positive)
    
    z = pyro.sample("z", Normal(loc = qz_mean, scale = qz_stddv))
    
    # Infer w params
    qw_mean = pyro.param("qw_mean", params["w_mean0"])
    qw_stddv = pyro.param("qw_stddv", params["w_std0"],
                          constraint=constraints.positive)
    W = pyro.sample("W", Normal(loc = qw_mean, scale = qw_stddv))
    
def step2_guide(params):
    # Z and W are just sampled using param values optimized in previous step
    z = pyro.sample("z", Normal(loc = params['qz_mean'], scale = params['qz_stddv']))
    W = pyro.sample("W", Normal(loc = params['qw_mean'], scale = params['qw_stddv']))
    
    # Infer regression params
    # parameters of (w : weight)
    w_loc = pyro.param('w_loc', params['weight_mean0'])
    w_scale = pyro.param('w_scale', params['weight_std0'])
    
    #w2_loc = pyro.param('w2_loc', params['weight2_mean0'])
    #w2_scale = pyro.param('w2_scale', params['weight2_std0'])

    # parameters of (b : bias)
    b_loc = pyro.param('b_loc', params['bias_mean0'])
    b_scale = pyro.param('b_scale', params['bias_std0'])
    # parameters of (sigma)
    sigma_loc = pyro.param('sigma_loc', params['sigma_mean0'])
    sigma_scale = pyro.param('sigma_scale', params['sigma_std0'])

    # sample (w, b, sigma)
    w = pyro.sample('weight', Normal(w_loc, w_scale))
    #w2 = pyro.sample('weight2', Normal(w2_loc, w2_scale))
    
    b = pyro.sample('bias', Normal(b_loc, b_scale))
    sigma = pyro.sample('sigma', Normal(sigma_loc, sigma_scale))

    
class Deconfounder():
    def __init__(self, step1_opt, step2_opt, latent_dim=50, step1_iters=2000, step2_iters=1500, seed=101):
        self.step1_opt = step1_opt
        self.step2_opt = step2_opt
        self.step1_iters = step1_iters
        self.step2_iters = step2_iters
        self.latent_dim = latent_dim
        pyro.set_rng_seed(seed)
        
    def step1_train(self, x_data, params):

        conditioned_on_x = pyro.condition(model, data = {"x" : x_data})
        svi = SVI(conditioned_on_x, step1_guide, self.step1_opt, loss=Trace_ELBO())

        print("\n Training Z marginal and W parameter marginal...")

        # do gradient steps
        pyro.get_param_store().clear()
        for step in range(self.step1_iters):
            loss = svi.step(params)
            if step % 100 == 0:
                print("[iteration %04d] loss: %.4f" % (step + 1, loss/len(x_data)))

        # grab the learned variational parameters

        updated_params = {k: v for k, v in params.items()}
        for name, value in pyro.get_param_store().items():
            print("Updating value of hypermeter{}".format(name))
            updated_params[name] = value.detach()

        return updated_params

    def step2_train(self, x_data, y_data, params, num_samples=1000):
        print("Training Bayesian regression parameters...")
        pyro.clear_param_store()
        # Create a regression model
        conditioned_on_x_and_y = pyro.condition(model, data = {
            "x": x_data,
            "y": y_data
        })

        svi = SVI(conditioned_on_x_and_y, step2_guide, self.step2_opt, loss=Trace_ELBO(), num_samples=num_samples)
        for step in range(self.step2_iters):
            loss = svi.step(params)
            if step % 100 == 0:
                print("[iteration %04d] loss: %.4f" % (step + 1, loss/len(x_data)))

        updated_params = {k: v for k, v in params.items()}
        for name, value in pyro.get_param_store().items():
            print("Updating value of hypermeter: {}".format(name))
            updated_params[name] = value.detach()
            
        print("Training complete.")
        return updated_params
    
    def train(self, X_train, y_train, num_samples=1000):
        num_datapoints, data_dim = X_train.shape

        params0 = {
            'z_mean0': torch.zeros([num_datapoints, self.latent_dim]),
            'z_std0' : torch.ones([num_datapoints, self.latent_dim]),
            'w_mean0' : torch.zeros([self.latent_dim, data_dim]),
            'w_std0' : torch.ones([self.latent_dim, data_dim]),
            'weight_mean0': torch.zeros(data_dim + self.latent_dim),
            'weight_std0': torch.ones(data_dim + self.latent_dim),
            #'weight2_mean0': torch.zeros(data_dim + self.latent_dim),
            #'weight2_std0': torch.ones(data_dim + self.latent_dim),
            'bias_mean0': torch.tensor(0.),
            'bias_std0': torch.tensor(1.),
            'sigma_mean0' : torch.tensor(1.),
            'sigma_std0' : torch.tensor(0.05)
        } # These are our priors

        params1 = self.step1_train(X_train, params0)
        params2 = self.step2_train(X_train, y_train, params1, num_samples=num_samples)
        self.step1_params = params1
        self.step2_params = params2
        self.test_params = None
        return params1, params2
    
    def condition_causal(self, their_tensors, absent_tensors, movie_inds, num_samples=1000):
        
        their_cond = pyro.condition(model, data = {"x" : their_tensors})
        absent_cond = pyro.condition(model, data = {"x" : absent_tensors})

        their_y = []
        for _ in range(num_samples):
            their_y.append(torch.sum(their_cond(self.step2_params)['y'][movie_inds]).item())

        absent_y = []
        for _ in range(num_samples):
            absent_y.append(torch.sum(absent_cond(self.step2_params)['y'][movie_inds]).item())

        their_mean = np.mean(their_y)
        absent_mean = np.mean(absent_y)
        causal_effect_noconf = their_mean - absent_mean

        return causal_effect_noconf
    
    def do_causal(self, their_tensors, absent_tensors, movie_inds, num_samples=1000):
        # With confounding
        their_do = pyro.do(model, data = {"x" : their_tensors})
        absent_do = pyro.do(model, data = {"x" : absent_tensors})

        their_do_y = []
        for _ in range(num_samples):
            their_do_y.append(torch.sum(their_do(self.step2_params)['y'][movie_inds]).item())

        absent_do_y = []
        for _ in range(num_samples):
            absent_do_y.append(torch.sum(absent_do(self.step2_params)['y'][movie_inds]).item())

        their_do_mean = np.mean(their_do_y)
        absent_do_mean = np.mean(absent_do_y)
        causal_effect_conf = their_do_mean - absent_do_mean

        return causal_effect_conf
    
    def infer_z(self, X_test):
        num_datapoints, data_dim = X_test.shape

        params0 = {
            'z_mean0': torch.zeros([num_datapoints, self.latent_dim]),
            'z_std0' : torch.ones([num_datapoints, self.latent_dim]),
            'w_mean0' : self.step1_params['w_mean0'],
            'w_std0' : self.step1_params['w_std0'],
            #'w_mean0' : torch.zeros([self.latent_dim, data_dim]),
            #'w_std0' : torch.ones([self.latent_dim, data_dim]),
            'weight_mean0': self.step1_params['weight_mean0'],
            'weight_std0': self.step1_params['weight_std0'],
            #'weight2_mean0': self.step1_params['weight2_mean0'],
            #'weight2_std0': self.step1_params['weight2_std0'],
            'bias_mean0': self.step1_params['bias_mean0'],
            'bias_std0': self.step1_params['bias_std0'],
            'sigma_mean0' : self.step1_params['sigma_mean0'],
            'sigma_std0' : self.step1_params['sigma_std0']
        } # These are our priors

        self.test_params = self.step1_train(X_test, params0)
    
    def do_predict(self, X_test, num_samples=1000):
        
        if self.test_params is None:
            self.infer_z(X_test)
        
        do = pyro.do(model, data = {"x" : X_test})
        predictions = np.zeros((X_test.shape[0]))
        for _ in range(num_samples):
            y_pred = do(self.test_params)['y']
            predictions += y_pred.detach().numpy()
        
        return predictions / num_samples, self.test_params
        #x_do = pyro.do(f_x, data = {"x" : X_test})
        
    def cond_predict(self, X_test, num_samples=1000):
        if self.test_params is None:
            self.infer_z(X_test)
        
        cond = pyro.condition(model, data = {"x" : X_test})
        predictions = np.zeros((X_test.shape[0]))
        for _ in range(num_samples):
            y_pred = cond(self.test_params)['y']
            predictions += y_pred.detach().numpy()
        
        return predictions / num_samples, self.test_params
    
    '''def do_predict(self, X_test, num_samples=1000):
        
        predictive = Predictive(model, posterior_samples={'x': X_test},
                                guide=None, return_sites=())
        
        return predictive'''