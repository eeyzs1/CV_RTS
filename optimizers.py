# -*- coding: utf-8 -*-
"""
PUGD in Pytorch
by Ching-Hsun Tseng
"""
from math import pi
from numpy import cos, sin
import torch


class PUGDX(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, step_x = 0, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDX, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
        self.step_x = step_x

        
    @torch.no_grad()
    def step(self, closure=None):              
        assert closure is not None, 'There should be a closure'
        closure = torch.enable_grad()(closure)
        self.first_step()
        closure()
        self.second_step(zero_grad=True)
        
    @torch.no_grad()
    def xp_step(self, zero_grad=True):  
        '''UGD = NGD-FW in Tensor'''
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.grad = p.grad / (grad_norm + 1e-12)   
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def first_step(self):
        abs_grad_norm = self._abs_grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = torch.abs(p) * p.grad/ (abs_grad_norm + 1e-12)
                p.add_(self.state[i]["e_w"])

    @torch.no_grad()
    def test_step(self):
        abs_grad_norm = self._abs_grad_norm()
        for group in self.param_groups:       
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                temp_e_w = torch.abs(p) * p.grad/ (abs_grad_norm + 1e-12)
                p.add_(temp_e_w)
                self.state[i]["e_w"] += temp_e_w

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad / (grad_norm + 1e-12)

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm.cpu()

    @torch.no_grad()
    def test_last_layer_step(self):
        for group in self.param_groups:
            abs_grad_norm = self._abs_grad_norm(group)
            for i, p in enumerate(group["params"]):
                if hasattr(p,'last') and p.last:
                    if p.grad is None: continue
                    temp_e_w = torch.abs(p) * p.grad/ (abs_grad_norm + 1e-12)
                    p.add_(temp_e_w)
                    self.state[i]["e_w"] += temp_e_w

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm
    
    def _abs_grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm


class PUGDXA(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, step_x = 0, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDXA, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
        self.step_x = step_x

        
    @torch.no_grad()
    def step(self, closure=None):              
        assert closure is not None, 'There should be a closure'
        closure = torch.enable_grad()(closure)
        self.first_step()
        closure()
        self.second_step(zero_grad=True)
        
    @torch.no_grad()
    def xp_step(self, zero_grad=True):  
        '''UGD = NGD-FW in Tensor'''
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.grad = p.grad / (grad_norm + 1e-12)   
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def first_step(self):
        abs_grad_norm = self._abs_grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = 2 * torch.abs(p) * p.grad/ (abs_grad_norm + 1e-12)
                p.add_(self.state[i]["e_w"])
                p.grad *= 2

    @torch.no_grad()
    def test_step(self):
        abs_grad_norm = self._abs_grad_norm()
        for group in self.param_groups:       
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                temp_e_w = torch.abs(p) * p.grad/ (abs_grad_norm + 1e-12)
                p.add_(temp_e_w)
                self.state[i]["e_w"] += temp_e_w

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad / (2 * grad_norm + 1e-12)

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm.cpu()

    @torch.no_grad()
    def test_last_layer_step(self):
        for group in self.param_groups:
            abs_grad_norm = self._abs_grad_norm(group)
            for i, p in enumerate(group["params"]):
                if hasattr(p,'last') and p.last:
                    if p.grad is None: continue
                    temp_e_w = torch.abs(p) * p.grad/ (abs_grad_norm + 1e-12)
                    p.add_(temp_e_w)
                    self.state[i]["e_w"] += temp_e_w

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm
    
    def _abs_grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm



class PUGDXCOS(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, step_x = 0, max_epochs = 40, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDXCOS, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
        self.step_x = step_x
        self.max_epochs = max_epochs
        self.alpha = 2

    # the alpha range from 2.1 to 0.1
    def update_alpha(self, epoch):
        self.alpha = 2 * (cos(epoch/self.max_epochs * pi/2)) + 0.1
        
    @torch.no_grad()
    def step(self, closure=None):              
        assert closure is not None, 'There should be a closure'
        closure = torch.enable_grad()(closure)
        self.first_step()
        closure()
        self.second_step(zero_grad=True)

    @torch.no_grad()
    def first_step(self):
        abs_grad_norm = self._abs_grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = torch.abs(p) * p.grad/ (abs_grad_norm + 1e-12)
                p.add_(self.state[i]["e_w"])
                # p.grad *= self.alpha

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad / (grad_norm + 1e-12)

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm.cpu()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm
    
    def _abs_grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm


class PUGDXSIN(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, step_x = 0, max_epochs = 40, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDXSIN, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
        self.step_x = step_x
        self.max_epochs = max_epochs
        self.alpha = 2

    # the alpha range from 0.1 to 2.1
    def update_alpha(self, epoch):
        self.alpha = 2 * (sin(epoch/self.max_epochs * pi/2)) + 0.1
        
    @torch.no_grad()
    def step(self, closure=None):              
        assert closure is not None, 'There should be a closure'
        closure = torch.enable_grad()(closure)
        self.first_step()
        closure()
        self.second_step(zero_grad=True)

    @torch.no_grad()
    def first_step(self):
        abs_grad_norm = self._abs_grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = torch.abs(p) * p.grad/ (abs_grad_norm + 1e-12)
                p.add_(self.state[i]["e_w"])
                # p.grad *= self.alpha

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad / (grad_norm + 1e-12)

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm.cpu()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm
    
    def _abs_grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm

class SAMX(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, step_x = 0, r_ho = 1, **kwargs):
        defaults = dict(**kwargs)
        super(SAMX, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
        self.step_x = step_x
        self.rho = r_ho

        
    @torch.no_grad()
    def step(self, closure=None):              
        assert closure is not None, 'There should be a closure'
        closure = torch.enable_grad()(closure)
        self.first_step()
        closure()
        self.second_step(zero_grad=True)
        
    @torch.no_grad()
    def xp_step(self, zero_grad=True):  
        '''UGD = NGD-FW in Tensor'''
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.grad = p.grad / (grad_norm + 1e-12)   
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = p.grad/ (grad_norm + 1e-12)
                p.add_(self.state[i]["e_w"])

    @torch.no_grad()
    def test_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:       
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                temp_e_w = p.grad/ (grad_norm + 1e-12)
                p.add_(temp_e_w)
                self.state[i]["e_w"] += temp_e_w

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad / (grad_norm + 1e-12)

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm.cpu()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return norm
