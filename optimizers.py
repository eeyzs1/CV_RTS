# -*- coding: utf-8 -*-
"""
PUGD in Pytorch
by Ching-Hsun Tseng
"""
from math import pi
from numpy import cos, sin
import torch


class PUGDX(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDX, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']

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
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.grad = p.grad * grad_norm_reciprocal   
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def first_step(self):
        abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                p.add_(self.state[i]["e_w"])

    @torch.no_grad()
    def test_step(self):
        abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal()
        for group in self.param_groups:       
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                temp_e_w = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                p.add_(temp_e_w)
                self.state[i]["e_w"] += temp_e_w

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad * grad_norm_reciprocal

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm_reciprocal.cpu()

    @torch.no_grad()
    def test_last_layer_step(self):
        for group in self.param_groups:
            abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal(group)
            for i, p in enumerate(group["params"]):
                if hasattr(p,'last') and p.last:
                    if p.grad is None: continue
                    temp_e_w = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                    p.add_(temp_e_w)
                    self.state[i]["e_w"] += temp_e_w

    def _grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)
    
    def _abs_grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)


# R for perturbation radius
class PUGDXRS(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, min_beta_r = 0.1, max_beta_r = 3, min_beta_s = 0.1, max_beta_s = 3, method_r = 'sin', method_s = 'sin', max_epochs = 40, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDXRS, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
        self.max_epochs = max_epochs
        self.min_beta_r = min_beta_r
        self.max_beta_r = max_beta_r
        self.method_r = method_r
        self.min_beta_s = min_beta_s
        self.max_beta_s = max_beta_s
        self.method_s = method_s
        self.update_alpha(0)

    # sin: the alpha range from min to max, cos: range from max to min, then the magnitude inverse
    def update_alpha(self, epoch):
        match self.method_r:
            case 'isin':
                self.alpha_r = 1.0/(self.min_beta_r + (self.max_beta_r - self.min_beta_r) * sin(epoch/self.max_epochs * pi/2))
            case 'icos':
                self.alpha_r = 1.0/(self.min_beta_r + (self.max_beta_r - self.min_beta_r) * cos(epoch/self.max_epochs * pi/2))
            case 'sin':
                self.alpha_r = self.min_beta_r + (self.max_beta_r - self.min_beta_r) * sin(epoch/self.max_epochs * pi/2)
            case 'cos':
                self.alpha_r = self.min_beta_r + (self.max_beta_r - self.min_beta_r) * cos(epoch/self.max_epochs * pi/2)
        match self.method_s:
            case 'isin':
                self.alpha_s = 1.0/(self.min_beta_s + (self.max_beta_s - self.min_beta_s) * sin(epoch/self.max_epochs * pi/2))
            case 'icos':
                self.alpha_s = 1.0/(self.min_beta_s + (self.max_beta_s - self.min_beta_s) * cos(epoch/self.max_epochs * pi/2))
            case 'sin':
                self.alpha_s = self.min_beta_s + (self.max_beta_s - self.min_beta_s) * sin(epoch/self.max_epochs * pi/2)
            case 'cos':
                self.alpha_s = self.min_beta_s + (self.max_beta_s - self.min_beta_s) * cos(epoch/self.max_epochs * pi/2)
        
    @torch.no_grad()
    def step(self, closure=None):              
        assert closure is not None, 'There should be a closure'
        closure = torch.enable_grad()(closure)
        self.first_step()
        closure()
        self.second_step(zero_grad=True)

    @torch.no_grad()
    def first_step(self):
        abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal() * self.alpha_r
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                p.add_(self.state[i]["e_w"])
                p.grad *= self.alpha_s

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad * grad_norm_reciprocal

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm_reciprocal.cpu()

    def _grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)
    
    def _abs_grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)

# R for perturbation radius
class PUGDXR(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, min_beta = 0.1, max_beta = 3, method = 'sin', max_epochs = 40, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDXR, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
        self.max_epochs = max_epochs
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.method = method
        self.update_alpha(0)

    # sin: the alpha range from min to max, cos: range from max to min, then the magnitude inverse
    def update_alpha(self, epoch):
        match self.method:
            case 'isin':
                self.alpha = 1.0/(self.min_beta + (self.max_beta - self.min_beta) * sin(epoch/self.max_epochs * pi/2))
            case 'icos':
                self.alpha = 1.0/(self.min_beta + (self.max_beta - self.min_beta) * cos(epoch/self.max_epochs * pi/2))
            case 'sin':
                self.alpha = self.min_beta + (self.max_beta - self.min_beta) * sin(epoch/self.max_epochs * pi/2)
            case 'cos':
                self.alpha = self.min_beta + (self.max_beta - self.min_beta) * cos(epoch/self.max_epochs * pi/2)
        
    @torch.no_grad()
    def step(self, closure=None):              
        assert closure is not None, 'There should be a closure'
        closure = torch.enable_grad()(closure)
        self.first_step()
        closure()
        self.second_step(zero_grad=True)

    @torch.no_grad()
    def first_step(self):
        abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal() * self.alpha
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                p.add_(self.state[i]["e_w"])
                # p.grad *= self.alpha

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad * grad_norm_reciprocal

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm_reciprocal.cpu()

    def _grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)
    
    def _abs_grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)

# T for timing
class PUGDXT(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDXT, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']

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
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.grad = p.grad * grad_norm_reciprocal   
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def first_step(self):
        abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                p.add_(self.state[i]["e_w"])

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad * grad_norm_reciprocal

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm_reciprocal.cpu()

    @torch.no_grad()
    def base_step(self, zero_grad=False):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
        return grad_norm_reciprocal.cpu()

    @torch.no_grad()
    def base_step_no_norm(self, zero_grad=False):
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)
    
    def _abs_grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)



# S for scale of gradient
class PUGDXS(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, min_beta = 0.1, max_beta = 3, method = 'sin', max_epochs = 40, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDXS, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
        self.max_epochs = max_epochs
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.method = method
        self.update_alpha(0)

    # sin: the alpha range from min to max, cos: range from max to min, then the magnitude inverse
    def update_alpha(self, epoch):
        match self.method:
            case 'isin':
                self.alpha = 1.0/(self.min_beta + (self.max_beta - self.min_beta) * sin(epoch/self.max_epochs * pi/2))
            case 'icos':
                self.alpha = 1.0/(self.min_beta + (self.max_beta - self.min_beta) * cos(epoch/self.max_epochs * pi/2))
            case 'sin':
                self.alpha = self.min_beta + (self.max_beta - self.min_beta) * sin(epoch/self.max_epochs * pi/2)
            case 'cos':
                self.alpha = self.min_beta + (self.max_beta - self.min_beta) * cos(epoch/self.max_epochs * pi/2)

    @torch.no_grad()
    def first_step(self):
        abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                p.add_(self.state[i]["e_w"])
                p.grad *= self.alpha

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad * grad_norm_reciprocal
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
        return grad_norm_reciprocal.cpu()

    def _grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)
    
    def _abs_grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)



class PUGDXA(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDXA, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']

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
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.grad = p.grad * grad_norm_reciprocal   
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def first_step(self):
        abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = 2 * torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                p.add_(self.state[i]["e_w"])
                p.grad *= 2

    @torch.no_grad()
    def test_step(self):
        abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal()
        for group in self.param_groups:       
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                temp_e_w = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                p.add_(temp_e_w)
                self.state[i]["e_w"] += temp_e_w

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad / (2 * grad_norm_reciprocal + 1e-12)

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm_reciprocal.cpu()

    @torch.no_grad()
    def test_last_layer_step(self):
        for group in self.param_groups:
            abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal(group)
            for i, p in enumerate(group["params"]):
                if hasattr(p,'last') and p.last:
                    if p.grad is None: continue
                    temp_e_w = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                    p.add_(temp_e_w)
                    self.state[i]["e_w"] += temp_e_w

    def _grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)
    
    def _abs_grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)



class PUGDXCOS(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, max_epochs = 40, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDXCOS, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
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
        abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                p.add_(self.state[i]["e_w"])
                # p.grad *= self.alpha

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad * grad_norm_reciprocal

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm_reciprocal.cpu()

    def _grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)
    
    def _abs_grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)


class PUGDXSIN(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, max_epochs = 40, **kwargs):
        defaults = dict(**kwargs)
        super(PUGDXSIN, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
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
        abs_grad_norm_reciprocal = self._abs_grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = torch.abs(p) * p.grad * abs_grad_norm_reciprocal
                p.add_(self.state[i]["e_w"])
                # p.grad *= self.alpha

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad * grad_norm_reciprocal

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm_reciprocal.cpu()

    def _grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)
    
    def _abs_grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        (torch.abs(p)*p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)



class SAMX(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, r_ho = 1, **kwargs):
        defaults = dict(**kwargs)
        super(SAMX, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.wd = self.param_groups[0]['weight_decay']
        self.mom = self.param_groups[0]['momentum']
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
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.grad = p.grad * grad_norm_reciprocal   
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def first_step(self):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                self.state[i]["e_w"] = p.grad * grad_norm_reciprocal
                p.add_(self.state[i]["e_w"])

    @torch.no_grad()
    def test_step(self):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:       
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                temp_e_w = p.grad * grad_norm_reciprocal
                p.add_(temp_e_w)
                self.state[i]["e_w"] += temp_e_w

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm_reciprocal = self._grad_norm_reciprocal()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                p.sub_(self.state[i]["e_w"])
                p.grad = p.grad * grad_norm_reciprocal

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()
        
        return grad_norm_reciprocal.cpu()

    def _grad_norm_reciprocal(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                   p=2
               )
        return 1/(norm + 1e-12)
