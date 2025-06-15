import torch
import time, copy
import numpy as np

def train_model_org(model, criterion, optimizer, scheduler, num_epochs, trainloader, device, dataset_sizes):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    metrics = {
    'train': {
        'loss': [],
        'acc': [] },
    'valid': {
        'loss': [],
        'acc': [] },
    "bast_acc": 0        
                 }

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
          
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in trainloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        # print()
        if (epoch + 1)%10 == 0:
            torch.cuda.empty_cache()
        print(time.time() - start)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    metrics['bast_acc'] = best_acc
    model.load_state_dict(best_model_wts)
    
    return model, metrics


def train_model(model, criterion, optimizer, scheduler, num_epochs, trainloader, device, dataset_sizes):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    metrics = {
    'train': {
        'loss': [],
        'acc': [] },
    'valid': {
        'loss': [],
        'acc': [] },
    "bast_acc": 0        
                 }

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
          
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in trainloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.first_step()
                        
                        outputs = model(inputs)        
                        loss = criterion(outputs, labels)       
                        loss.backward()
                        optimizer.second_step(zero_grad=True)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        # print()
        if (epoch + 1)%10 == 0:
            torch.cuda.empty_cache()
        print(time.time() - start)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    metrics['bast_acc'] = best_acc
    model.load_state_dict(best_model_wts)
    
    return model, metrics


def train_modelt(model, criterion, optimizer, scheduler, start_epochs, num_epochs, trainloader, device, dataset_sizes):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    metrics = {
    'train': {
        'loss': [],
        'acc': [] },
    'valid': {
        'loss': [],
        'acc': [] },
    "bast_acc": 0        
                 }

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
          
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in trainloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if epoch > start_epochs:
                            loss.backward()
                            optimizer.first_step()
                            
                            outputs = model(inputs)        
                            loss = criterion(outputs, labels)       
                            loss.backward()
                            optimizer.second_step(zero_grad=True)
                        else:
                            loss.backward()
                            optimizer.step_org()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        # print()
        if (epoch + 1)%10 == 0:
            torch.cuda.empty_cache()
        print(time.time() - start)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    metrics['bast_acc'] = best_acc
    model.load_state_dict(best_model_wts)
    
    return model, metrics


def train_model_alpha(model, criterion, optimizer, scheduler, num_epochs, trainloader, device, dataset_sizes):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    metrics = {
    'train': {
        'loss': [],
        'acc': [] },
    'valid': {
        'loss': [],
        'acc': [] },
    "bast_acc": 0        
                 }

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
          
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in trainloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.first_step()
                        
                        outputs = model(inputs)        
                        loss = criterion(outputs, labels)       
                        loss.backward()
                        optimizer.second_step(zero_grad=True)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
                optimizer.update_alpha(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        # print()
        if (epoch + 1)%10 == 0:
            torch.cuda.empty_cache()
        print(time.time() - start)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    metrics['bast_acc'] = best_acc
    model.load_state_dict(best_model_wts)
    
    return model, metrics


#init_t: when to calculate the init_grad_var, gamma: attenuation coefficient, k:sliding window length for how many epochs used for calculating var of grad norm 
def train_model_timing_var(model, criterion, optimizer, scheduler, num_epochs, trainloader, device, dataset_sizes, init_t, gamma, k):
    since = time.time()

    best_model_wts = model.state_dict()
    grad_norms =[]
    best_acc = 0.0
    init_grad_var = -1
    enable_PUGD = False
    metrics = {
    'train': {
        'loss': [],
        'acc': [] },
    'valid': {
        'loss': [],
        'acc': [] },
    "bast_acc": 0,
    "enable_pugd_epoch": -1,
    "init_grad_var": -1,
    "grad_var": -1
                 }

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
          
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in trainloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if enable_PUGD:
                            optimizer.first_step()
                            
                            outputs = model(inputs)        
                            loss = criterion(outputs, labels)       
                            loss.backward()
                            optimizer.second_step(zero_grad=True)
                        else:
                            grad_norms.append(1.0/optimizer.base_step(zero_grad=True))
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # print()
                if not enable_PUGD:
                    if epoch == init_t:
                        init_grad_var = gamma * np.var(grad_norms) 
                    elif epoch > k and epoch > init_t:
                        grad_var = np.var(grad_norms[-k:])
                        if grad_var < init_grad_var: 
                            enable_PUGD = True
                            metrics['enable_pugd_epoch'] = epoch
                            metrics['init_grad_var'] = init_grad_var
                            metrics['grad_var'] = grad_var

        if (epoch + 1)%10 == 0:
            torch.cuda.empty_cache()
        print(time.time() - start)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    metrics['bast_acc'] = best_acc
    model.load_state_dict(best_model_wts)
    
    return model, metrics


#xi: num of epochs for cal delta base, mu: Proportional threshold, t:sampling period
def train_model_timing_delta(model, criterion, optimizer, scheduler, num_epochs, trainloader, device, dataset_sizes, xi, mu, t):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    delta_base = 0.0
    delta = 0.0
    delta_base = -1
    enable_PUGD = False
    metrics = {
    'train': {
        'loss': [],
        'acc': [] },
    'valid': {
        'loss': [],
        'acc': [] },
    "bast_acc": 0,
    "enable_pugd_epoch": -1,
    "delta_base": -1,
    "delta": -1
                 }

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
          
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in trainloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if enable_PUGD:
                            optimizer.first_step()
                            
                            outputs = model(inputs)        
                            loss = criterion(outputs, labels)       
                            loss.backward()
                            optimizer.second_step(zero_grad=True)
                        else:
                            optimizer.base_step_no_norm(zero_grad=True)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)
            # deep copy the model
            if phase == 'valid': 
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # print()
                if not enable_PUGD:
                    if epoch < xi:
                        delta_base += np.abs(metrics['valid']['loss'][-1] - metrics['train']['loss'][-1])
                    elif epoch == xi:
                        delta_base = delta_base/xi * mu
                    elif (epoch + 1) % t == 0:
                        delta = np.abs(metrics['valid']['loss'][-1] - metrics['train']['loss'][-1]) 
                        if delta > delta_base:
                            enable_PUGD = True
                            metrics['enable_pugd_epoch'] = epoch
                            metrics['delta_base'] = delta_base
                            metrics['delta'] = delta

        if (epoch + 1)%10 == 0:
            torch.cuda.empty_cache()
        print(time.time() - start)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    metrics['bast_acc'] = best_acc
    model.load_state_dict(best_model_wts)

    return model, metrics


def train_modelx(model, criterion, optimizer, scheduler, num_epochs, trainloader, device, dataset_sizes):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    metrics = {
    'train': {
        'loss': [],
        'acc': [] },
    'valid': {
        'loss': [],
        'acc': [] },
    "bast_acc": 0        
                 }

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
          
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in trainloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.first_step()
                        
                        # for i in range(optimizer.step_x):
                        #     outputs = model(inputs)        
                        #     loss = criterion(outputs, labels)       
                        #     loss.backward()
                        #     optimizer.test_step()
                        
                        outputs = model(inputs)        
                        loss = criterion(outputs, labels)       
                        loss.backward()
                        optimizer.second_step(zero_grad=True)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        # print()
        if (epoch + 1)%10 == 0:
            torch.cuda.empty_cache()
        print(time.time() - start)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    metrics['bast_acc'] = best_acc
    model.load_state_dict(best_model_wts)
    
    return model, metrics


def train_modelx_last(model, criterion, optimizer, scheduler, num_epochs, trainloader, device, dataset_sizes):
    since = time.time()
    metrics = {
    'train': {
        'loss': [],
        'acc': [] },
    'valid': {
        'loss': [],
        'acc': [] },
    "bast_acc": 0        
                 }
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
          
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in trainloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.first_step()
                        
                        # for i in range(optimizer.step_x):
                        #     outputs = model(inputs)        
                        #     loss = criterion(outputs, labels)       
                        #     loss.backward()
                        #     optimizer.test_last_layer_step()
                        
                        outputs = model(inputs)        
                        loss = criterion(outputs, labels)       
                        loss.backward()
                        optimizer.second_step(zero_grad=True)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            metrics[phase]['loss'].append(epoch_loss)
            metrics[phase]['acc'].append(epoch_acc)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        # print()
        if (epoch + 1)%10 == 0:
            torch.cuda.empty_cache()
        print(time.time() - start)
            
    metrics['bast_acc'] = best_acc
    model.load_state_dict(best_model_wts)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model, metrics



from torch.cuda.amp  import autocast, GradScaler 

def train_fp16(model, criterion, optimizer, scheduler, num_epochs, trainloader, device, dataset_sizes):
    # 初始化梯度缩放器（防止下溢）
    scaler = GradScaler('cuda')
    
    for epoch in range(num_epochs):
        for inputs, labels in trainloader['train']:
            inputs, labels = inputs.to(device),  labels.to(device) 
            
            # 前向传播（自动混合精度）
            with autocast('cuda', dtype=torch.float16):   # PyTorch 2.3+支持显式指定dtype 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 反向传播与梯度缩放 
            scaler.scale(loss).backward() 
            scaler.step(optimizer) 
            scaler.update() 
            optimizer.zero_grad() 

# from torch import accelerator
 
# # 初始化加速器（自动检测最优后端）
# acc = accelerator.Accelerator(
#     mixed_precision='fp8',  # 支持fp8/fp16/bf16
#     log_level="DEBUG"
# )
 
# # 包装模型与数据
# model, optimizer, trainloader = acc.prepare( 
#     model, optimizer, trainloader
# ) 
 
# # 自动处理梯度缩放与设备迁移 
# with acc.accumulate(model):   # 梯度累积兼容 
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     acc.backward(loss) 
