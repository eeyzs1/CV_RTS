import numpy as np 
from tqdm import tqdm 
import torch 
import matplotlib.pyplot as plt 
from matplotlib.colors  import LogNorm
import matplotlib.cm  as cm 
from itertools import cycle 
from matplotlib.lines  import Line2D 

def get_weights(model, trainable_idx):
    ''' Extract parameters from model, and return a list of tensors'''
    paras = list(model.parameters())
    return [paras[i].data.clone() for i in trainable_idx]

def get_diff_weights(start_weights, end_weights2):
    ''' Produce a direction from 'weights' to 'weights2'.'''
    return [w2 - w for (w, w2) in zip(start_weights, end_weights2)]


# optimizers = {
#         'SGD': torch.optim.SGD(model.parameters(),  lr=0.1),
#         'Adam': torch.optim.Adam(model.parameters(),  lr=0.01),
#         'PUGD': PUGD(model.parameters(),  lr=0.1)  # 自定义优化器 
#     }
#The input model should be the one that not trained or used for pretraining
def compare_optimizers_data_collector(model, trainloader, scheduler, device, dataset_sizes, epochs, criterion, optimizers):
    '''测试不同优化器的轨迹'''
    trajectories = {}
    trainable_idx = [i for i, p in enumerate(model.parameters())  if p.requires_grad] 
    initial_weights = get_weights(model, trainable_idx)
    for name, optimizer in optimizers.items():
        weights = []
        directions = []
        trajectories[name] = {}

        metrics = {
            'train': {
                'loss': [],
                'acc': [] },
            'valid': {
                'loss': [],
                'acc': [] }     
                        }

        for epoch in tqdm(range(epochs), desc=f'{name}'):
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
                            if name == 'PUGDX':
                                loss.backward()
                                optimizer.first_step()
                                for i in range(optimizer.step_x):
                                    outputs = model(inputs)        
                                    loss = criterion(outputs, labels)       
                                    loss.backward()
                                    optimizer.test_step()
                                outputs = model(inputs)        
                                loss = criterion(outputs, labels)       
                                loss.backward()
                                optimizer.second_step(zero_grad=True)
                            else:
                                loss.backward()
                                optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                    cur_weights = get_weights(model, trainable_idx)
                    weights.append(cur_weights)
                    directions.append(get_diff_weights(initial_weights, cur_weights))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                metrics[phase]['loss'].append(epoch_loss)
                metrics[phase]['acc'].append(epoch_acc)

        trajectories[name]['metrics'] = metrics
        trajectories[name]['weights'] = weights
        trajectories[name]['directions'] = directions

        parameters = list(model.parameters())
        for i in trainable_idx:
            parameters[i].data = initial_weights[i]

        return model

#(0, (3, 1)) 是 Matplotlib自定义虚线模式 的元组表示，其含义为：
# (0, (3, 1))	━━━ ━━━ ━━━	'dashdot'变体
# (0, (5, 1))	━━━━━ ━━━━━	更长实线段
# 第一个数字 0：相位偏移（单位：像素），通常为0表示从起点开始
# 元组 (3, 1)：描述虚线模式：
# 3：实线段长度（3像素）
# 1：空白段长度（1像素）
def generate_styles(n):
        '''生成自动扩展的样式配置'''
        color_map = cm.rainbow(np.linspace(0,  1, n))
        marker_cycle = cycle(['o', '^', 's', 'D', 'P', '*', 'X', 'd', 'h', 'v'])
        linestyle_cycle = cycle(['-', '--', ':', '-.', (0, (3, 1))])
    
        return [
            {
                'color': color_map[i],
                'marker': next(marker_cycle),
                'linestyle': next(linestyle_cycle)
            } 
            for i in range(n)
        ]

def visualize_3d_trajectories(x_dir, y_dir, trajectories, proj_method, show_loss):
    for name, data in trajectories.items():
        directions = data['directions']
        x_coords, y_coords =  zip(*[project_2D(d, x_dir, y_dir, proj_method) for d in directions])
        trajectories[name]['x_coords'] = x_coords
        trajectories[name]['y_coords'] = y_coords
        
        # 存储坐标 
        trajectories[name].update({
            'x_coords': np.array(x_coords),   # 转为numpy数组便于后续处理 
            'y_coords': np.array(y_coords) 
            })

    fig = plt.figure(figsize=(12,  8))
    ax = fig.add_subplot(111,  projection='3d')

    # 使用示例 
    style_config = dict(zip(trajectories.keys(),  generate_styles(len(trajectories))))

    for name, data in trajectories.items(): 
        X = np.array(data['x_coords']) 
        Y = np.array(data['y_coords']) 
        Z = np.array(data['loss']) if show_loss else np.array(data['acc'])
        
        # 绘制折线 
        ax.plot(X,  Y, Z, 
                color=style_config[name]['color'],
                linestyle=style_config[name]['linestyle'],
                linewidth=2,
                label=name)
        
        # 标记数据点（每隔5个点标记一次）
        ax.scatter(X[::5],  Y[::5], Z[::5],
                color=style_config[name]['color'],
                marker=style_config[name]['marker'],
                s=50,  # 点大小 
                edgecolors='w')  # 点边缘色 
    
 
    # 1. 创建自定义图例句柄 
    legend_elements = [
        Line2D([0], [0],
            color=style_config[name]['color'],
            linestyle=style_config[name]['linestyle'],
            marker=style_config[name]['marker'],
            markersize=8,
            label=name,
            linewidth=2)
        for name in trajectories.keys() 
    ]
    
    # 2. 应用自定义图例 
    ax.legend(handles=legend_elements,  loc='best')
    
    # 3. 可选增强配置 
    plt.setp(ax.get_legend().get_texts(),  fontsize=10)  # 统一字体大小 

    # 4. 图形优化 
    ax.set_xlabel('X  Axis (Weight Dim 1)', labelpad=15)
    ax.set_ylabel('Y  Axis (Weight Dim 2)', labelpad=15)
    ax.set_zlabel('Z  Axis (Loss)' if show_loss else 'Z  Axis (Accuracy)', labelpad=15)
    ax.set_title('Optimization  Trajectories in 3D Space', pad=20)
    ax.grid(True,  alpha=0.3)
    
    # 5. 视角调整 
    ax.view_init(elev=25,  azim=45)  # 仰角25度，方位角45度 
    
    plt.show()
    plt.tight_layout() 
    plt.savefig('optim_comparison.png',  dpi=300, bbox_inches='tight')


def visualize_3d_trajectories_and_loss_landscape(x_dir, y_dir, trajectories, proj_method, n_grid, show_loss):
    # 使用无穷大初始化 
    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = float('inf'), -float('inf')
    for name, data in trajectories.items():
        directions = data['directions']
        x_coords, y_coords =  zip(*[project_2D(d, x_dir, y_dir, proj_method) for d in directions])
        trajectories[name]['x_coords'] = x_coords
        trajectories[name]['y_coords'] = y_coords
        # 更新全局极值 
        x_min = min(x_min, min(x_coords))
        x_max = max(x_max, max(x_coords))
        y_min = min(y_min, min(y_coords))
        y_max = max(y_max, max(y_coords))
        
        # 存储坐标 
        trajectories[name].update({
            'x_coords': np.array(x_coords),   # 转为numpy数组便于后续处理 
            'y_coords': np.array(y_coords) 
            })


    x_pad, y_pad = (x_max-x_min)*0.05, (y_max-y_min)*0.05 
    x_range = (x_min-x_pad, x_max+x_pad)
    y_range = (y_min-y_pad, y_max+y_pad)
 
    # 2. 生成动态网格 
    X = np.linspace(*x_range,  n_grid)
    Y = np.linspace(*y_range,  n_grid)
    X, Y = np.meshgrid(X,  Y)



def project_1D(w, d):
    ''' Project vector w to vector d and get the length of the projection.

        Args:
            w: vectorized weights
            d: vectorized direction

        Returns:
            the projection scalar
    '''
    assert len(w) == len(d), 'dimension does not match for w and '
    scale = torch.dot(w, d)/d.norm()
    return scale.item()


def project_2D(d, dx, dy, proj_method):
    ''' Project vector d to the plane spanned by dx and dy.

        Args:
            d: vectorized weights
            dx: vectorized direction
            dy: vectorized direction
            proj_method: projection method
        Returns:
            x, y: the projection coordinates
    '''

    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == 'lstsq':
        # solve the least squre problem: Ax = d
        A = np.vstack([dx.numpy(), dy.numpy()]).T
        [x, y] = np.linalg.lstsq(A, d.numpy())[0]

    return x, y
