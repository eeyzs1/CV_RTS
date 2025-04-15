import torch
import time, copy


def test_model(model, criterion, testloader, device, dataset_sizes):
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    metrics = {
        'loss': [],
        'acc': [],
    "bast_acc": 0        
                 }
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
    
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)  

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / dataset_sizes
    test_acc = running_corrects.double() / dataset_sizes

    metrics['loss'].append(test_loss)
    metrics['acc'].append(test_acc)
    # deep copy the model
    if test_acc > best_acc:
        best_acc = test_acc
        # best_model_wts = copy.deepcopy(model.state_dict())
    # print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    metrics['bast_acc'] = best_acc
    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, metrics
