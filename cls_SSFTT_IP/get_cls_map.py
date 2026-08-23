import numpy as np
import matplotlib.pyplot as plt

def get_classification_map(y_pred, y):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1

    return  cls_labels

import matplotlib.colors as mcolors

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    # Pre-define some distinct colors (including the original 16 to preserve look if possible)
    # But since we want it dynamic, let's use a large matplotlib colormap
    cmap = plt.get_cmap('tab20')
    max_val = int(np.max(x_list))
    
    # Alternatively, build a palette
    palette = {0: np.array([0, 0, 0])}
    
    # Original Indian pines colors for 1-16
    orig_colors = [
        [147, 67, 46], [0, 0, 255], [255, 100, 0], [0, 255, 123],
        [164, 75, 155], [101, 174, 255], [118, 254, 172], [60, 91, 112],
        [255, 255, 0], [255, 255, 125], [255, 0, 255], [100, 0, 255],
        [0, 172, 254], [0, 255, 0], [171, 175, 80], [101, 193, 60]
    ]
    
    for i in range(1, max_val + 1):
        if i <= len(orig_colors):
            palette[i] = np.array(orig_colors[i-1]) / 255.
        else:
            # Fallback to matplotlib tab20 for > 16 classes
            c = cmap((i - len(orig_colors) - 1) % 20)
            palette[i] = np.array(c[:3])

    for index, item in enumerate(x_list):
        y[index] = palette.get(int(item), np.array([0, 0, 0]))

    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def get_cls_map(net, device, all_data_loader, y, dataset_name='IP'):

    y_pred, y_new = test(device, net, all_data_loader)
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'classification_maps')
    os.makedirs(out_dir, exist_ok=True)
    classification_map(y_re, y, 300,
                       os.path.join(out_dir, f'{dataset_name}_predictions.eps'))
    classification_map(y_re, y, 300,
                       os.path.join(out_dir, f'{dataset_name}_predictions.png'))
    classification_map(gt_re, y, 300,
                       os.path.join(out_dir, f'{dataset_name}_gt.png'))
    print('------Get classification maps successful-------')