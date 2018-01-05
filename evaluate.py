#!/usr/bin/env python

import argparse
import os
import os.path as osp
import utils
import fcnutils
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import tqdm
from scipy.misc import imsave
from train_fcn8s import SBDClassSeg
from train_fcn8s import FCN8s
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    root = osp.expanduser('~/data/datasets')
    val_loader = torch.utils.data.DataLoader(
        SBDClassSeg(
            root, split = 'new_val1', transform = True),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    n_class = len(val_loader.dataset.class_names)
    print(osp.basename(model_file))
    model = FCN8s(n_class=2)
    
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating with VOC2011ClassSeg seg11valid')
    visualizations = []
    label_trues, label_preds = [], []
    roc_prob, roc_label = [], []
    img_id = 1
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        score = model(data)
        imgs = data.data.cpu()
        
        roc_prob.extend(list(1 / (1 + np.exp(-score.data.cpu().numpy()[0][1].flatten()))))
        roc_label.extend(list(target.data.cpu().numpy()[0].flatten()))
        
        #print(target.data.cpu().numpy()[0].flatten().shape)
        
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            
            label_trues.append(lt)
            label_preds.append(lp)
            
            if img_id % 1 == 0:
                imsave('img/' + str(img_id).zfill(4) + 'orig' + '.png', img)
                imsave('img/' + str(img_id).zfill(4) + 'mask' + '.png', lp)
            #print(str(batch_idx) + ' ' + str(img_id))
            img_id = img_id + 1
            if len(visualizations) < 9:
                viz = fcnutils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                    label_names=val_loader.dataset.class_names)
                visualizations.append(viz)
    metrics = utils.label_accuracy_score(
        label_trues, label_preds, n_class=n_class)
    
    print('The length of labels is:')
    print(len(roc_label))
    print(roc_label[:10])
    print(roc_prob[:10])
    '''
    fpr, tpr, thres = roc_curve(roc_label, roc_prob, pos_label=1)
    precision, recall, thres = precision_recall_curve(roc_label, roc_prob, pos_label=1)
    print(len(fpr))
    print(len(tpr))
    f_fpr = open('precision.txt', 'wb')
    for x in range(0, len(precision), 1000):
        f_fpr.write(str(precision[x]) + ' ')
    f_fpr.close()
    f_tpr = open('recall.txt', 'wb')
    for x in range(0, len(recall), 1000):
        f_tpr.write(str(recall[x]) + ' ')
    f_tpr.close()
    
    print('The auc is ')
    print(auc(fpr, tpr))
    '''
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

    viz = fcnutils.get_tile_image(visualizations)
    skimage.io.imsave('viz_evaluate.png', viz)


if __name__ == '__main__':
    main()
