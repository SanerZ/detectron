# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:44:23 2018

@author: ADMIN
"""

import os
import os.path as osp
import numpy as np
from ConfigParser import ConfigParser

import xml.etree.ElementTree as ET
from easydict import EasyDict as edict

from .ds_common import imdb

# USAGE:
#   ds_config.read((Path(cfg.ds_cfgDir)/'datasets.conf').as_posix())
ds_config = ConfigParser()


def ds_factory(ds_name):
    ds_params = dict(ds_config.items(ds_name))
    ds_class = eval(ds_params.pop('handler'))
    return ds_class(ds_name, **ds_params) 

"""
                PASCAL_VOC DATASET
"""
class pascal_voc(imdb):
    def __init__(self, name, data_path, image_set):
        imdb.__init__(self, name, data_path, image_set)
        
        self._cfg = {
                'data_format': 'LTRB',
                'image_ext': '.jpg',
        }
        self.cfg.update(self._cfg)
#        self.cfg.update(**ds_params)
        self.cfg = edict(self.cfg)
        
        self._image_index = self._load_image_set_index()
        
      
    def image_path_at(self, i):
        return osp.join(self._data_path, 'JPEGImages', self.image_index[i]+self._image_ext)           

    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
       
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
                
        return image_index
                            
    def _load_xml_annotation(self, i):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', self.image_index[i] + '.xml')
        tree = ET.parse(filename)
        size = tree.find('size')
        self._widths[i] = int(size.find('width').text)
        self._heights[i] = int(size.find('height').text)
        
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.float)
        gt_classes = ['']*num_objs
        diff = np.zeros((num_objs))

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls = obj.find('name').text.lower().strip()
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            diff[ix] = int(obj.find('difficult').text)
#            diff[ix] = min(x2-x1, y2-y1)<30
        
        boxes = self.check_and_modify_bbox(boxes, i)

        return {'boxes' : boxes,
                'cls': gt_classes,
                'diff' : diff
                }

    
    def _get_gt_roidb(self):      
        self._widths = np.zeros((self.num_images), dtype=np.int16)
        self._heights = np.zeros((self.num_images), dtype=np.int16)
        return [self._load_xml_annotation(i) for i in range(self.num_images)]


"""
                FCN_WEISY_GT DATASET
"""    
    
class fcn_gt(imdb):
    def __init__(self, name, data_path, gt_lst, **ds_params):
        imdb.__init__(self, name, data_path)
        self._gt_lst = gt_lst
        
        self._cfg = {
                'data_format': 'LTWH',
                'labels': [
                            'background',
                            'face',
                           ],
                'box_len':  5,
        }
        self.cfg.update(self._cfg)
        self.cfg.update(**ds_params)
        self.cfg = edict(self.cfg)
        if type(self.cfg.labels) == str:
            self.cfg.labels = eval(self.cfg.labels)
            
        
        self._image_index, self._gt_roidb = self._load_gt_lst()
        
        
    
    def _load_gt_lst(self):
        assert osp.exists(self._gt_lst), \
                'Path does not exist: {}'.format(self._gt_lst)
        
        self.cfg.box_len = int(self.cfg.box_len)
        
        img_index = list()
        gt_roidb = list()
        with open(self._gt_lst) as f:
            for l in f.readlines():
#                im, *gt = l.strip().split()
                ls = l.strip().split()
                im, gt = ls[0], ls[1:]
                assert len(gt)%self.cfg.box_len == 0, \
                    'length of ground truth %d shoude be times of box_len(5) %d, \
                     format as cls/ign, x, y, w, h' % (len(gt), self.cfg.box_len)
                gt = np.array(gt, dtype=float).reshape((-1,self.cfg.box_len))
                if self.cfg.box_len == 4:
                    gt = np.column_stack((1, gt))
                gt = gt[..., :5]
                cls = [self.cfg.labels[abs(int(g[0]))] for g in gt]
                diff = (gt[:,0]<0).astype(int)
                # print(cls, diff)
                # im, _ = osp.splitext(im)
                img_index.append(im)
                gt_roidb.append({'boxes': gt[:,1:],
                                 'cls':   cls,
                                 'diff':  diff #np.zeros(len(cls))
                                })
        return img_index, gt_roidb
    
"""
                FCN_WEISY_WITH_LANDMARK DATASET
"""    
    
class fcn_lmk(imdb):
    def __init__(self, name, data_path, lmk_lst, **ds_params):
        imdb.__init__(self, name, data_path)
        self._lmk_lst = lmk_lst
        # self.anno_len = 77
        
        # self.data_format = 'LTRB'
        # self.labels = [
                # 'face',
                # ]
        self._cfg = edict({
                'data_format': 'LTRB',
                'labels': [
                            'face',
                           ],
                'anno_len': 77
        })
        self.cfg.update(self._cfg)
        self.cfg.update(**ds_params)
        self.cfg = edict(self.cfg)
        if type(self.cfg.labels) == str:
            self.cfg.labels = eval(self.cfg.labels)
        
        self._image_index, self._gt_roidb = self._load_lmk_lst()
        
    
    def _load_lmk_lst(self):
        assert osp.exists(self._lmk_lst), \
                'Path does not exist: {}'.format(self._lmk_lst)
        
        self.cfg.anno_len = int(self.cfg.anno_len)
        
        img_index = list()
        gt_roidb = list()
        with open(self._lmk_lst) as f:
            for l in f.readlines():
#                im, *gt = l.strip().split()
                ls = l.strip().split()
                im, gt_lmk = ls[0], ls[1:]
                assert len(gt_lmk)%(self.cfg.anno_len * 2) == 0, \
                    'length of ground truth shoude be times of 77 * 2, \
                     format as lmk1, lmk2, ..., lmk72, 5 key points'
                gt_lmk = np.array(gt_lmk, dtype=float).reshape((-1,self.cfg.anno_len * 2))
                
                idx_bg = self.cfg.anno_len-5
                gt = gt_lmk[:, idx_bg*2:]
                gt = gt[:, [0,1,2,5]]
                
                # im, _ = osp.splitext(im)
                img_index.append(im)
                gt_roidb.append({'boxes': gt,
                                 'cls':   self.cfg.labels * gt.shape[0],
                                 'diff':  np.zeros(gt.shape[0]).astype(int)
                                })
        return img_index, gt_roidb
        

    
"""
                FDDB_GT DATASET
"""    
    
class fddb_gt(imdb):
    def __init__(self, name, data_path, gt_lst, **ds_params):
        imdb.__init__(self, name, data_path)
        self._gt_lst = gt_lst
        
        self._cfg = edict({
                'data_format': 'LTWH',
                'labels': [
                            'background',
                            'face',
                           ],
        })
        self.cfg.update(self._cfg)
        self.cfg.update(**ds_params)
        self.cfg = edict(self.cfg)
        if type(self.cfg.labels) == str:
            self.cfg.labels = eval(self.cfg.labels)
                
        self._image_index, self._gt_roidb = self._load_fddb_gt()

                    
    def _load_fddb_gt(self):
        assert osp.exists(self._gt_lst), \
                'Path does not exist: {}'.format(self._gt_lst)

        angle_cls = eval(self.cfg.get('angle_cls', 'False'))       
        image_set = self.cfg.get('image_set','')
        if image_set:
            self._image_index = self._load_image_set_index()
        
        img_index = list()
        gt_roidb = list()
        with open(self._gt_lst) as f:
            dts = f.readlines()
        ls = len(dts)
        i = 0
        while i < ls:
            # im, _ = osp.splitext(dts[i].strip())
            im = dts[i].strip()
            ngt = int(dts[i+1].strip())
            i += ngt+2
            
            if ngt == 0 or image_set and im not in self._image_index:
                continue
                
            gt = [g.strip().split() for g in dts[i-ngt:i]]
            gt = np.array(gt, dtype=float)#.reshape((-1,5))
            if gt.shape[1] == 4:
                gt = np.column_stack((gt, np.ones(ngt)))
            cls = [self.cfg.labels[abs(int(g[4]))] for g in gt]
            diff = (gt[:,4]<0).astype(int)
            img_index.append(im)
            gt_label = {'boxes': gt[:,:4],
                         'cls':   cls,
                         'diff':  diff, #np.zeros(len(cls))
                       }
            if angle_cls：
                frontal = gt[:, 5]==1
                gt_label.update({'frontal':frontal})
            gt_roidb.append(gt_label)
        
        return img_index, gt_roidb
    

