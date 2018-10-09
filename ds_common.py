# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:10:01 2018

@author: SanerZ
"""
import pdb
import numpy as np
import matplotlib.pyplot as plt

import PIL
import os.path as osp
from pathlib2 import Path
import os
from xml.dom.minidom import Document

from .bbs_utils import overlay_bounding_boxes, boxFormatTransform

class imdb(object):
    """Image database."""
    
    def __init__(self, name, data_path, image_set):
        self._name = name
        self._data_path = data_path
        self._image_index = []
        self._image_set = image_set
        self._gt_roidb = None
        
        self._image_ext = '.jpg'
        
        self.filter_params = {
                'filterGt':     self._filterGtFun,
                'hr':           [0, np.inf],
                'labels':       None,     # example
                'use_diff':     False,
                'display':      True,
                }
        
        
    @property
    def name(self):
        return self._name
    
    @property
    def image_index(self):
        return self._image_index
    
    @property
    def num_images(self):
        return len(self._image_index)
    
    @property
    def widths(self):
        try:
            return self._widths
        except:
            self._widths = self._get_image_size(0)
        return self._widths
    
    @property
    def heights(self):
        try:
            return self._heights
        except:
            self._heights = self._get_image_size(1)
        return self._heights
    
    @property
    def num_objects(self):
        try:
            num_objs = np.sum([len(self.gt_box_filter[i]) for i in range(self.num_images)])
        except:
            num_objs = np.sum([len(self.gt_roidb[i]['boxes']) for i in range(self.num_images)])
        
        return num_objs
    
    @property
    def gt_roidb(self):
        if self._gt_roidb is None:
            self._gt_roidb = self._get_gt_roidb()
        return self._gt_roidb
        
    
    def _get_gt_roidb(self):
        raise NotImplementedError
        
    def _load_image_set_index(self):
        image_set_file = self._image_set
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
       
        with open(image_set_file) as f:
            image_index = [osp.splitext(x.strip())[0] for x in f.readlines()]
#            image_index = [str(Path(x.strip()).with_suffix('')) for x in f.readlines()]
                
        return image_index
    
    def _get_image_size(self, index):
        return [PIL.Image.open(self.image_path_at(i)).size[index] \
                for i in range(self.num_images)]
    
    
    # TODO: 
    """ Helper Functioins """
    
    def image_path_at(self, i):
        return osp.join(self._data_path, self.image_index[i]+self._image_ext)
        
    def check_and_modify_bbox(self, box, i):
        x1 = np.minimum(np.maximum(0, box[:,[0]]), self.widths[i]-2)
        y1 = np.minimum(np.maximum(0, box[:,[1]]), self.heights[i]-2)

        if self.data_format == 'LTWH':
            x2 = np.minimum(self.widths[i]-1, x1+np.maximum(1,box[:,[2]]))
            y2 = np.minimum(self.heights[i]-1, y1+np.maximum(1,box[:,[3]]))
            w = x2 - x1
            h = y2 - y1
            return np.hstack((x1, y1, w, h))
        else:
            x2 = np.minimum(self.widths[i]-1, np.maximum(x1+1,box[:,[2]]))
            y2 = np.minimum(self.heights[i]-1, np.maximum(y1+1,box[:,[3]])) 
            return np.hstack((x1, y1, x2, y2))
        
        
    def bbox_display(self, idx, lw=2, color=None):
        """display the idxTh picture with bounding box"""
        
        imgpath = self.image_path_at(idx)
        
        im = plt.imread(imgpath)
        boxes = self.gt_roidb[idx]['boxes'] 
        overlay_bounding_boxes(im, boxes, lw, color, wh = self.data_format == 'LTWH')
        
        plt.figure(figsize=[10,8])
        plt.imshow(im)
        plt.show()    


    # TODO: 
    """  Analysis """   
    def _filterGtFun(self, bb, bbv):
        hr = self.filter_params['hr']
        h = bb[3]                   
        p = h>=hr[0] and h<hr[1]                    # height range
        
        return p
    
    def _img_gt_filter(self, i):
        """
        filter the ground truth each image 
        
        % INPUT
        %  gt_roidb[i]  - a dict {'boxes': boxes, 'cls': cls, 'diff': diff}
        %                    x, y, w, h   [n x 4]        [n]           [n]
        %  labels       - a list of object labels to evaluate
        %  filterGt     - ignore = not filterGt(lbl, bb, bbv)
        %
        % OUTPUTS
        %  gt           - [n x 5] array containg ground truth for image        
        """
        use_diff = self.filter_params['use_diff']
        filterGt = self.filter_params['filterGt']
        labels = self.filter_params['labels']
        
        height = float(self.heights[i])
        width = float(self.widths[i])
        
        bb = self.gt_roidb[i]['boxes'].copy()
        nObj = len(bb)
        if nObj == 0:
            return np.zeros((nObj, 5))
        
        if self.data_format == 'LTRB':
            bb[:,2:] -= bb[:,:2] - 1
            
        cls = self.gt_roidb[i]['cls']
        diff = np.array(self.gt_roidb[i]['diff']) if not use_diff \
            else np.zeros(nObj)
        
        bbv = self.gt_roidb[i]['bbv'] if 'bbv' in self.gt_roidb[i].keys() else np.zeros((nObj, 4))
        
        keep = np.where([labels is None or cls[idx] in labels for idx in range(nObj)])[0]
        if filterGt is False:
            ign = diff[keep]
        else:
            imgsize = np.tile((width, height), 2)
            ign = np.array([not filterGt(bb[idx]/imgsize, bbv[idx]/imgsize) for idx in keep]).astype(bool)

        self.gt_box_filter.append(bb[keep][~ign])
        
        return np.column_stack((bb[keep], ign))
    
    
    def gt_filter(self, **filter_params):
        """ 
         update filter params and filter gt_roidb
         OUT_GT:  list of gts format as LTWH
        """
        self.filter_params.update(**filter_params)
        self.gt_box_filter = list()
        
        return [self._img_gt_filter(i) for i in range(self.num_images)]
    
    def _draw_dist(self, data):
        """Draw distribution figure of data array"""
        
        xlim, ylim = np.unique(data, return_counts=True)
            
        fig = plt.figure()            
        plt.bar(xlim, ylim, linewidth=0.2)
        
        if self.filter_params['display']:
            plt.show()
        
        return fig, [xlim, ylim]

   
    def density_dist(self, **filter_params):
        """Distribution of groundtruth number per image """
        self.gt_filter(**filter_params)
        
        box_nums = [len(self.gt_box_filter[i]) for i in range(self.num_images)]

        den, hist = self._draw_dist(box_nums)

        
    def scale_dist(self, pix=False, **filter_params):
        """Distribution of groundtruth scale"""
        self.gt_filter(**filter_params)
              
        gt_scale = list()
        for i in range(self.num_images):
            box = self.gt_box_filter[i]
            if box.shape[0] == 0:
                continue

            gt_scale.extend(box[:,3])    
            if not pix:
                gt_scale[-1] = gt_scale[-1]*1000.0/self.heights[i]
                
        scale, hist = self._draw_dist(gt_scale)



    # TODO: 
    """ Format Output """
    
    def write_image_set_list(self, fpath, max_num=np.inf):
        fpath = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        
        max_num = min(max_num, self.num_images)
        image_index = self.image_index[:max_num]
        image_index.append('')

        lines = (self._image_ext + '\n').join(image_index)
        with open(str(fpath), 'w') as fid:
            fid.writelines(lines)

    

         
    #### create voc-like xml
    def _write_voc_xml(self, i, outdir):
        f_xml = osp.join(outdir, self.image_index[i] +'.xml')  #osp.splitext(self.image_index[i])[0]
        
        boxes = self.gt_roidb[i]['boxes'].copy()
        if len(boxes) == 0:
            return 
        
        if self.data_format == 'LTWH':
            boxes[:,2:] += boxes[:,:2] - 1
        
        cls = self.gt_roidb[i]['cls']
        diff = self.gt_roidb[i]['diff']

        Path(f_xml).parent.mkdir(parents=True,exist_ok=True)
        
        doc = Document()
        anno = doc.createElement('annotation')
        doc.appendChild(anno)
        
        fname = doc.createElement('filename')
        fname_text = doc.createTextNode(self.image_index[i]+self._image_ext)
        fname.appendChild(fname_text)
        anno.appendChild(fname)
        
        size = doc.createElement('size')
        anno.appendChild(size)
        ##需要修改的就是这部分，宽高
        width = doc.createElement('width')
        width.appendChild(doc.createTextNode(str(self.widths[i])))
        height = doc.createElement('height')
        height.appendChild(doc.createTextNode(str(self.heights[i])))
        depth = doc.createElement('depth')
        depth.appendChild(doc.createTextNode('3'))
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        
        segmented = doc.createElement('segmented')
        segmented.appendChild(doc.createTextNode('0'))
        anno.appendChild(segmented)
        ##需要添加目标
        for idx in range(boxes.shape[0]):
            objects = doc.createElement('object')
            anno.appendChild(objects)
            object_name = doc.createElement('name')
            object_name.appendChild(doc.createTextNode(cls[idx]))
            objects.appendChild(object_name)
            pose = doc.createElement('pose')
            pose.appendChild(doc.createTextNode('Unspecified'))
            objects.appendChild(pose)
            truncated = doc.createElement('truncated')
            truncated.appendChild(doc.createTextNode('0'))
            objects.appendChild(truncated)
            difficult = doc.createElement('difficult')
            difficult.appendChild(doc.createTextNode(str(int(diff[idx]))))
            objects.appendChild(difficult)
            bndbox = doc.createElement('bndbox')
            objects.appendChild(bndbox)
            xmin = doc.createElement('xmin')
            xmin.appendChild(doc.createTextNode((str(int(np.round(boxes[idx][0]))))))
            bndbox.appendChild(xmin)
            ymin = doc.createElement('ymin')
            ymin.appendChild(doc.createTextNode((str(int(np.round(boxes[idx][1]))))))
            bndbox.appendChild(ymin)
            xmax = doc.createElement('xmax')
            xmax.appendChild(doc.createTextNode((str(int(np.round(boxes[idx][2]))))))
            bndbox.appendChild(xmax)
            ymax = doc.createElement('ymax')
            ymax.appendChild(doc.createTextNode((str(int(np.round(boxes[idx][3]))))))
            bndbox.appendChild(ymax)
    
        f = open(f_xml,'w')
        f.write(doc.toprettyxml(indent = '\t'))
        f.close() 

    
    def write_xml(self, outdir):   
        for i in range(self.num_images):
            self._write_voc_xml(i, outdir)
            
            
    # fcn gt & lmk        
    def write_lst_with_gt(self, fname):
        fid = open(fname, 'w')
        for i in range(self.num_images):
            line = [self.image_index[i]+self._image_ext]
            gt = self.gt_roidb[i]['boxes'].copy()
            if self.data_format == 'LTRB':
                gt[:,2:] -= gt[:,:2] - 1
            classes = self.gt_roidb[i]['cls']
            cls = [self.labels.index(c) for c in classes]
            boxes = np.array(gt[:,:4], dtype = str)
            gt_line = np.column_stack((cls, boxes))
            line.extend(gt_line.reshape(-1))
            fid.write(' '.join(line)+'\n')
        fid.close() 
            
    
    def write_lst_with_lmk(self, fname, use_diff=True, anno_len=77):
        fid = open(fname, 'w')
        for i in range(self.num_images):
            line = [self.image_index[i]+self._image_ext]
            gt = self.gt_roidb[i]['boxes'].copy()
            
            no_diff = ~np.array(self.gt_roidb[i]['diff'], dtype = bool)
            gt = gt if use_diff else gt[no_diff]
            if self.data_format == 'LTWH':
                gt[:,2:] += gt[:,:2] - 1
            for g in gt:
                lbl=anno_len*2*['-1']
                idx_bg = anno_len-5
                
                lbl[idx_bg*2] = lbl[idx_bg*2+6] = str(g[0])
                lbl[idx_bg*2+1] = lbl[idx_bg*2+3] = str(g[1])
                lbl[idx_bg*2+2] = lbl[idx_bg*2+4] = str(g[2])
                lbl[idx_bg*2+5] = lbl[idx_bg*2+7] = str(g[3])
                lbl[idx_bg*2+8] = str(0.5*(g[0]+g[2]))
                lbl[idx_bg*2+9] = str(0.5*(g[1]+g[3]))
                line.extend(lbl)
            fid.write(' '.join(line)+'\n')
        fid.close()
        
        
    # fddb
    def write_fddb(self, fname):
        with open(fname, 'w') as f:
            for i in range(self.num_images):
                self._write_fddb_gt(i, f)
                    
    def _write_fddb_gt(self, i, f):
        imginfo = self.image_index[i] + self._image_ext
        gt = self.gt_roidb[i]['boxes'] 
        det = gt if self.data_format == 'LTWH' else boxFormatTransform(gt, wh=False)
        
        f.write('{:s}\n'.format(imginfo))
        f.write('{:d}\n'.format(det.shape[0]))
        
        classes = self.gt_roidb[i]['cls']
        try:
            cls = [self.labels.index[c] for c in classes]
        except:
            cls = np.ones(len(classes)).astype(int)

        for i in range(det.shape[0]):
            xmin = det[i][0]
            ymin = det[i][1]
            w = det[i][2]
            h = det[i][3]
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:d}\n'.
                    format(xmin, ymin, w, h, cls[i]))

       
       



        
        