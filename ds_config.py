# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:49:31 2018

@author: ADMIN
"""

from collections import namedtuple

"""
List of dataset settings: {name}
"""
dtParams = namedtuple('dtParameters',['ds', 'resize', 'params'])

dtsDict = {
    'benchmark_1w_gt':    
        dtParams('fcn_gt', 0,
                 {'gt_lst': 'D:/zym/data/benchmark_1w_gt.lst'
                 }),
        
    'anfang_bg1w_gt':       
        dtParams('fcn_gt', 0,
                 {'gt_lst': 'D:/work/data/anfang_bg1w_gt.lst'
                 }),
        
    'wider':    
        dtParams('pascal_voc', 1,
                 {'data_path': 'D:/work/data/wider_part',
                  'image_set': 'img'
                 }),
    'wider_fcn':    
        dtParams('fcn_gt', 1,
                 {'data_path': 'D:/work/data/wider_part/JPEGImages',
                  'gt_lst': 'D:/work/data/wider_part/fcn_gt.lst',
                 }),
    'wider_fddb':    
        dtParams('fddb_gt', 1,
                 {'data_path': 'D:/work/data/wider_part/JPEGImages',
                  'gt_lst': 'D:/work/data/wider_part/fddb_gt.lst',
                 }),
    'wider_fddb2':    
        dtParams('fddb_gt', 1,
                 {'data_path': 'D:/work/data/wider_part/JPEGImages',
                  'gt_lst': 'D:/work/data/wider_part/fddb_gt.lst',
                  'image_set': 'D:/work/data/wider_part/fddb_img.txt',
                 }),
    
    'imdb':
        dtParams('fcn_lmk', 0,
                 {'data_path': 'D:/KYEimages/imdb',
                  'lmk_lst': 'D:/KYEimages/imdb_json/imdb.lst',
                 }),
    'imdb2':
        dtParams('fcn_lmk', 0,
                 {'data_path': 'D:/KYEimages/imdb',
                  'lmk_lst': 'D:/KYEimages/imdb_json/imdb.lst.multi_face',
                 }),
    'wiki':
        dtParams('fcn_lmk', 0,
                 {'data_path': 'D:/KYEimages/wiki',
                  'lmk_lst': 'D:/KYEimages/wiki_json/wiki.lst',
                 }),
    'wiki2':
        dtParams('fcn_lmk', 0,
                 {'data_path': 'D:/KYEimages/wiki',
                  'lmk_lst': 'D:/KYEimages/wiki_json/wiki.lst.multi_face',
                 }),
    'video':    
        dtParams('fddb_gt', 1,
                 {'data_path': 'D:/zym/data/POC/benchmark',
                  'gt_lst': 'D:/zym/data/POC_benchmark2.lst',
                #  'image_set': 'D:/zym/data/POC/benchmark/benchmark.lst',
                 }),
    'HK':    
        dtParams('fddb_gt', 1,
                 {'data_path': 'D:/zym/data/HKAirPort',
                  'gt_lst': 'D:/zym/data/HKAirPort/gt.lst',
                  'image_set': 'D:/zym/data/HKAirPort/img.lst',
                 }),
    'security_benchmark':    
        dtParams('fddb_gt', 1,
                 {'data_path': 'D:/zym/data/',
                  'gt_lst': 'D:/zym/data/security_benchmark.lst',
                #  'image_set': 'D:/zym/data/HKAirPort/img.lst',
                 }),

}