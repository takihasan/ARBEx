#!/usr/bin/env python
# -*- coding: utf-8 -*-

class BaseConfig():
    DIR_IMG = '../../_DATA/crops_aligned/crops_aligned'
    DIR_ANN_TRAIN = '../../_DATA/annotations/EXPR_Classification_Challenge/Train_Set'
    DIR_ANN_DEV = '../../_DATA/annotations/EXPR_Classification_Challenge/Validation_Set'
    # index to class mapping
    index2class = {
            0: 'neutral',
            1: 'anger',
            2: 'disgust',
            3: 'fear',
            4: 'happiness',
            5: 'sadness',
            6: 'surprise',
            7: 'other',
            }

    # class to index mapping
    class2index = {
             'neutral': 0,
             'anger': 1,
             'disgust': 2,
             'fear': 3,
             'happiness': 4,
             'sadness': 5,
             'surprise': 6,
             'other': 7,
            }
