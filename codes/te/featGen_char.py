#!/usr/bin/env python

"""
A feature extractor for chunking.
Copyright 2010,2011 Naoaki Okazaki.
"""

# Separator of field values.
separator = ' '

# Field names of the input data.
fields = 'w ws pos open y'

# Attribute templates.

'''
#第一套特征
templates = (
    (('w', -2), ),
    (('w', -1), ),
    (('w',  0), ),
    (('w',  1), ),
    (('w',  2), ),
    (('w', -1), ('w',  0)),
    (('w',  0), ('w',  1)),
    (('w', -1), ('w', 1)),
    (('w', -2), ('w', -1)),
    (('w', -2), ('w', 0)),
    (('w', -3), ('w', -1)),
    (('open', 0),),
    )
'''

#第二套特征
templates = (
    (('w', -2), ),
    (('w', -1), ),
    (('w',  0), ),
    (('w',  1), ),
    (('w',  2), ),
    (('w', -1), ('w', 0)),
    (('w', 0), ('w', 1)),
    (('w', -1), ('w', 1)),
    (('w', 1), ('w', 2)),
    (('w', -2), ('w', -1)),
    (('ws', 0), ),
    (('pos', 0), ),
    (('pos', 1), ),
    (('pos', 2), ),
    (('pos', -1), ),
    (('pos', -2), ),
    (('pos', -1), ('pos', 0)),
    (('pos', 0), ('pos', 1)),
    (('open', 0), ),
    )
import crfutils_char

def feature_extractor(X):
    # Apply attribute templates to obtain features (in fact, attributes)
    crfutils_char.apply_templates(X, templates)
    if X:
	# Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature

if __name__ == '__main__':
    crfutils_char.main(feature_extractor, fields=fields, sep=separator)
