#!/usr/bin/env python
# coding: utf-8

DIR = '.'

# Untar the K400, K600 and K700_200 files in the above directory.
# This results in subfolders kinetics400, kinetics600 and kinetics700_200.
import csv
import json
import os
import pandas as pd
import numpy as np


# Read all train, test and validate CSV files.
# Note: test file for K700 does not have labels.
k600 = pd.read_csv(f'{DIR}/kinetics600/test.csv')
k600 = k600.append(pd.read_csv(f'{DIR}/kinetics600/train.csv'))
k600 = k600.append( pd.read_csv(f'{DIR}/kinetics600/validatE.csv'))

k400 = pd.read_csv(f'{DIR}/kinetics400/test.csv')
k400 = k400.append(pd.read_csv(f'{DIR}/kinetics400/train.csv'))
k400 = k400.append(pd.read_csv(f'{DIR}/kinetics400/validatE.csv'))

k700 = pd.read_csv(f'{DIR}/kinetics700_2020/train.csv')
k700 = k700.append(pd.read_csv(f'{DIR}/kinetics700_2020/validatE.csv'))


# Create a set of unique labels per each kinetics data set.
k700 = k700.set_index('youtube_id')
k600 = k600.set_index('youtube_id')
k400 = k400.set_index('youtube_id')

k700_label_count = k700.groupby(['label']).count()
k600_label_count = k600.groupby(['label']).count()
k400_label_count = k400.groupby(['label']).count()
k400_label_count.drop(['time_start','time_end','split'], inplace=True, axis=1)
k600_label_count.drop(['time_start','time_end','split'], inplace=True, axis=1)
k700_label_count.drop(['time_start','time_end','split'], inplace=True, axis=1)

k700_labels = pd.DataFrame([{'K700':row[0]} for row in k700_label_count.iterrows()])
k700_labels = k700_labels.set_index('K700')
k600_labels = pd.DataFrame([{'K600':row[0]} for row in k600_label_count.iterrows()])
k600_labels = k600_labels.set_index('K600')
k400_labels = pd.DataFrame([{'K400':row[0]} for row in k400_label_count.iterrows()])
k400_labels = k400_labels.set_index('K400')

# find exact label matches between each data set

#k400_label_count.to_csv('/Users/robertsone/Downloads/k400keys.csv')
#k600_label_count.to_csv('/Users/robertsone/Downloads/k600keys.csv')
#k700_label_count.to_csv('/Users/robertsone/Downloads/k700keys.csv')

direct_mapping700to600 = k700_label_count.join(k600_label_count,how='inner', rsuffix='_600')
direct_mapping600to400 = k600_label_count.join(k400_label_count,how='inner', rsuffix='_400')
direct_mapping700to400 = k700_label_count.join(k400_label_count,how='inner', rsuffix='_400')

direct_mapping700to600 =direct_mapping700to600.reset_index()
direct_mapping700to600.rename(columns={'label': 'K700'},inplace=True)
direct_mapping700to600['K600'] = direct_mapping700to600['K700']
direct_mapping700to600 = direct_mapping700to600.set_index(['K700'])

direct_mapping600to400 =direct_mapping600to400.reset_index()
direct_mapping600to400.rename(columns={'label': 'K600'},inplace=True)
direct_mapping600to400['K400'] = direct_mapping700to600['K600']
direct_mapping600to400 = direct_mapping600to400.set_index(['K600'])

direct_mapping700to400 =direct_mapping700to400.reset_index()
direct_mapping700to400.rename(columns={'label': 'K700'},inplace=True)
direct_mapping700to400['K400'] = direct_mapping700to400['K700']
direct_mapping700to400 = direct_mapping700to400.set_index(['K700'])


# Count the number of common youtube IDs per each label.
mapping700to600 = k700.join(k600,rsuffix='_600',how='inner').groupby(['label','label_600']).count()
mapping700to600 = mapping700to600.reset_index()
mapping700to600 = mapping700to600[['label','label_600','time_start']]
mapping700to600 = mapping700to600.rename(columns={'label': 'K700', 'label_600': 'K600',
                                                  'time_start':'count'})
mapping700to600 = mapping700to600.set_index(['K700'])

mapping700to400 = k700.join(k400,rsuffix='_400',how='inner').groupby(['label','label_400']).count()
mapping700to400 = mapping700to400.reset_index()
mapping700to400 = mapping700to400[['label','label_400','time_start']]
mapping700to400 = mapping700to400.rename(columns={'label': 'K700', 'label_400': 'K400','time_start':'count'})
mapping700to400 = mapping700to400.set_index(['K700'])

mapping600to400 = k600.join(k400,rsuffix='_400',how='inner').groupby(['label','label_400']).count()
mapping600to400 = mapping600to400.reset_index()
mapping600to400 = mapping600to400[['label','label_400','time_start']]
mapping600to400 = mapping600to400.rename(columns={'label': 'K600', 'label_400': 'K400','time_start':'count'})
mapping600to400 = mapping600to400.set_index(['K600'])


# Add counts for direct label mappings if missing.
#mapping700to400.join(direct_mapping700to400, how='outer', rsuffix='_400')
#mapping600to400.join(direct_mapping600to400, how='outer', rsuffix='_400')
#mapping700to600.join(direct_mapping700to600, how='outer', rsuffix='_600')
mapping600to400['direct'] = mapping600to400.index == mapping600to400['K400']
mapping700to600['direct'] = mapping700to600.index == mapping700to600['K600']
mapping700to400['direct'] = mapping700to400.index == mapping700to400['K400']

mapping700to600.to_csv(f'{DIR}/k700to600.csv')
mapping600to400.to_csv(f'{DIR}/k700to400.csv')
mapping700to400.to_csv(f'{DIR}/k700to400.csv')


# Find best matches based on count.  Assuming a label matching is a a best
# mapping.
best_700to600 = mapping700to600[['K600','count']].groupby(['K700']).max()
best_700to400 = mapping700to400[['K400','count']].groupby(['K700']).max()
best_600to400 = mapping600to400[['K400','count']].groupby(['K600']).max()


# Build DataFrame for K700 to  K600 mappings based on counts including unmapped
# K700 and K600 classes
final_700to600 = k700_labels.join(mapping700to600, how='right')
tempdf = pd.DataFrame([{'K700': 'NA', 'K600':r[0], 'count':0, 'direct': False} for r in k600_labels.iterrows()
                      if len(mapping700to600[mapping700to600['K600'] == r[0]]) == 0])
tempdf  = tempdf.set_index('K700')
final_700to600 = final_700to600.append(tempdf)
final_700to600.to_csv(f'{DIR}/k700to600all.csv')
tempdf

final_700to400 = k700_labels.join(mapping700to400, how='right')
tempdf = pd.DataFrame([{'K700': 'NA', 'K400':r[0], 'count':0, 'direct': False} for r in k400_labels.iterrows()
                      if len(final_700to400[final_700to400['K400'] == r[0]]) == 0])
tempdf  = tempdf.set_index('K700')
final_700to400 = final_700to400.append(tempdf)
final_700to400.to_csv(f'{DIR}//k700to400all.csv')
tempdf

# Build DataFrame for the K700 to  K640 mappings based on counts including
# unmapped K700 and K400 classes
k400to700suggestions = [
    ('cleaning floor', 'vacuuming floor' ,'400 superclass'),
    ('cleaning floor', 'mopping floor', '400 superclass' ),
    ('drinking','drinking shots', '400 superclass'),
    ('filling eyebrows','dyeing eyebrows', '400 superclass'),
    ('playing cards', 'playing poker', '400 superclass'),
    ('strumming guitar', 'playing guitar', '400 superclass'),
    ('spraying','spray painting','400 superclass')]


# Record in a DataFrame best mappings for K700 to K600 and to k400

best_700to600.to_csv('/Users/robertsone/Downloads/k700to600best.csv')
best_700to400.to_csv('/Users/robertsone/Downloads/k700to400best.csv')


# Build master file with all best mappings and suggestions. Assumes one best mapping for each 700.
# It is possible for 600 and 400 to have more than one mapping.

k700dfall = best_700to600
k700dfall = k700dfall.join(best_700to400,   how='left', lsuffix='_600', rsuffix='_400')
k400supers = pd.DataFrame([{'K700':v[1],'K400 Superclass':v[0]} for v in k400to700suggestions])
k400supers = k400supers.set_index('K700')
k700dfall  =k700dfall.join(k400supers,   how='left')
k700dfall.fillna('NA',inplace=True)
k700dfall.to_csv('/Users/robertsone/Downloads/k700recommended_mappings.csv')
k700dfall

full_csv = pd.read_csv(f'{DIR}/kinetics_400_600_700_2020.csv')
full_csv

errors = set()
DEFAULT = ['NA',np.nan, False]
def _get_final(row):
    candidates = 0
    try:
        if type(row[2]) == str or not np.isnan(row[2]):
            return [row[2],np.nan,True]
        if (type(row[1])) == str or not np.isnan(row[1]):
            candidates = mapping700to600[mapping700to600['K600'] == row[1]]
            x = candidates.iloc[[candidates['count'].argmax()]]
           # print(x)
            return [x.index[[0]][0],
                    np.asarray(x)[0][1],
                    np.asarray(x)[0][2]]
            return candidates.iloc[[candidates['count'].argmax()]].index[[0]]
        elif  (type(row[0])) == str or not np.isnan(row[0]):
            candidates = mapping700to400[mapping700to400['K400'] == row[0]]
            x = candidates.iloc[[candidates['count'].argmax()]]
            return [x.index[[0]][0],
                    np.asarray(x)[0][1],
                    np.asarray(x)[0][2]]
    except ValueError:
        return DEFAULT
    except Exception as ex:
        errors.add((row[0],row[1],row[2]))
        raise(ex)
    return DEFAULT
new_columms =  full_csv.apply(_get_final, axis=1)


for i, column_name in  enumerate(['derived_label_kinetics700', 'derived_count', 'derived_is_direct']):
    full_csv[column_name] = [x[i] for x in new_columms]
full_csv.to_csv(f'{DIR}/kinetics_400_600_700_2020_with_default.csv')


full_csv.groupby(['split_kinetics700_2020', 'derived_is_direct' ]).count()
