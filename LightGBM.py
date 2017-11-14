#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
import pandas as pd
import numpy as np
import os
import datetime
import math
import random
import xgboost as xgb
import lightgbm as lgb
import time
import random
from sklearn.metrics import mean_absolute_error

random.seed(888)




to_drop = ['order_unix_time', 'arriveshop_unix_time', 'fetch_unix_time', 'finish_unix_time', 'order_id',  'date', 'log_unix_time','time', 'type', 'per_realdist_food_time', 'per_dist_food_time' ,'mod_delivery_time','mod_per_dist_time',"avg_per_dist_time", "avg_delivery_time", "ori_order_unix_time","per_food_time", 'minute',"minute_gap","delivery_duration"]  
to_drop.extend(['poi_lat', 'poi_lng', 'customer_latitude', 'customer_longitude','area_id',"poi_area_geo"])

to_drop.extend([u'avg_area_id_finish_unix_time_fetch_unix_time_perdistance', u'gap_poi_userarea_orders_1800s_waiting_orders', u'user_area_orders_3000s', u'gap_usearea_poi_orders3600s',"avg_poi_per_food_time_prehour_x", "median_poi_per_food_time_prehour_x",'avg_poi_orderInterval_x', u'median_poi_orderInterval_x'])

remove_col =[]
for i in waimai.columns:
    if  "max_" in i or 'std_' in i or 'mid_' in i :#or 'prehour_0060' in i: 
        remove_col.extend([i])

to_drop.extend(remove_col)
features = list(np.setdiff1d(waimai.columns.tolist(), to_drop))

category_f =['poi_id','user_area_geo'] 

clean_data = waimai[(waimai.type =="train") & (waimai.hour >= 10) & (waimai.hour.values <= 20)]
datemask = clean_data.date <= 20170723
selectmask = ((clean_data.delivery_duration <= 12.25) & (clean_data.delivery_duration >= 8.89)) & clean_data.hour.isin([10,11,12,13,17,18,19])
x_train = clean_data[datemask & selectmask]
x_test = clean_data[~datemask & ((clean_data.hour == 11) | (clean_data.hour.values == 17)) ]
x_valid = clean_data[~datemask & selectmask]
x_sub = waimai[waimai.type =="test"]



dtrain = lgb.Dataset(x_train[features], label=x_train["delivery_duration"],  categorical_feature=category_f, free_raw_data=False)
dtest = lgb.Dataset(x_test[features], label=x_test["delivery_duration"], categorical_feature=category_f, free_raw_data=False)
dvalid = lgb.Dataset(x_valid[features], label=x_valid["delivery_duration"], categorical_feature=category_f, free_raw_data=False)
dsub = lgb.Dataset(x_sub[features], label=x_sub["delivery_duration"], categorical_feature=category_f, free_raw_data=False)


param = {'num_leaves':40,'num_boost_round':700, 'objective':'huber', "feature_fraction":0.66, "bagging_freq" : 1 , "bagging_fraction": 0.6, 'huber_delta':0.97,  'metric':'mae',"learning_rate" : 0.05, "boosting":"gbdt", "max_cat_group": 4} 

now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
bst = lgb.train(param, dtrain, valid_sets=[dtrain],  verbose_eval=100) #
print('train mae: %g' % mean_absolute_error(np.power(2,x_train["delivery_duration"]), np.power(2,bst.predict(dtrain.data))))
print('valid mae: %g' % mean_absolute_error(np.power(2,x_valid["delivery_duration"]), np.power(2,bst.predict(dvalid.data))))
test_mae1 = mean_absolute_error(np.power(2,x_test["delivery_duration"]), np.power(2,bst.predict(dtest.data)))
print('test mae: %g' % test_mae1)

bst1 = lgb.train(param, dvalid,  init_model=bst)
print('train mae: %g' % mean_absolute_error(np.power(2,x_train["delivery_duration"]), np.power(2,bst1.predict(dtrain.data))))
print('valid mae: %g' % mean_absolute_error(np.power(2,x_valid["delivery_duration"]), np.power(2,bst1.predict(dvalid.data))))
test_mae = mean_absolute_error(np.power(2,x_test["delivery_duration"]), np.power(2,bst1.predict(dtest.data)))
print('test mae: %g' % test_mae)



 
#保存进入模型的特征
feature_file = open("score_feature/"+ now_time + "_score_" +str(test_mae) + 'features.txt', 'w')  
for fea in features:  
    feature_file.write(fea)  
    feature_file.write('\n')  
feature_file.close()

print('feature importance...')
imp = bst.feature_importance(importance_type='gain', iteration=-1)
#imp = bst.feature_importance(importance_type='split', iteration=-1)
feat_importance = pd.Series(imp,bst.feature_name()).to_dict()
feat_importance = sorted(feat_importance.iteritems() ,key = lambda asd:asd[1],reverse=True)
imp = pd.DataFrame(feat_importance)
bst1.save_model('score_feature/lgb_' + now_time + '.model')
imp.to_csv('score_feature/'+ now_time + 'fea_imp.csv', index=False)


##get submission
pred = np.power(2, bst.predict(dsub.data)) -1
id_test = x_sub['order_id']
sub = pd.DataFrame({'order_id': id_test, 'delivery_duration': pred})
print('saving submission...')
sub.to_csv('sub_xgb_lightgbm.csv', index=False)
sub.describe()



####下面都不是


def huber_approx_obj(preds, dtrain):
    np.mean(np.abs((np.power(2,x_train["delivery_duration"]) -1) - (np.power(2,bst1.predict(dtrain.data)) -1) ))
    d = dtrain.get_labels() - preds  #remove .get_labels() for sklearn
    h = 1  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess



