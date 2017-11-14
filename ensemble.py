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
from sklearn.linear_model import Lasso
import time
import random
import bisect


def get_threshold(tree_json):
    threshold = []
    if "right_child" in tree_json.keys():
        threshold.extend(get_threshold(tree_json['right_child']))
    if "left_child" in tree_json.keys():
        threshold.extend(get_threshold(tree_json['left_child']))
    if "threshold" in tree_json.keys():
        threshold.extend([tree_json['threshold']])
    return threshold


def lgb_create_features(model, lgbdata, split_raw_data):
    pred_leaf = model.predict(lgbdata, pred_leaf = True)
    pd_pred_leaf = pd.DataFrame(pred_leaf).reset_index(drop=True)
    category_lr =['poi_id','user_area_geo','weekday','hour']
    law_cate_feature = pd.get_dummies(lgbdata[category_lr],sparse=True,columns=category_f).reset_index(drop=True)
    pd_pred_feature = pd.get_dummies(pd_pred_leaf,sparse=True,columns=pd_pred_leaf.columns).reset_index(drop=True) 
    newdata = pd.concat([split_raw_data.reset_index(drop=True), pd_pred_feature, law_cate_feature], axis=1, ignore_index=True)
    newdata.columns = split_raw_data.columns.append(pd_pred_feature.columns).append(law_cate_feature.columns)
    return newdata.fillna(0)

def split_raw_data(reg_data, dvalid_data, dtest_data, dsub_data):
    tmp = pd.DataFrame()
    reg_data["type"] = "train"
    dvalid_data["type"] = "valid"
    dtest_data["type"] = "test"
    dsub_data["type"] = "sub"
    all_data = reg_data.append(dvalid_data, ignore_index=True).append(dtest_data, ignore_index=True).append(dsub_data, ignore_index=True)
    gbmreg = lgb.LGBMRegressor(num_leaves = 8, max_depth=15, n_estimators= 1)
    for col in reg_data.columns:
        if col in category_f or col == "type":
            continue
        print col
        gbmreg.fit(pd.DataFrame(reg_data[col].fillna(0)),x_train["delivery_duration"])
        split = sorted(get_threshold(gbmreg.booster_.dump_model()["tree_info"][0]["tree_structure"]))  
        categories = all_data[col].fillna(0).apply(lambda x: bisect.bisect_left(split, x))
        tmp = pd.concat([tmp,categories],axis=1)
    tmp = pd.get_dummies(tmp,sparse=True,columns=tmp.columns).reset_index(drop=True)
    reg_data.drop("type",axis=1,inplace=True)
    dvalid_data.drop("type",axis=1,inplace=True)
    dtest_data.drop("type",axis=1,inplace=True) 
    dsub_data.drop("type",axis=1,inplace=True) 
    return tmp, tmp[all_data["type"] == "train"], tmp[all_data["type"] == "valid"], tmp[all_data["type"] == "test"], tmp[all_data["type"] == "sub"] 

to_drop = ['order_unix_time', 'arriveshop_unix_time', 'fetch_unix_time', 'finish_unix_time', 'order_id',  'date', 'log_unix_time','time', 'type', 'per_realdist_food_time', 'per_dist_food_time' ,'mod_delivery_time','mod_per_dist_time',"avg_per_dist_time", "avg_delivery_time", "ori_order_unix_time","per_food_time", 'minute',"minute_gap","delivery_duration"]  
to_drop.extend(['poi_lat', 'poi_lng', 'customer_latitude', 'customer_longitude','area_id',"poi_area_geo"])

to_drop.extend([u'avg_area_id_finish_unix_time_fetch_unix_time_perdistance', u'gap_poi_userarea_orders_1800s_waiting_orders', u'user_area_orders_3000s', u'gap_usearea_poi_orders3600s'])

remove_col =[]
for i in waimai.columns:
    if  "max_" in i or 'std_' in i or 'mid_' in i:
        remove_col.extend([i])

to_drop.extend(remove_col)
features = list(np.setdiff1d(waimai.columns.tolist(), to_drop))

category_f =['poi_id','user_area_geo'] 

clean_data = waimai[(waimai.type =="train") & (waimai.hour >= 10) & (waimai.hour.values <= 20)]
datemask = clean_data.date <= 20170729
selectmask = ((clean_data.delivery_duration <= 12.25) & (clean_data.delivery_duration >= 8.89)) & clean_data.hour.isin([10,11,12,13,17,18,19])
x_train = clean_data[datemask & selectmask]
x_test = clean_data[~datemask & ((clean_data.hour == 11) | (clean_data.hour.values == 17)) ]
x_valid = clean_data[~datemask & selectmask]
x_sub = waimai[waimai.type =="test"]


dtrain = lgb.Dataset(x_train[features], label=x_train["delivery_duration"],  categorical_feature=category_f, free_raw_data=False)
dtest = lgb.Dataset(x_test[features], label=x_test["delivery_duration"], categorical_feature=category_f, free_raw_data=False)
dvalid = lgb.Dataset(x_valid[features], label=x_valid["delivery_duration"], categorical_feature=category_f, free_raw_data=False)
dsub = lgb.Dataset(x_sub[features], label=x_sub["delivery_duration"], categorical_feature=category_f, free_raw_data=False)


param = {'num_leaves':12,'num_boost_round':400, 'objective':'huber', 'huber_delta':0.97, 'metric':'mae',"learning_rate" : 0.1, "boosting":"gbdt", "max_cat_group":4}

bst = lgb.train(param, dtrain, valid_sets=[dtrain],  verbose_eval=100) #
print('train mae: %g' % np.mean(np.abs((np.power(2,x_train["delivery_duration"]) -1) - (np.power(2,bst.predict(dtrain.data)) -1) )))
print('valid mae: %g' % np.mean(np.abs((np.power(2,x_valid["delivery_duration"])-1) - (np.power(2, bst.predict(dvalid.data)) -1) )))
test_mae1 = np.mean(np.abs((np.power(2,x_test["delivery_duration"])-1) - (np.power(2, bst.predict(dtest.data)) -1) ))
print('test mae: %g' % test_mae1)



all_tmp, split_dtrain, split_dvalid, split_dtest , split_dsub = split_raw_data(dtrain.data, dvalid.data, dtest.data, dsub.data)

new_train_data = lgb_create_features(bst,dtrain.data, split_dtrain)
new_valid_data = lgb_create_features(bst,dvalid.data, split_dvalid)
new_test_data = lgb_create_features(bst,dtest.data, split_dtest)
new_sub_data = lgb_create_features(bst,dsub.data, split_dsub)

union_feature = list(new_train_data.columns.intersection(new_valid_data.columns).intersection(new_test_data.columns).intersection(new_sub_data.columns))

lr = Lasso(alpha=0.0002, normalize=False, copy_X=True, max_iter=100000, warm_start =True, precompute=True)
model = lr.fit(new_train_data[union_feature], x_train["delivery_duration"])

print(lr.n_iter_)
print(sum(lr.coef_!=0))

re_lasso_train = lr.predict(new_train_data[union_feature])
re_lasso_valid = lr.predict(new_valid_data[union_feature])
re_lasso_test = lr.predict(new_test_data[union_feature])

print('train mae: %g' % np.mean(np.abs((np.power(2,x_train["delivery_duration"]) -1) - (np.power(2,re_lasso_train) -1) )))
print('valid mae: %g' % np.mean(np.abs((np.power(2,x_valid["delivery_duration"])-1) - (np.power(2, re_lasso_valid) -1) )))
test_mae1 = np.mean(np.abs((np.power(2,x_test["delivery_duration"])-1) - (np.power(2, re_lasso_test) -1) ))
print('test mae: %g' % test_mae1)


pred = np.power(2, lr.predict(new_sub_data[union_feature])) -1
id_test = x_sub['order_id']
sub = pd.DataFrame({'order_id': id_test, 'delivery_duration': pred})
print('saving submission...')
sub.to_csv('gbdt_lr.csv', index=False)
sub.describe()


def get_median_duration_of_distance_bin_hour_of(df_waimai):
    df_selected = df_waimai.loc[(df_waimai.type!="test") & df_waimai.hour.isin([10,11,12,13,17,18,19])]
    grouped = df_selected.groupby(['poi_area_geo', 'distance_bin', 'hour'])['delivery_duration']
    distance_bin_hour_delivery_duration_mean = grouped.median().reset_index()
    distance_bin_hour_delivery_duration_mean.columns = ['poi_area_geo', 'distance_bin', 'hour', 'distance_bin_hour_delivery_duration_mean']
    df_waimai = df_waimai.merge(distance_bin_hour_delivery_duration_mean, how='left', on=['poi_area_geo', 'distance_bin', 'hour'])
    # df_waimai = df_waimai.merge(distance_bin_delivery_duration_avg, how='left', on=['distance_bin'])
    return df_waimai


sub1 = pd.read_csv("/Users/zhangkai/Desktop/gbdt_lr.csv")
final = sub.merge(sub1,on="order_id",how="left")
final["delivery_duration"]= 0.8 * final["delivery_duration_x"] + 0.2* final["delivery_duration_y"]
final[["delivery_duration","order_id"]].to_csv('final.csv', index=False)


#####Random Forest
from sklearn.ensemble import RandomForestRegressor
#[m/3]
regr = RandomForestRegressor(n_estimators= 80, max_depth=5, criterion= "mae", random_state=0, n_jobs = 30, oob_score =False, max_features= "log2", warm_start = True)
rf_feature = list(set(features) - set(category_f))

indices_to_keep = ~x_train[rf_feature].isin([np.nan, np.inf, -np.inf]).any(1)

regr.fit(x_train[rf_feature].replace([np.inf,np.nan], 0), x_train["delivery_duration"])

rf_train = regr.predict(x_train[rf_feature].fillna(0))
rf_test = regr.predict(x_test[rf_feature].fillna(0))
rf_valid = regr.predict(x_valid[rf_feature].fillna(0))


print('train mae: %g' % np.mean(np.abs((np.power(2,x_train["delivery_duration"]) -1) - (np.power(2,rf_train) -1) )))
print('valid mae: %g' % np.mean(np.abs((np.power(2,x_valid["delivery_duration"])-1) - (np.power(2, rf_valid) -1) )))
test_mae1 = np.mean(np.abs((np.power(2,x_test["delivery_duration"])-1) - (np.power(2, rf_test) -1) ))
print('test mae: %g' % test_mae1)


regr.feature_importances_,rf_feature

feat_importance = pd.Series(regr.feature_importances_,rf_feature).to_dict()
feat_importance = sorted(feat_importance.iteritems() ,key = lambda asd:asd[1],reverse=True)
imp = pd.DataFrame(feat_importance)

#######xgboost
import xgboost as xgb
import random
random.seed(888)


rf_feature = list(set(features) - set(category_f))

xgbtrain = xgb.DMatrix(x_train[rf_feature].fillna(0), label=x_train["delivery_duration"])
xbgtest = xgb.DMatrix(x_test[rf_feature].fillna(0), label=x_test["delivery_duration"])
xgbvalid = xgb.DMatrix(x_valid[rf_feature].fillna(0), label=x_valid["delivery_duration"])
xgbsub = xgb.DMatrix(x_sub[rf_feature].fillna(0), label=x_sub["delivery_duration"])


print('training model...')
watchlist = [(xgbtrain, 'train'), (xgbvalid, 'eval')]
param = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'eta': 0.05,
        'num_round': 500,
        'max_depth': 5,
        'nthread': -1,
        'seed': 888,
        'silent': 1,
    }

xgbmodel = xgb.train(param, xgbtrain, param['num_round'], watchlist, verbose_eval=1)


print('train mae: %g' % np.mean(np.abs((np.power(2,x_train["delivery_duration"]) -1) - (np.power(2,xgbmodel.predict(xgbtrain)) -1) )))
print('valid mae: %g' % np.mean(np.abs((np.power(2,x_valid["delivery_duration"])-1) - (np.power(2, xgbmodel.predict(xgbvalid)) -1) )))
test_mae1 = np.mean(np.abs((np.power(2,x_test["delivery_duration"])-1) - (np.power(2, xgbmodel.predict(xbgtest)) -1) ))
print('test mae: %g' % test_mae1)

 

pred = np.power(2, xgbmodel.predict(xgbsub)) -1
id_test = x_sub['order_id']
xgbsub = pd.DataFrame({'order_id': id_test, 'delivery_duration_xgb': pred})
print('saving submission...')
xgbsub.to_csv('sub_xgb_final.csv', index=False)
xgbsub.describe()



xgbsub = pd.read_csv("/Users/zhangkai/Desktop/sub_xgb_final.csv")
final = final.merge(xgbsub,on="order_id",how="left")
final["delivery_duration"]= 0.4 * final["delivery_duration_x"] + 0.2 * final["delivery_duration_y"] + 0.4 * final["delivery_duration_xgb"] 
final[["delivery_duration","order_id"]].to_csv('final.csv', index=False)
