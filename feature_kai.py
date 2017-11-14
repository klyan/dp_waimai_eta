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
import copy



#feature Engineering

#调整unix时间
def unixtime_to_datetime(df,c):
	df[c] = df[c].apply(lambda x: datetime.datetime.fromtimestamp(x))
	if c == "order_unix_time" or c == "log_unix_time":
		df['date'] = df[c].apply(lambda x: x.strftime('%Y%m%d'))
		df['hour'] = df[c].apply(lambda x: x.hour)
		df['minute'] = df[c].apply(lambda x: x.minute)
		df.date = df.date.astype(int)
	return df

def get_ordernum_window(df, gkey, window_size):
	grouped = df.groupby(gkey)['order_id', 'order_unix_time']
	order_num = grouped.rolling(window = window_size, on = 'order_unix_time', closed = 'left').count().reset_index().fillna(0)
	if gkey == "area_id":
		newcol = "area_orders_" + window_size
	elif gkey == "poi_id":
		newcol = "poi_orders_" + window_size
	elif gkey == "poi_area_geo":
		newcol = "poi_area_orders" + window_size
	order_num.rename(columns = {'order_id': newcol }, inplace =True)
	df = df.merge(order_num[["level_1",newcol]], how='left', left_index=True, right_on='level_1').drop(['level_1'], axis = 1)
	if gkey == "poi_id":
		df["gap_geted_waiting_orders_poi" + window_size]= df[newcol] - df["waiting_order_num"]
	elif gkey == "poi_area_geo":
		df["hot_poi_area_orders_" + window_size] = df[newcol] / (df["area_orders_" + window_size]+1)
	return df


def getNearestOrder(df, offset):
	df = df.groupby(["poi_id"]).apply(lambda x: x.sort_values(["order_unix_time"], ascending=True)).reset_index(drop=True)
	df['order_unix_time_shifted'] = df.groupby(['poi_id'])[['order_unix_time']].shift(offset)
	df['last_' + str(offset) + '_food_num'] = (df.groupby(['poi_id'])[['food_num']].shift(offset)).fillna(0)
	df['last_' + str(offset) + '_predict_foodtimes'] = (df.groupby(['poi_id'])[['predict_foodtimes']].shift(offset)).fillna(0)	
	df['last_' + str(offset) + '_foodvalue'] = (df.groupby(['poi_id'])[['food_total_value']].shift(offset)).fillna(0)
	df['last_' + str(offset) + '_waiting_ordernum'] = (df.groupby(['poi_id'])[['waiting_order_num']].shift(offset)).fillna(0)
	df['last_' + str(offset) + '_OrderInterval'] = ((df['order_unix_time'] - df['order_unix_time_shifted']).apply(lambda x: x.total_seconds())).fillna(3600)
	df["last_"+ str(offset) +"_foodtime_gap"] = df['last_' + str(offset) + '_predict_foodtimes'] - df['last_' + str(offset) + '_OrderInterval']
	return df[["order_id",'last_' + str(offset) + '_OrderInterval', 'last_' + str(offset) + '_food_num', 'last_' + str(offset) + '_foodvalue','last_' + str(offset) + '_waiting_ordernum', "last_"+ str(offset) +"_foodtime_gap"]]


def getUserAreaNearestOrder(df):
	df = df.groupby(["user_area_geo"]).apply(lambda x: x.sort_values(["order_unix_time"], ascending=True)).reset_index(drop=True)
	df['order_unix_time_shifted'] = df.groupby(['user_area_geo'])['order_unix_time'].shift(1)
	df["userarea_nearestOrderInterval"] = (df['order_unix_time'] - df['order_unix_time_shifted']).apply(lambda x: x.total_seconds())
	return df[["order_id","userarea_nearestOrderInterval"]].fillna(3600)


def get_poi_order_price_stat(df, colname):
	grouped = df.groupby('poi_id')[colname]
	poi_food_total_value_stat = grouped.agg(['mean','median']).reset_index().rename(
		index=str,
		columns= {
			'mean':'avg_' + colname,
			'median':'median_' + colname
		}
	)
	return poi_food_total_value_stat

def get_userarea_orders(df, window_size):
	user_area_order_num = df.groupby("user_area_geo")['order_id', 'order_unix_time'].rolling(window = window_size, on = 'order_unix_time', closed = 'left').count().reset_index().fillna(0)
	user_area_order_num.rename(columns = {'order_id': 'user_area_orders_' + window_size}, inplace =True)
	user_area_order_num.drop(["user_area_geo", 'order_unix_time'], axis = 1, inplace = True)
	df = df.merge(user_area_order_num, how='left', left_index=True, right_on='level_1').drop(['level_1'], axis = 1)
	df["gap_usearea_poi_orders" + window_size] = df["user_area_orders_" + window_size] - df["poi_orders_" + window_size]
	return df

def poi_userarea_orders(df, window_size):
	user_area_order_num = df.groupby(["poi_id","user_area_geo"])['order_id', 'order_unix_time'].rolling(window = window_size, on = 'order_unix_time', closed = 'left').count().reset_index().fillna(0)
	user_area_order_num.rename(columns = {'order_id': 'poi_userarea_orders_' + window_size}, inplace =True)
	user_area_order_num.drop(["poi_id","user_area_geo", 'order_unix_time'], axis = 1, inplace = True)
	df = df.merge(user_area_order_num, how='left', left_index=True, right_on='level_2').drop(['level_2'], axis = 1)
	df["gap_poi_userarea_orders_" + window_size + "_waiting_orders"] = df['poi_userarea_orders_' + window_size] - df["waiting_order_num"]
	return df 


def twoareas_orders(df, window_size):
	user_area_order_num = df.groupby(["poi_area_geo","user_area_geo"])['order_id', 'order_unix_time'].rolling(window = window_size, on = 'order_unix_time', closed = 'left').count().reset_index().fillna(0)
	user_area_order_num.rename(columns = {'order_id': 'twoareas_orders_' + window_size}, inplace =True)
	user_area_order_num.drop(["poi_area_geo","user_area_geo", 'order_unix_time'], axis = 1, inplace = True)
	df = df.merge(user_area_order_num, how='left', left_index=True, right_on='level_2').drop(['level_2'], axis = 1)
	return df 

def get_poi_foods_time(df_waimai):
	current_data = copy.deepcopy(df_waimai[df_waimai.type!="test"])
	colname = "per_food_time"
	current_data["fetch_unix_time"] = current_data["fetch_unix_time"].apply(lambda x: datetime.datetime.fromtimestamp(x))
	current_data[colname] = ((current_data["fetch_unix_time"]-current_data["order_unix_time"]) / current_data["food_num"]/ (current_data["waiting_order_num"] + 1)).apply(lambda x: x.total_seconds())		
	df = current_data.loc[current_data.hour.isin([10,11,12,16,17,18]), current_data.columns]
	poi_food_duration = df.groupby(["poi_id"])[colname].agg(['median']).reset_index().rename(
		index=str, columns= {
		'median':'mid_poi_per_food_time'
		}
	)
	df_waimai = df_waimai.merge(poi_food_duration, how='left', on=['poi_id'])
	df_waimai["mid_poi_per_food_time"].fillna(df_waimai["mid_poi_per_food_time"].median(),inplace=True)
	df_waimai["predict_foodtimes"] = df_waimai['food_num'] * df_waimai["mid_poi_per_food_time"] * (df_waimai["waiting_order_num"] +1)
	poi_windows = pd.DataFrame()
	hour_array = np.arange(10,22,1)
	for slice_hour in hour_array:
		df = current_data.loc[current_data['hour'] == (slice_hour - 1), current_data.columns]
		timestamp = 'prehour'    				
		poi_hour_duration = df.groupby(["poi_id"])[colname].agg(['mean', 'median']).reset_index().rename(
			index=str, columns= {
			'mean':'avg_poi_' + colname + '_' + timestamp,
			'median':'median_poi_' + colname + '_' + timestamp
			}
		)
		poi_hour_duration["hour"] = slice_hour  
		poi_windows = poi_windows.append(poi_hour_duration)
	addcols = np.setdiff1d(poi_windows.columns.tolist(), df_waimai.columns.tolist())
	df_waimai = df_waimai.merge(poi_windows, how='left', on=['poi_id','hour'])
	for i in addcols:
		df_waimai[i].fillna(df_waimai[i].median(),inplace=True)
		addcolname = "current_" + i
		df_waimai[addcolname] = df_waimai['food_num'] * df_waimai[i] * (df_waimai["waiting_order_num"] +1)
	return df_waimai


def get_poi_order_duration(df_waimai):  
	current_data = copy.deepcopy(df_waimai[df_waimai.type!="test"])
	colname = "per_realdist_food_time"
	current_data[colname]= current_data["delivery_duration"]/current_data["delivery_distance"]/current_data["food_num"]
	df = current_data.loc[current_data.hour.isin([10,11,12,16,17,18]), current_data.columns]
	poi_food_duration = df.groupby(["poi_id"])[colname].agg(['median']).reset_index().rename(
		index=str, columns= {
		'median':'mid_per_realdist_food_time'
		}
	)
	df_waimai = df_waimai.merge(poi_food_duration, how='left', on=['poi_id'])
	df_waimai["mid_per_realdist_food_time"].fillna(df_waimai["mid_per_realdist_food_time"].median(),inplace=True)
	df_waimai["predict_duration"] = df_waimai['food_num'] * df_waimai["mid_per_realdist_food_time"] * df_waimai["delivery_distance"]
	poi_windows = pd.DataFrame()
	hour_array = np.arange(10,22,1)
	for slice_hour in hour_array:
		df = current_data.loc[current_data['hour'] == (slice_hour - 1), current_data.columns]
		timestamp = 'prehour_0060'    				
		poi_hour_duration = df.groupby(["poi_id"])[colname].agg(['mean', 'median']).reset_index().rename(
			index=str, columns= {
			'mean':'avg_poi_' + colname + '_' + timestamp,
			'median':'median_poi_' + colname + '_' + timestamp
			}
		)
		poi_hour_duration["hour"] = slice_hour  ##过滤掉8点前的样本
		poi_windows = poi_windows.append(poi_hour_duration)
	addcols = np.setdiff1d(poi_windows.columns.tolist(), df.columns.tolist())
	df_waimai = df_waimai.merge(poi_windows, how='left', on=['poi_id','hour'])
	for i in addcols:
		df_waimai[i].fillna(df_waimai[i].mean(),inplace=True)
		addcolname = "current_" + i
		df_waimai[addcolname] = df_waimai['delivery_distance'] * df_waimai['food_num'] * df_waimai[i]
	return df_waimai


remove_col = ["mid_per_realdist_food_time",'predict_duration']

for i in waimai.columns:
	if 'prehour_0060' in i:
		remove_col.extend([i])

waimai.drop(remove_col, inplace=True, axis=1)

def poi_userarea_delivery_duration(df_waimai):
	df = df_waimai.loc[(df_waimai.type!="test") & df_waimai.hour.isin([10,11,12,16,17,18])]
	df["poi_userarea_delivery_duration_perfood"] = df["delivery_duration"]/df["food_num"]
	poi_userarea_delivery_duration = df.groupby(['poi_id','user_area_geo']).poi_userarea_delivery_duration_perfood.agg(['mean']).reset_index().rename(
			index=str, columns= {
			'mean':'mean_poi_userarea_delivery_duration_perfood'
			}
	)
	df_waimai = df_waimai.merge(poi_userarea_delivery_duration, how="left", on=["poi_id","user_area_geo"])
	df_waimai["mean_poi_userarea_delivery_duration"] = df_waimai["mean_poi_userarea_delivery_duration_perfood"] * df_waimai["food_num"]
	df_waimai.drop("mean_poi_userarea_delivery_duration_perfood", axis=1, inplace=True)
	return df_waimai


def poi_rider_time(df_waimai,firsttime, sectime):
	#avg、max、min、mid(骑手到店时间-用户下单时间)
	df = df_waimai[(df_waimai.type!="test") & df_waimai.hour.isin([10,11,12,16,17,18])].reset_index()
	if firsttime != "order_unix_time":
		df[firsttime] = df[firsttime].apply(lambda x: datetime.datetime.fromtimestamp(x))
	if sectime != "order_unix_time":
		df[sectime] = df[sectime].apply(lambda x: datetime.datetime.fromtimestamp(x))
	colname = str(firsttime) + "_" + str(sectime)
	df[colname] = (df[firsttime]-df[sectime]).apply(lambda x: x.total_seconds())
	poi_col_time = df.groupby('poi_id')[colname].agg(['mean', 'median']).reset_index().rename(
		index=str,
		columns= {
			'mean':'avg_' + colname,
			'median':'median_' + colname
		}
	)
	df_waimai = df_waimai.merge(poi_col_time, on="poi_id", how="left")
	return df_waimai

def rider_familar_degree(df_waimai, groupkey):
	df = df_waimai[(df_waimai.type!="test") & df_waimai.hour.isin([10,11,12,16,17,18])].reset_index()
	firsttime = "finish_unix_time"
	sectime = "fetch_unix_time"
	df[firsttime] = df[firsttime].apply(lambda x: datetime.datetime.fromtimestamp(x))
	df[sectime] = df[sectime].apply(lambda x: datetime.datetime.fromtimestamp(x))
	colname = groupkey + "_" + str(firsttime) + "_" + str(sectime) + "_perdistance" 
	df[colname] = (df[firsttime]-df[sectime]).apply(lambda x: x.total_seconds())/df["delivery_distance"]
	area_col_time = df.groupby(groupkey)[colname].agg(['mean', 'median']).reset_index().rename(
		index=str,
		columns= {
			'mean':'avg_' + colname,
			'median':'median_' + colname
		}
	)
	addcols = np.setdiff1d(area_col_time.columns.tolist(), df.columns.tolist())
	df_waimai = df_waimai.merge(area_col_time, on=groupkey, how="left")
	for col in addcols:
		newcol = 'current_' + col + 'distance' 
		df_waimai[newcol]= df_waimai[col] * df_waimai["delivery_distance"]	
	return df_waimai



def get_area_realtime_windows(df, colname, groupkey, window_size):
	area_rider_rolling = df.groupby(groupkey)[colname, 'log_unix_time'].rolling(window = window_size, on = 'log_unix_time', closed = 'left')
	area_max = area_rider_rolling.max().reset_index().rename(columns = {colname: 'max_'+ colname +'_area_' + window_size}).drop([groupkey, "log_unix_time"], axis = 1)
	df = df.merge(area_max, how='left', left_index=True, right_on='level_1').drop(['level_1'], axis = 1)
	df[colname + "_degree" + window_size] = df[colname] /(1+ df['max_'+ colname +'_area_' + window_size])
	return df

def userarea_delivery_duration(df_waimai):
	df = df_waimai.loc[(df_waimai.type!="test") & df_waimai.hour.isin([10,11,12,16,17,18])]
	df["finish_unix_time"] = df["finish_unix_time"].apply(lambda x: datetime.datetime.fromtimestamp(x))
	df["fetch_unix_time"] = df["fetch_unix_time"].apply(lambda x: datetime.datetime.fromtimestamp(x))
	colname = "rider_time"
	df[colname] = (df["finish_unix_time"]-df["fetch_unix_time"]).apply(lambda x: x.total_seconds())
	df["rider_speed"] = df[colname]/df["delivery_distance"]/(df["avg_unfetchedorders_riders"]+1)
	userarea_delivery_duration = df.groupby(['user_area_geo']).rider_speed.agg(['median']).reset_index().rename(
			index=str, columns= {
			'median':'median_userarea_rider_speed'
			}
	)
	df_waimai = df_waimai.merge(userarea_delivery_duration, how="left", on=["user_area_geo"])
	df_waimai["median_userarea_delivery_duration"] = df_waimai["median_userarea_rider_speed"] * df_waimai["delivery_distance"] * (df["avg_unfetchedorders_riders"]+1)
	return df_waimai

def poi_per_order_time(df_waimai):
	current_data = copy.deepcopy(df_waimai[df_waimai.type!="test"])
	colname = "per_order_time"
	current_data["fetch_unix_time"] = current_data["fetch_unix_time"].apply(lambda x: datetime.datetime.fromtimestamp(x))
	current_data[colname] = ((current_data["fetch_unix_time"]-current_data["order_unix_time"])).apply(lambda x: x.total_seconds())		
	df = current_data.loc[current_data.hour.isin([10,11,12,16,17,18]), current_data.columns]
	poi_food_duration = df.groupby(["poi_id"])[colname].agg(['median']).reset_index().rename(
		index=str, columns= {
		'median':'mid_poi_per_order_time'
		}
	)
	df_waimai = df_waimai.merge(poi_food_duration, how='left', on=['poi_id'])
	df_waimai["mid_poi_per_order_time"].fillna(df_waimai["mid_poi_per_order_time"].median(),inplace=True)
	return df_waimai

def rider_speed(df_waimai):
	df = df_waimai[(df_waimai.type!="test") & (df_waimai.hour.isin([10,11,12,16,17,18]))].reset_index()
	firsttime = "finish_unix_time"
	sectime = "fetch_unix_time"
	df[firsttime] = df[firsttime].apply(lambda x: datetime.datetime.fromtimestamp(x))
	df[sectime] = df[sectime].apply(lambda x: datetime.datetime.fromtimestamp(x))
	colname =  "userarea_riderduration" 
	df[colname] = (df[firsttime]-df[sectime]).apply(lambda x: math.log(x.total_seconds()+1,2))/df["delivery_distance"]
	area_col_time = df.groupby(["user_area_geo"])[colname].agg(['mean', 'median']).reset_index().rename(
		index=str,
		columns= {
			'mean':'avg_' + colname,
			'median':'median_' + colname
		}
	)
	addcols = np.setdiff1d(area_col_time.columns.tolist(), df.columns.tolist())
	df_waimai = df_waimai.merge(area_col_time, on=["user_area_geo"], how="left")
	for col in addcols:
		newcol = 'period_' + col + 'distance' 
		print col, ":", newcol
		df_waimai[newcol]= df_waimai[col]  * df["delivery_distance"]
	return df_waimai


def getNearestFutureOrder(df, offset):
	df = df.groupby(["poi_id"]).apply(lambda x: x.sort_values(["order_unix_time"], ascending=True)).reset_index(drop=True)
	df['order_unix_time_shifted' + str(offset)] = df.groupby(['poi_id'])[['order_unix_time']].shift(offset)
	df['last_' + str(offset) + '_food_num'] = (df.groupby(['poi_id'])[['food_num']].shift(offset)).fillna(0)
	df['last_' + str(offset) + '_foodvalue'] = (df.groupby(['poi_id'])[['food_total_value']].shift(offset)).fillna(0)
	df['last_' + str(offset) + '_waiting_ordernum'] = (df.groupby(['poi_id'])[['waiting_order_num']].shift(offset)).fillna(0)
	df['last_' + str(offset) + '_OrderInterval'] = ((df['order_unix_time_shifted' + str(offset)] - df['order_unix_time']).apply(lambda x: x.total_seconds())).fillna(3600)
	return df[["order_id",'last_' + str(offset) + '_OrderInterval', 'last_' + str(offset) + '_food_num', 'last_' + str(offset) + '_foodvalue','last_' + str(offset) + '_waiting_ordernum']]




#if __name__ == '__main__':
work_path = "/Users/zhangkai/Desktop/dp_waimai_kaggle/比赛数据"
#work_path = "/data/kai.zhang/MDD/data"
pd.set_option('display.max_columns', None)
os.getcwd() 
os.chdir(work_path) 

waybill = pd.read_csv("waybill_info.csv") #训练数据
waybill_a = pd.read_csv("waybill_info_test_a.csv",sep="\t") #辅助特征
test = pd.read_csv("waybill_info_test_b.csv")
area_realtime = pd.read_csv("area_realtime.csv") #区域实时特征,unit:60s
area_realtime_test = pd.read_csv("area_realtime_test.csv")
train_weather = pd.read_csv("weather_realtime.csv")
test_weather = pd.read_csv("weather_realtime_test.csv")
waybill_a.columns = waybill.columns
	#waybill和waybill_a合并
waybill["type"] = "train"
waybill_a["type"] = "assist"
test["type"] = "test"
waimai = waybill.append(waybill_a, ignore_index=True)
waimai.dropna(inplace=True)
waimai = waimai.append(test, ignore_index=True)


waimai["ori_order_unix_time"] = waimai["order_unix_time"]
waimai = unixtime_to_datetime(waimai, "order_unix_time")
waimai["hour_mins"] =  waimai["hour"] * 60 + waimai["minute"]
waimai["delivery_distance"] = waimai["delivery_distance"].apply(lambda x: math.log(x + 1, 2)) #.replace(0,1)
waimai["delivery_duration"] = waimai["delivery_duration"].apply(lambda x: math.log(x + 1, 2))


#区域内商户数量,#区域内商户数量占比区域平均商户数量
waimai["area_pois"] = waimai.groupby('area_id')["poi_id"].transform('nunique')
area_orders = waimai.groupby('area_id').order_id.count().reset_index().rename(columns = {"order_id": "area_orders"} )
waimai = waimai.merge(area_orders[["area_id","area_orders"]], on ="area_id", how="left") 

#订单均值
waimai["per_food_value"] =  (waimai["box_total_value"] + waimai["food_total_value"])/waimai["food_num"]
pre_order_num_stat = get_poi_order_price_stat(waimai, "per_food_value")
order_num_stat = get_poi_order_price_stat(waimai, "food_total_value")
waimai = waimai.merge(order_num_stat, on='poi_id', how = 'left').merge(pre_order_num_stat, on='poi_id', how = 'left')
waimai["avg_food_total_value_precent"] = waimai["food_total_value"]/waimai["avg_food_total_value"]
waimai["median_food_total_value_precent"] = waimai["food_total_value"]/waimai["median_food_total_value"]
waimai["food_num_per"] = waimai["food_num"] / waimai.groupby('poi_id').food_num.transform('median')


##区域实时特征
all_area_realtime = area_realtime.append(area_realtime_test, ignore_index=True)
all_area_realtime = unixtime_to_datetime(all_area_realtime,"log_unix_time")
all_area_realtime.loc[all_area_realtime.not_fetched_order_num < 0, "not_fetched_order_num"] = 0
all_area_realtime.loc[all_area_realtime.deliverying_order_num < 0, "deliverying_order_num"] = 0
all_area_realtime["unfinished_order_num"] = all_area_realtime["not_fetched_order_num"] + all_area_realtime["deliverying_order_num"]
all_area_realtime["unfetched_order_ratio"] = all_area_realtime["not_fetched_order_num"] / (all_area_realtime["unfinished_order_num"] +1)


#星期、节假日,星期几，分类型
all_area_realtime["weekday"] = all_area_realtime.log_unix_time.apply(lambda day: day.weekday())

#区域内空闲骑手比例、方差
all_area_realtime["notbusy_ratio"] = all_area_realtime["notbusy_working_rider_num"]/all_area_realtime["working_rider_num"]
all_area_realtime["busy_working_riders"] = all_area_realtime["working_rider_num"] - all_area_realtime["notbusy_working_rider_num"]

##No1，空载量
all_area_realtime["avg_unfetchedorders_notbusy_rider"] = all_area_realtime["not_fetched_order_num"]/(all_area_realtime["notbusy_working_rider_num"]+1)

##骑手是否接了很多单，正在接货途中
#all_area_realtime["is_rider_getingorders"] = ((all_area_realtime["deliverying_order_num"] == 0) & (all_area_realtime["busy_working_riders"] > 0)) * 1.0

##忙碌骑手的负载量,重要度很低
all_area_realtime["avg_deliveryingorders_busy_rider"] = (all_area_realtime["deliverying_order_num"] / (all_area_realtime["busy_working_riders"] +1))

all_area_realtime["gap_unfetched_areaorders"] = all_area_realtime["notbusy_working_rider_num"] * all_area_realtime["avg_deliveryingorders_busy_rider"] - all_area_realtime["not_fetched_order_num"]
all_area_realtime["avggap_unfetchedorders_notbusy_rider"] = all_area_realtime["avg_unfetchedorders_notbusy_rider"] - all_area_realtime["avg_deliveryingorders_busy_rider"]

##join实时区域信息
waimai = pd.merge(waimai, all_area_realtime, on=['date', 'hour', 'minute', 'area_id'], how='left')


##商户未完成订单占比区域未取订单
waimai["waiting_order_num_per"] = waimai["waiting_order_num"] / (waimai.groupby('poi_id').waiting_order_num.transform('median') +1)
##商户未完成订单量平均需要多少忙碌骑手
waimai["poi_waitingorders_needriders"] = waimai["waiting_order_num"]/(waimai["avg_deliveryingorders_busy_rider"]+1)

#配送未完成订单额外需要多少骑手
waimai["poi_waitingorders_needextra_riders"]= all_area_realtime["notbusy_working_rider_num"] - waimai["poi_waitingorders_needriders"]
waimai["poi_waitingorders_neednotbusyriders"] = waimai["waiting_order_num"]/(1+waimai["avg_unfetchedorders_notbusy_rider"])
waimai["poi_waitingorders_needextra_unbusyriders"]= all_area_realtime["notbusy_working_rider_num"] - waimai["poi_waitingorders_neednotbusyriders"]
waimai.drop(["poi_waitingorders_needriders","poi_waitingorders_neednotbusyriders"],axis=1,inplace=True)

waimai["avg_unfetchedorders_riders"] = waimai["not_fetched_order_num"]/ waimai["working_rider_num"]
waimai["avg_unfinished_orders_riders"] = waimai["unfinished_order_num"] / waimai["working_rider_num"]
waimai["avg_waitingorders_unbusyriders"] = waimai["waiting_order_num"]/(waimai["notbusy_working_rider_num"] +1)

#join天气数据
weather = train_weather.append(test_weather, ignore_index=True)
weather = unixtime_to_datetime(weather,"log_unix_time")
waimai = pd.merge(waimai, weather.drop(["time","log_unix_time"],axis=1), on=['date', 'hour', 'minute', 'area_id'], how='left')
##向前填充weather
waimai = waimai.sort_values(by=["order_unix_time"],ascending=True).reindex()
waimai['temperature'] = waimai.groupby(['area_id'])['temperature'].ffill()
waimai['wind'] = waimai.groupby(['area_id'])['wind'].ffill()
waimai['rain'] = waimai.groupby(['area_id'])['rain'].ffill()



##区域、商户滑动窗口内的订单量
waimai = get_ordernum_window(waimai, "area_id", '3600s')
waimai = get_ordernum_window(waimai, "area_id", '3000s')
waimai = get_ordernum_window(waimai, "area_id", '2400s')
waimai = get_ordernum_window(waimai, "area_id", '1800s')
waimai = get_ordernum_window(waimai, "area_id", '1200s')
waimai = get_ordernum_window(waimai, "area_id", '600s')
waimai = get_ordernum_window(waimai, "area_id", '300s')

waimai = get_ordernum_window(waimai,"poi_id",'3600s')
waimai = get_ordernum_window(waimai,"poi_id",'3000s')
waimai = get_ordernum_window(waimai,"poi_id",'2400s')
waimai = get_ordernum_window(waimai,"poi_id",'1800s')
waimai = get_ordernum_window(waimai,"poi_id",'1200s')
waimai = get_ordernum_window(waimai,"poi_id",'600s')
waimai = get_ordernum_window(waimai,"poi_id",'300s')



#用户小区，公司订单量
waimai['customer_latitude'] = waimai['customer_latitude'].apply(lambda x: round(x,2))
waimai['customer_longitude'] = waimai['customer_longitude'].apply(lambda x: round(x,2))
waimai["user_area_geo"] = waimai['customer_longitude'].map(str)+'_'+waimai['customer_latitude'].map(str)
waimai["user_area_geo"] = waimai["user_area_geo"].astype('category')
waimai['user_area_geo'].cat.categories= np.arange(1,waimai["user_area_geo"].nunique()+1)
waimai["user_area_geo"] = waimai["user_area_geo"].astype(int)


waimai['poi_lat'] = waimai['poi_lat'].apply(lambda x: round(x,2))
waimai['poi_lng'] = waimai['poi_lng'].apply(lambda x: round(x,2))
waimai["poi_area_geo"] = waimai['poi_lng'].map(str)+'_'+waimai['poi_lat'].map(str)
waimai["poi_area_geo"] = waimai["poi_area_geo"].astype('category')
waimai['poi_area_geo'].cat.categories=range(waimai["poi_area_geo"].nunique())  #148
waimai["poi_area_geo"] = waimai["poi_area_geo"].astype(int)

waimai["poi_area_pois"] = waimai.groupby("poi_area_geo").poi_id.transform('nunique')
waimai["poi_area_hot"] = waimai["poi_area_pois"]/ waimai["area_pois"] 
waimai["poi_area_unbusy_riders"] = waimai["notbusy_working_rider_num"] * waimai["poi_area_hot"]
waimai["avg_waiting_orders_per_poi_unbusy_rider"]= waimai["waiting_order_num"]/(1+waimai["poi_area_unbusy_riders"])
waimai["poiarea_unfetchedorders"] = (waimai["not_fetched_order_num"] * waimai["poi_area_hot"])

#addedby order
waimai["poi_area_orders"] = waimai.groupby("poi_area_geo").order_id.transform('count')
waimai["poi_area_orderhot"] = waimai["poi_area_orders"]/ waimai["area_orders"] 
waimai["poi_area_unbusy_riders_byorders"] = waimai["notbusy_working_rider_num"] * waimai["poi_area_orderhot"]
waimai["avg_waiting_orders_per_poi_unbusy_rider2"]= waimai["waiting_order_num"]/(1+ waimai["poi_area_unbusy_riders_byorders"])
waimai["poiarea_unfetchedorders2"] = (waimai["not_fetched_order_num"] * waimai["poi_area_unbusy_riders_byorders"])


#小区滑动窗口订单量
waimai = get_userarea_orders(waimai, '3600s')
waimai = get_userarea_orders(waimai, '3000s')
waimai = get_userarea_orders(waimai, '2400s')
waimai = get_userarea_orders(waimai, '1800s')
waimai = get_userarea_orders(waimai, '1200s')
waimai = get_userarea_orders(waimai, '600s')
waimai = get_userarea_orders(waimai, '300s')

#小区到商户维度的订单量 与 商户积压订单量差
waimai = poi_userarea_orders(waimai, '3600s')
waimai = poi_userarea_orders(waimai, '3000s')
waimai = poi_userarea_orders(waimai, '2400s')
waimai = poi_userarea_orders(waimai, '1800s')
waimai = poi_userarea_orders(waimai, '1200s')
waimai = poi_userarea_orders(waimai, '600s')
waimai = poi_userarea_orders(waimai, '300s')


##商户出菜时间
waimai = get_poi_foods_time(waimai)

userarea_lastorder= getUserAreaNearestOrder(waimai)
waimai = waimai.merge(userarea_lastorder, on="order_id", how='left')

##商户最近一次订单距离当前订单的间隔时长
nearest_ordertime= getNearestOrder(waimai,1)
waimai = waimai.merge(nearest_ordertime, on="order_id", how='left')

#商户平均距离每菜的送达时间
waimai = get_poi_order_duration(waimai)

#avg、max、min、mid(骑手取餐时间-用户下单时间)
waimai = poi_rider_time(waimai,"arriveshop_unix_time","order_unix_time")

#avg、max、min、mid(骑手取餐时间-骑手到店时间)
waimai = poi_rider_time(waimai,"fetch_unix_time","arriveshop_unix_time")

waimai = rider_familar_degree(waimai, "area_id")


waimai = get_area_realtime_windows(waimai,"not_fetched_order_num","area_id", "3600s")

waimai = get_area_realtime_windows(waimai,"waiting_order_num", "poi_id", "3600s")

waimai = userarea_delivery_duration(waimai)
waimai.drop("max_waiting_order_num_area_3600s",axis=1,inplace=True)

#waiting_orders_time *商户的出餐时间
waimai = poi_per_order_time(waimai)
waimai["waiting_orders_foodtime"] = waimai["waiting_order_num"] * waimai["mid_poi_per_order_time"] 

waimai["poi_id"] = waimai["poi_id"].astype('category')
waimai['poi_id'].cat.categories= np.arange(1,waimai["poi_id"].nunique()+1)
waimai["poi_id"] = waimai["poi_id"].astype(int)

waimai["underloader_ratio"] = waimai["avg_unfetchedorders_notbusy_rider"] / (waimai["avg_deliveryingorders_busy_rider"] + 1)

###poi到小区的平均时长
waimai = poi_userarea_delivery_duration(waimai)
#waimai = poiarea_userarea_delivery_duration(waimai)
#waimai["median_poiuser_area_delivery_duration_perfood"] = waimai["median_poiuser_area_delivery_duration_perfood"].fillna(waimai["median_poiuser_area_delivery_duration_perfood"].median())

waimai = rider_speed(waimai)


nearest_ordertime2= getNearestOrder(waimai,2)
waimai = waimai.merge(nearest_ordertime2, on="order_id", how='left')

waimai["avg_unfetchedorders_riders_delivery_distance"] = waimai["avg_unfetchedorders_riders"] * waimai["delivery_distance"]

waimai["predict_foodtimes_avg_unfetchedorders_riders"] = waimai["predict_foodtimes"] / (1 + waimai["avg_unfetchedorders_riders"])

#####Done!!!!!

nearest_future_ordertime= getNearestFutureOrder(waimai,-1)
waimai = waimai.merge(nearest_future_ordertime, on="order_id", how='left')



waimai_bak = copy.deepcopy(waimai)

waimai = copy.deepcopy(waimai_bak)

waimai = get_area_trend(waimai, all_area_realtime, 'avg_deliveryingorders_busy_rider', 5)




def get_area_trend(df_waimai, all_area_realtime, column_name, minute):
	all_area_realtime_copy = copy.deepcopy(all_area_realtime)
	all_area_realtime_copy["avg_unfetchedorders_riders"] = all_area_realtime_copy["not_fetched_order_num"]/ all_area_realtime_copy["working_rider_num"]
	all_area_realtime_copy['minute'] = all_area_realtime_copy.minute.apply(lambda x: x - minute)
	all_area_realtime_selected = copy.deepcopy(all_area_realtime_copy[['date', 'hour', 'minute', 'area_id', column_name]])
	all_area_realtime_selected.columns = ['date', 'hour', 'minute', 'area_id', column_name + '_' + str(minute)]
	df_waimai = df_waimai.merge(all_area_realtime_selected, how='left', on=['date', 'hour', 'minute', 'area_id'])
	df_waimai[column_name + '_' + str(minute) + '_change'] = (1 + df_waimai[column_name + '_' + str(minute)] - df_waimai[column_name])/(1 + df_waimai["notbusy_working_rider_num"])
	df_waimai.drop([column_name + '_' + str(minute)], axis=1, inplace=True)
	return df_waimai



waimai = get_area_trend(waimai, all_area_realtime, 'not_fetched_order_num', 1)
waimai = get_area_trend(waimai, all_area_realtime, 'not_fetched_order_num', 2)
waimai = get_area_trend(waimai, all_area_realtime, 'not_fetched_order_num', 5)
waimai = get_area_trend(waimai, all_area_realtime, 'not_fetched_order_num', 10)
waimai = get_area_trend(waimai, all_area_realtime, 'not_fetched_order_num', 15)
waimai = get_area_trend(waimai, all_area_realtime, 'not_fetched_order_num', 20)
waimai = get_area_trend(waimai, all_area_realtime, 'not_fetched_order_num', 30)

waimai = get_area_trend(waimai, all_area_realtime, 'deliverying_order_num', 1)
waimai = get_area_trend(waimai, all_area_realtime, 'deliverying_order_num', 2)
waimai = get_area_trend(waimai, all_area_realtime, 'deliverying_order_num', 5)
waimai = get_area_trend(waimai, all_area_realtime, 'deliverying_order_num', 10)
waimai = get_area_trend(waimai, all_area_realtime, 'deliverying_order_num', 15)
waimai = get_area_trend(waimai, all_area_realtime, 'deliverying_order_num', 20)
waimai = get_area_trend(waimai, all_area_realtime, 'deliverying_order_num', 30)



waimai["waiting_ordernum_gap"] = waimai["waiting_order_num"] / (waimai["last_1_OrderInterval"] + 1)

waimai["waiting_ordernum_gap"] = (waimai["last_-1_waiting_ordernum"] -waimai["waiting_order_num"])/(waimai["last_-1_OrderInterval"]+1)

#area: not_fetched_order_num(+deliverying_order_num)/区域每个小时的订单数量

##模型时效性问题；时间段切开统计，出餐时间分工作日




def finish_fun(df_waimai):
	df_waimai = waimai[waimai.type!="test"]
	df_waimai["finish_unix_time"] = df_waimai["finish_unix_time"].apply(lambda x: datetime.datetime.fromtimestamp(x))
	df_waimai['finish_hour'] = df_waimai["finish_unix_time"].apply(lambda x: x.hour)
	df_waimai['finish_minute'] = df_waimai["finish_unix_time"].apply(lambda x: x.minute)
	area_finished_orders_realtime= df_waimai.groupby(["area_id","date","finish_hour","finish_minute"]).order_id.count().reset_index()
	area_finished_orders_realtime.columns = ["area_id","date","hour","minute", "funished_orders"]
	df_waimai= df_waimai.merge(area_finished_orders_realtime, on =["area_id","date","hour","minute"], how="left")
	df_waimai["funished_orders"] = df_waimai["funished_orders"].fillna(0)
	df_waimai["area_unfinished_ratio"] = df_waimai["unfinished_order_num"] / (df_waimai["funished_orders"] + df_waimai["unfinished_order_num"] +1)
	return df_waimai.drop(['finish_hour','finish_minute','funished_orders'],axis=1)

waimai = finish_fun(waimai)


def get_order_counts_in_one_minute_of(df_waimai, group_key,offset):
	order_count = df_waimai.groupby([group_key, 'date', 'hour', 'minute'])['order_id'].count().reset_index()
	order_count.columns = [group_key, 'date', 'hour', 'minute', 'orders_one_minute_of_' + group_key]
	df_waimai = df_waimai.merge(order_count, how='left', on=[group_key, 'date', 'hour', 'minute'])
	area_unfinished_orders_realtime = all_area_realtime.groupby(['area_id', 'date', 'hour', 'minute']).unfinished_order_num.sum().reset_index()
	area_unfinished_orders_realtime.minute = area_unfinished_orders_realtime.minute.apply(lambda x: x-offset)
	area_unfinished_orders_realtime.columns = [group_key, 'date', 'hour', 'minute', "next_unfinished"]
	df_waimai =df_waimai.merge(area_unfinished_orders_realtime, on=['area_id', 'date', 'hour', 'minute'], how="left")
	df_waimai["area_finished_ratio"] = (df_waimai['orders_one_minute_of_' + group_key] - df_waimai["next_unfinished"])/ (df_waimai['orders_one_minute_of_' + group_key] + 1)
	return df_waimai

waimai = get_order_counts_in_one_minute_of(waimai, 'area_id', 1)



waimai = copy.deepcopy(waimai_bak)



def poi_waitingorders_gapspeed(df, offset, speed_col):
	df = df.groupby(["poi_id"]).apply(lambda x: x.sort_values(["order_unix_time"], ascending=True)).reset_index(drop=True)
	df['last_' + str(offset) + '_' + speed_col] = (df.groupby(['poi_id'])[speed_col].shift(offset)).fillna(0)
	df["gapspeed_"+ str(offset) + "_" + speed_col] = (df[speed_col] - df['last_' + str(offset) + '_' + speed_col])/(df['last_' + str(offset) + '_' + speed_col] + 1)
	return df[["order_id","gapspeed_"+ str(offset) + "_" + speed_col]]

waitingorders_gapspeed = poi_waitingorders_gapspeed(waimai,1,"poi_orders_600s")
waimai = waimai.merge(waitingorders_gapspeed,on="order_id",how="left")



def rider_familar_degree2(df_waimai, groupkey):
	df_waimai["noonpeak"] = df_waimai.hour.apply(lambda x :  1 if x in [10,11,12] else 2 if x in [16,17,18] else 0)
	df = df_waimai[(df_waimai.type!="test") & df_waimai.hour.isin([10,11,12,16,17,18])].reset_index()
	firsttime = "finish_unix_time"
	sectime = "fetch_unix_time"
	df[firsttime] = df[firsttime].apply(lambda x: datetime.datetime.fromtimestamp(x))
	df[sectime] = df[sectime].apply(lambda x: datetime.datetime.fromtimestamp(x))
	colname = groupkey + "_" + str(firsttime) + "_" + str(sectime) + "_pernoon" 
	df[colname] = (df[firsttime]-df[sectime]).apply(lambda x: x.total_seconds())/df["delivery_distance"]
	area_col_time = df.groupby([groupkey,"noonpeak"])[colname].agg(['median']).reset_index().rename(
		index=str,
		columns= {
			'median':'median_' + colname
		}
	)
	addcols = np.setdiff1d(area_col_time.columns.tolist(), df.columns.tolist())
	df_waimai = df_waimai.merge(area_col_time, on=[groupkey,"noonpeak"], how="left")
	for col in addcols:
		newcol = 'current_' + col + '_distance' 
		df_waimai[newcol]= df_waimai[col] * df_waimai["delivery_distance"]	
	return df_waimai.drop(['noonpeak'],axis=1)

waimai = rider_familar_degree2(waimai, "area_id")





waimai[np.isinf(waimai["avg_waiting_orders_per_poi_unbusy_rider"])].head()



###LR 调用


waimai = get_ordernum_window(waimai,"poi_area_geo","3600s")
waimai = get_ordernum_window(waimai,"poi_area_geo","3000s")
waimai = get_ordernum_window(waimai,"poi_area_geo","2400s")
waimai = get_ordernum_window(waimai,"poi_area_geo","1800s")
waimai = get_ordernum_window(waimai,"poi_area_geo","1200s")
waimai = get_ordernum_window(waimai,"poi_area_geo","600s")
waimai = get_ordernum_window(waimai,"poi_area_geo","300s")

def rider_speed_prehour(df_waimai):
	df_waimai["workday"] = df_waimai.weekday.isin([5,6]) * 1.0
	df = df_waimai[(df_waimai.type!="test")].reset_index()
	firsttime = "finish_unix_time"
	sectime = "fetch_unix_time"
	df[firsttime] = df[firsttime].apply(lambda x: datetime.datetime.fromtimestamp(x))
	df[sectime] = df[sectime].apply(lambda x: datetime.datetime.fromtimestamp(x))
	colname =  "area_" + str(firsttime) + "_" + str(sectime) 
	df[colname] = (df[firsttime]-df[sectime]).apply(lambda x: x.total_seconds())/df["delivery_distance"].apply(lambda x: math.log(x + 1, 2))
	area_col_time = df.groupby(['area_id','workday','hour'])[colname].agg(['mean', 'median']).reset_index().rename(
		index=str,
		columns= {
			'hour':'stat_hour',
			'mean':'avg_prehour' + colname,
			'median':'median_prehour' + colname
		}
	)
	df_waimai['stat_hour'] = df_waimai['hour'].apply(lambda x: x-1)
	addcols = np.setdiff1d(area_col_time.columns.tolist(), df_waimai.columns.tolist())
	df_waimai = df_waimai.merge(area_col_time, on=['area_id','workday','stat_hour'], how="left")
	for col in addcols:
		newcol = 'current_' + col + 'distance' 
		df_waimai[newcol]= df_waimai[col] * df_waimai["delivery_distance"].apply(lambda x: math.log(x + 1, 2))
		df_waimai.drop(col,inplace=True,axis=1)	
	return df_waimai.drop(['stat_hour','workday'],axis=1)


waimai = rider_speed_prehour(waimai)

def weekday_realtime_poi_foodstime(df_waimai):
	df_waimai["workday"] = df_waimai.weekday.isin([5,6]) * 1.0
	df_waimai["noonpeak"] = df_waimai.hour.apply(lambda x :  1 if x in [10,11,12] else 2 if x in [16,17,18] else 0)
	current_data = copy.deepcopy(df_waimai[df_waimai.type!="test"])
	colname = "pre_realtime_foodtime"
	current_data["fetch_unix_time"] = current_data["fetch_unix_time"].apply(lambda x: datetime.datetime.fromtimestamp(x))
	current_data[colname] = ((current_data["fetch_unix_time"]-current_data["order_unix_time"]) / current_data["food_num"]/ (current_data["waiting_order_num"] + 1)).apply(lambda x: x.total_seconds())	
	poi_foodtime_duration = current_data.groupby(["poi_id","workday","noonpeak"])[colname].agg(['mean']).reset_index().rename(
		index=str, columns= {
		'mean':'pre_realtime_foodtime'
		}
	)
	df_waimai = df_waimai.merge(poi_foodtime_duration, how='left', on=["poi_id","workday","noonpeak"])
	df_waimai["pre_realtime_foodtime"].fillna(df_waimai["pre_realtime_foodtime"].median(),inplace=True)
	df_waimai["predict_realtime_foodtime"] = df_waimai['food_num'] * df_waimai["pre_realtime_foodtime"] * (df_waimai["waiting_order_num"] +1)
	df_waimai.drop(["pre_realtime_foodtime","workday","noonpeak"], inplace=True, axis=1)
	return df_waimai

waimai = weekday_realtime_poi_foodstime(waimai)


def poi_prehour_OrderInterval(df, offset):
	df = df.groupby(["poi_id"]).apply(lambda x: x.sort_values(["order_unix_time"], ascending=True)).reset_index(drop=True)
	df['order_unix_time_shifted'] = df.groupby(['poi_id'])[['order_unix_time']].shift(offset)
	df['last_' + str(offset) + '_OrderInterval'] = ((df['order_unix_time'] - df['order_unix_time_shifted']).apply(lambda x: x.total_seconds())).fillna(3600)
	df.drop("order_unix_time_shifted",axis=1,inplace=True)
	tmp = df.groupby(["poi_id","hour"])['last_' + str(offset) + '_OrderInterval'].mean().reset_index()
	tmp.columns = ["poi_id","hour","mean_order_interval"]
	df = df.merge(tmp,on=["poi_id","hour"], how="left")
	df["avg_unfetchedorders_riders_mean_order_interval"] = df["mean_order_interval"] * df["avg_unfetchedorders_riders"]
	return df

waimai = poi_prehour_OrderInterval(waimai,1)

def get_trend_stat_in_one_hour(df_waimai, group_key, column_name):
	grouped = df_waimai.groupby([group_key, 'date', 'hour'])[column_name]
	stat_median = grouped.median().reset_index()
	stat_median.columns = [group_key, 'date', 'hour', column_name + '_one_hour_median']
	df_waimai = df_waimai.merge(stat_median, how='left', on=[group_key, 'date', 'hour'])
	df_waimai[column_name + '_one_hour_median_ratio'] = df_waimai[column_name] / (df_waimai[column_name + '_one_hour_median'] + 1)
	df_waimai.drop([column_name + '_one_hour_median'], axis=1, inplace=True)
	return df_waimai

#商家、区域一小时趋势量
#poi: waiting_order_num/waiting_order_num中值
waimai = get_trend_stat_in_one_hour(waimai, 'poi_id', 'waiting_order_num')

#area: not_fetched_order_num/not_fetched_order_num的中值
waimai = get_trend_stat_in_one_hour(waimai, 'area_id', 'not_fetched_order_num')

#area: deliverying_order_num/deliverying_order_num的中值
waimai = get_trend_stat_in_one_hour(waimai, 'area_id', 'deliverying_order_num')

#area:  (not)busy_working_rider_num/每小时(not)busy_working_rider_num的中值
waimai = get_trend_stat_in_one_hour(waimai, 'area_id', 'busy_working_riders')

waimai = get_trend_stat_in_one_hour(waimai, 'area_id', 'notbusy_working_rider_num')


##不加，商户/区域每个小时的订单数量， 
def poi_order_ratio(df_waimai):
	poi_hours_orders = df_waimai.groupby(["poi_id","date","hour"]).order_id.count().reset_index()
	area_hours_orders = df_waimai.groupby(["area_id","date","hour"]).order_id.count().reset_index()
	poi_hours_orders.columns=["poi_id","date","hour","poi_hour_orders"]
	area_hours_orders.columns=["area_id","date","hour","area_hour_orders"]
	df_waimai = df_waimai.merge(poi_hours_orders, on =["poi_id","hour","date"], how="left")
	df_waimai = df_waimai.merge(area_hours_orders, on =["area_id","hour","date"], how="left")
	df_waimai["poi_order_hotratio"] = df_waimai["poi_hour_orders"]/df_waimai["area_hour_orders"]
	return df_waimai

waimai = poi_order_ratio(waimai)



