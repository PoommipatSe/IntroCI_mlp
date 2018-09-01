import sys  
import pandas as pd
import numpy as np
from itertools import repeat

input_filename = sys.argv[1]
DELIMITER = '\t'
K_F_VALID = 10
target_num = 8

df = pd.read_csv(input_filename, delimiter = DELIMITER, header = 0 )
""" normalized_df=(df-df.mean())/df.std()
data_target_mean = df.iloc[:,8].mean()
data_target_std = df.iloc[:,8].std() """


def k_fold_cross_valiation(df, K_F_VALID):
    df_fold = df.sample(frac = 1, random_state = 2)
    BIN_SIZE = len(df_fold) // K_F_VALID

    fold_dataset_list = []
    for i in range(0,K_F_VALID):
        temp_df = df_fold.iloc[i:i+1*BIN_SIZE,:]
        fold_dataset_list.append(temp_df)
    return fold_dataset_list

def k_fold_combine_to_train_test_set(fold_dataset_list, K_F_VALID):
    train_set = []
    test_set = []
    for i in range(0,K_F_VALID):
        frames = []
        for i2 in range(0,K_F_VALID):
            if i2 == i:
                test_set.append(fold_dataset_list[i2])
            else:
                frames.append(fold_dataset_list[i2])
        train_set.append(pd.concat(frames))
    return train_set, test_set

def generate_train_test_from_dataset(df, K_F_VALID):
    fold_dataset_list = k_fold_cross_valiation(df,K_F_VALID)
    train_set, test_set = k_fold_combine_to_train_test_set(fold_dataset_list, K_F_VALID)
    return train_set, test_set

train_set, test_set = generate_train_test_from_dataset(df,K_F_VALID)
#print(test_set[0:9])

def seperate_data_to_training_target(df_set, target_num):
    #column start at index 0
    df_temp = df_set
    df_feed = df_temp.iloc[:,:target_num]
    df_desire = df_temp.iloc[:,target_num:]
    return df_feed, df_desire

layer_outline = "8-5-1"

def layer_extract_architect(layer_outline):
    layer_outline_list = layer_outline.split("-")
    return layer_outline_list

layer_outline_list = layer_extract_architect(layer_outline)

def weight_init(layer_outline_list):
    np.random.seed(seed=2)
    w_l = []
    w_del_def = []
    for i,j in zip(layer_outline_list, layer_outline_list[1:]):
        weight = []
        weight_zero = []
        for j2 in range(0,int(j)):
            weight.append(np.random.standard_normal(int(i)+1))
            weight_zero.append(list(repeat(0,int(i)+1)))
        w_l.append(weight)
        w_del_def.append(weight_zero)
    return w_l, w_del_def

def act_fnct_dif(v):
    return linear_diff_fnct(v)

def act_fnct(v):
    return linear_fnct(v)

def linear_fnct(v):
    return v 

def linear_diff_fnct(v):
    try:
        temp = []
        for i in range(0, len(v)):
            temp.append(1)
        return temp
    except TypeError:
        return 1

w, w_delta_old = weight_init(layer_outline_list)
working_set = train_set[0]
working_test_set = test_set[0]

df_feed, df_desire = seperate_data_to_training_target(working_set,target_num)

line_in = df_feed.iloc[1].tolist()
d = df_desire.iloc[1,].tolist()

def feed_forward(line_in):
    step_count = 0
    y_out = []
    for i,j in zip(layer_outline_list, layer_outline_list[1:]):
        v = []
        if step_count == 0:
            line_in.insert(0,1) #add bias input
            y_out.append(line_in)

        for j2 in range(0,int(j)):
            temp = sum(np.multiply(y_out[step_count], w[step_count][j2]))
            v.append(temp)
        y = act_fnct(v)
        y.insert(0,1)
        y_out.append(y)
        step_count += 1

    y_out[-1].pop(0)
    return y_out

y_out = feed_forward(line_in)

def back_propagation(y_out, d):
    e = np.subtract(d,y_out[-1]) # e = d - y

    grdnt_local = np.multiply(e,np.multiply(act_fnct_dif(y_out[-1]),y_out[-1]))
    grdnt_list.append(grdnt_local)

    step_count = 0
    for i,j in zip(layer_outline_list_rv, layer_outline_list_rv[1:]):

        #compute gradient
        for i2 in range(0,int(i)):
            node_count = 0
            y_c = (y_out[-(step_count+2)])
            frame_grdnt = []
            for j2 in range(0, int(j)+1):
                if node_count == 0:
                    node_count += 1
                    continue
                
                sum_prod_grdnt_weight_output = sum(np.multiply(grdnt_local, w[-(step_count+1)][i2][j2]))
                grdnt = (y_c[j2]*act_fnct_dif(y_c[j2])*sum_prod_grdnt_weight_output)
                frame_grdnt.append(grdnt)
                node_count += 1
            grdnt_list.append(frame_grdnt)

        #compute weight
        for i2 in range(0,int(i)):
            y_c = (y_out[-(step_count+2)])

            delta_w_c = n * np.multiply(grdnt_local,y_c)
            delta_w_old_c =  np.multiply(a,w_delta_old[-(step_count+1)][i2])
            delta_w_com = np.add(delta_w_old_c, delta_w_c)

            w_c = (w[-(step_count+1)][i2])
            w_new = np.add(w_c, delta_w_com)
            
            #update
            w_delta_old[-(step_count+1)][i2] = delta_w_c #delta_w_old_c
            w[-(step_count+1)][i2] = w_new #w_c

        step_count += 1

    return e

grdnt_list = []

layer_outline_list_rv = layer_outline_list[::-1]

a = 0.5
n = 0.5

print(w)
e = back_propagation(y_out,d)
print("xxx")
print(w)
print(e)



