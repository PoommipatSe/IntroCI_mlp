import sys  
import pandas as pd
import numpy as np
from itertools import repeat
import math
import pickle


input_filename = sys.argv[1]

DELIMITER = '\t'
K_F_VALID = 10
target_num = 8
header_file_num = 0
a = 0.1
n = 0.2
layer_outline = "8-5-1"

df = pd.read_csv(input_filename, delimiter = DELIMITER, header = header_file_num )

#normalized_df=(df-df.mean())/df.std()
#data_target_mean = df.iloc[:,target_num].mean()
#data_target_std = df.iloc[:,target_num].std()

#normalized = 2*((df-df.min())/((df.max())-(df.min())))-1 #[-1,1]

save_df_min = df.min()
save_df_max = df.max()
save_df_range = save_df_max - save_df_min
normalized = (df-save_df_min)/(save_df_range) #[0,1]

df = normalized

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


def seperate_data_to_training_target(df_set, target_num):
    #column start at index 0
    df_temp = df_set
    df_feed = df_temp.iloc[:,:target_num]
    df_desire = df_temp.iloc[:,target_num:]
    return df_feed, df_desire


def layer_extract_architect(layer_outline):
    layer_outline_list = layer_outline.split("-")
    return layer_outline_list

def weight_init(layer_outline_list):
    np.random.seed(seed=2)
    w_l = []
    for i,j in zip(layer_outline_list, layer_outline_list[1:]):
        weight = []
        for j2 in range(0,int(j)):
            weight.append(np.random.standard_normal(int(i)+1))
        w_l.append(weight)
    return w_l, w_l

def act_fnct(v):
    return sigmoid(v)

def act_fnct_dif(v):
    return sigmoid_dif(v)

def linear_fnct(v):
    return np.multiply(1,v) 

def linear_dif_fnct(v):
    return np.divide(v,v) 

def sigmoid(x):
  return 1 / (1 + np.exp(np.multiply(x,-1)))

def sigmoid_dif(x):
    return np.multiply(x,(np.subtract(1,x))) 

def tanh_f(x):
    return np.tanh(x)

def tanh_f_dif(x):
    #fix me
    return 1 - np.multiply(np.tanh(x),np.tanh(x))

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

        y = act_fnct(v).tolist()
        y.insert(0,1)
        y_out.append(y)
        step_count += 1

    y_out[-1].pop(0)
    return y_out


def back_propagation(y_out, d):
    global start_train_flag
    global w_old
    e = np.subtract(d,y_out[-1]) # e = d - y

    #compute local gradient
    grdnt_local = np.multiply(e,np.multiply(act_fnct_dif(y_out[-1]),y_out[-1]))
    grdnt_list = []
    grdnt_list.append(grdnt_local)
    
    step_count = 0
    delta_w_old_c_list = []
    for i,j in zip(layer_outline_list_rv, layer_outline_list_rv[1:]):

        #compute hidden layer gradient
        y_c = (y_out[-(step_count+2)])
        grdnt_list_step = []
        for j2 in range(0,int(j)+1):
            
            if j2 == 0:
                continue

            temp_y_mult_y_dif = np.multiply(y_c,act_fnct_dif(y_c))
            grdnt = 0
            sum_prod_grdnt_weight_output = 0     
            for i2 in range(0,int(i)):
                grdnt_mult_weight = np.sum(np.multiply(grdnt_list[step_count], w[-(step_count+1)][i2][j2]))
                
                sum_prod_grdnt_weight_output += grdnt_mult_weight
                
            grdnt = temp_y_mult_y_dif[j2] * sum_prod_grdnt_weight_output
            grdnt_list_step.append(grdnt)

        grdnt_list.append(grdnt_list_step)
        
        #compute weight
        delta_weight_list = []
        for j2 in range(0,int(j)+1):
            
            delta_w_c = n * np.multiply(grdnt_list[step_count],y_c[j2]) #n*grdnt*y
            
            if start_train_flag :
                delta_w_com = np.add(0, delta_w_c)
                start_train_flag = False
            else:
                delta_w_com = np.add(0, delta_w_c)
            
            delta_weight_list.append(delta_w_com)
        
        #update
        
        for num in range(0,len(delta_weight_list)):
            for num2 in range(0,len(delta_weight_list[num])):
                #print(num,num2)
                w[-(step_count+1)][num2][num] += a*(w[-(step_count+1)][num2][num] - w_old[-(step_count+1)][num2][num] ) + delta_weight_list[num][num2]
    
        step_count += 1
    
    w_old = w
    error_this_pass = sum(np.multiply(e,e))/len(e) 
    return error_this_pass

#initialize
start_train_flag = True
train_set, test_set = generate_train_test_from_dataset(df,K_F_VALID)
layer_outline_list = layer_extract_architect(layer_outline)
layer_outline_list_rv = layer_outline_list[::-1]

w, w_delta_old = weight_init(layer_outline_list)
w_old = w

""" with open('train.pickle', 'rb') as f:
    w, w_delta_old = pickle.load(f) """


#fix : loop the folds
working_set = train_set[0]
working_test_set = test_set[0]


df_feed, df_desire = seperate_data_to_training_target(working_set,target_num)

def mlp_train(epoch_max,df_feed, df_desire):
    
    error_epoch_list_compute_list = []
    for epoch in range(0,epoch_max):
        error_epoch_list = []

        for item in range(0,len(df_feed)):
            line_in = df_feed.iloc[item].tolist()
            d = df_desire.iloc[item,].tolist()
            y_out = feed_forward(line_in)
            e = back_propagation(y_out,d)
            error_epoch_list.append(e)

        error_epoch_list_compute = (sum(error_epoch_list))/len(df_feed)
        error_epoch_list_compute_list.append(error_epoch_list_compute)
        print("process : ", epoch ," / ", epoch_max)
        

    return error_epoch_list_compute_list

e_return = mlp_train(500, df_feed, df_desire)
print(e_return)

""" with open('train.pickle', 'wb') as f:
    pickle.dump([w,w_delta_old],f) """

df_feed_test, df_desire_test = seperate_data_to_training_target(working_test_set,target_num)

def predictor_cross_funct(y):
    #for cross.pat dataset
    if y[0] > y[1]:
        return [1.0, 0.0]
    else:
        return [0.0, 1.0]

def find_acc_cross_pat(df_feed_test):
    correct = 0
    for item in range(0,len(df_feed_test)):
        
        line_in = df_feed_test.iloc[item].tolist()
        d = df_desire_test.iloc[item,].tolist()
        y_out = feed_forward(line_in)
        predict_cross = predictor_cross_funct(y_out[-1])
        if predict_cross == d:
            correct += 1
            
    return correct/len(df_feed_test)
    

def denorm_range(y):
    return (y*save_df_range[target_num]) + save_df_min[target_num] #from [0,1] to original

def find_error_flood(df_feed_test):
    error_eval = 0
    for item in range(0,len(df_feed_test)):
            
        line_in = df_feed_test.iloc[item].tolist()
        d = df_desire_test.iloc[item,].tolist()
        y_out = feed_forward(line_in)
        error_eval += abs(denorm_range(y_out[-1][0]) - denorm_range(d[0]))
    return error_eval/len(df_feed_test)

error_eval = find_error_flood(df_feed_test)
print(error_eval)
#accuracy = find_acc_cross_pat(df_feed_test)
#print(accuracy)

