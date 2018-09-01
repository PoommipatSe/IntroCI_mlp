import sys  
import pandas as pd
import numpy as np

input_filename = sys.argv[1]
DELIMITER = '\t'
K_F_VALID = 10

df = pd.read_csv(input_filename, delimiter = DELIMITER, header = 0 )



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

layer_outline = "8-5-3-1"

def layer_extract_architect(layer_outline):
    layer_outline_list = layer_outline.split("-")
    return layer_outline_list

layer_outline_list = layer_extract_architect(layer_outline)

def weight_init(layer_outline_list):
    np.random.seed(seed=2)
    w_l = []
    for i,j in zip(layer_outline_list, layer_outline_list[1:]):
        print(i, j)
        weight = []
        for j2 in range(0,int(j)):
            weight.append(np.random.standard_normal(int(i)+1))
        w_l.append(weight)
    return w_l

w = weight_init(layer_outline_list)
print(w[2])

working_set = train_set[0]
working_test_set = test_set[0]

df_feed, df_desire = seperate_data_to_training_target(working_set,8)

line_in = df_feed.iloc[1].tolist()
line_in.insert(0,1) #add bias input

#print(line_in)
