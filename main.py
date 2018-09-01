import sys  
import pandas as pd

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

#train_set, test_set = generate_train_test_from_dataset(df,K_F_VALID)
#print(test_set[0:9])