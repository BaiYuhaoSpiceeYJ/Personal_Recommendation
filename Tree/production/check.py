from __future__ import division
import numpy as np
import xgboost as xgb
import train as TA
from scipy.sparse import csc_matrix
import math
import sys

def get_test_data(test_file, feature_num_file):

    """
    Args:
        test_file:file to check performance
        feature_num_file: the file record total num of feature
    Return:
         two np array: test _feature, test_label
    """
    total_feature_num = 103
    test_label = np.genfromtxt(test_file, dtype= np.float32, delimiter=",", usecols= -1)
    feature_list = range(total_feature_num)
    test_feature = np.genfromtxt(test_file, dtype= np.float32, delimiter=",", usecols= feature_list)
    return test_feature, test_label


def sigmoid(x):
    """
    sigmoid function
    """
    return 1/(1+math.exp(-x))


def get_auc(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of  test data
    auc = (sum(pos_index)-pos_num(pos_num + 1)/2)/pos_num*neg_num
    """
    total_list = []
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        label = test_label[index]
        total_list.append((label, predict_score))
    sorted_total_list = sorted(total_list, key = lambda ele:ele[1])
    neg_num = 0
    pos_num = 0
    count = 1
    total_pos_index = 0
    for zuhe in sorted_total_list:
        label, predict_score = zuhe
        if label == 0:
            neg_num += 1
        else:
            pos_num += 1
            total_pos_index += count
        count += 1
    auc_score = (total_pos_index - (pos_num)*(pos_num + 1)/2) / (pos_num*neg_num)
    print("auc:%.5f" %(auc_score))


def get_accuary(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of test data
    """
    score_thr = 0.5
    right_num = 0
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        if predict_score >= score_thr:
            predict_label = 1
        else:
            predict_label = 0
        if predict_label == test_label[index]:
            right_num += 1
    total_num = len(predict_list)
    accuary_score = right_num/total_num
    print("accuary:%.5f" %(accuary_score))


#测试纯gbdt模型，测试文件，gbdt文件保存路径，特征数
def run_check(test_file, tree_model_file, feature_num_file):
    """
    Args:
        test_file:file to test performance
        tree_model_file:gbdt model filt
        feature_num_file:file to store feature num
    """
    test_feature, test_label = get_test_data(test_file, feature_num_file)
    tree_model = xgb.Booster(model_file=tree_model_file)#加载模型
    predict_list = tree_model.predict(xgb.DMatrix(test_feature))
    get_auc(predict_list, test_label)
    get_accuary(predict_list, test_label)


def run_check_lr_gbdt(test_file, tree_mix_model_file, lr_coef_mix_model_file, feature_num_file):
    """
    Args:
        test_file:
        tree_mix_model_file: tree part of mix model
        lr_coef_mix_model_file:lr part of mix model
        feature_num_file:
    """
    test_feature, test_label = get_test_data(test_file, feature_num_file)#获取测试数据
    mix_tree_model = xgb.Booster(model_file=tree_mix_model_file)#恢复树模型
    mix_lr_coef = np.genfromtxt(lr_coef_mix_model_file, dtype=np.float32, delimiter=",")#恢复lr模型
    tree_leaf = mix_tree_model.predict(xgb.DMatrix(test_feature), pred_leaf=True)#让树预测叶子节点
    (tree_depth, tree_num, step_size) = (4,10,0.3)
    total_feature_list = TA.get_gbdt_and_lr_feature(tree_leaf, tree_depth=tree_depth, tree_num=tree_num)#通过叶子节点得到lr的输入
    #print(total_feature_list.tocsc()[0])
    sigmoid_ufunc = np.frompyfunc(sigmoid, 1, 1)
    predict_list = sigmoid_ufunc(np.dot(csc_matrix(mix_lr_coef), total_feature_list.tocsc().T).toarray()[0])#mix_lr_coef是稀疏矩阵，要转成csc模式便于计算
    get_auc(predict_list, test_label)
    get_accuary(predict_list, test_label)

if __name__ == "__main__":
    # if len(sys.argv) == 4:
    #     test_file = sys.argv[1]
    #     tree_model = sys.argv[2]
    #     feature_num_file = sys.argv[3]
    #     run_check(test_file, tree_model, feature_num_file)
    # elif len(sys.argv) == 5:
    #     test_file = sys.argv[1]
    #     tree_mix_model = sys.argv[2]
    #     lr_coef_mix_model = sys.argv[3]
    #     feature_num_file = sys.argv[4]
    #     run_check_lr_gbdt(test_file, tree_mix_model, lr_coef_mix_model,  feature_num_file)
    # else:
    #     print("check gbdt model usage: python xx.py test_file  tree_model feature_num_file")
    #     print("check lr_gbdt model usage: python xx.py test_file tree_mix_model lr_coef_mix_model feature_num_file")
    #     sys.exit()

    run_check("../data/test_file.txt","../data/xgb.model", "../data/feature_num.txt")
    run_check_lr_gbdt("../data/test_file.txt", "../data/xgb_mix_model", "../data/lr_coef_mix_model.txt", "../dta/feature_num.txt")
