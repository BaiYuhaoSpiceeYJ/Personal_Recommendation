from __future__ import division
import numpy as np
from sklearn.externals import joblib
import math
import sys
import get_feature_num as gf

#返回测试集的特征以及label
def get_test_data(test_file, feature_num_file):
    """
    Args:
        test_file:file to check performance
        feature_num_file: the file record total num of feature
    Return:
         two np array: test _feature, test_label
    """
    total_feature_num = gf.get_feature_num(feature_num_file)
    test_label = np.genfromtxt(test_file, dtype= np.float32, delimiter=",", usecols= -1)
    feature_list = range(total_feature_num)
    test_feature = np.genfromtxt(test_file, dtype= np.float32, delimiter=",", usecols= feature_list)
    return test_feature, test_label

#用lr-model预测
def predict_by_lr_model(test_feature, lr_model):
    """
    predict by lr_model
    """
    result_list = []
    prob_list = lr_model.predict_proba(test_feature)
    for index in range(len(prob_list)):
        result_list.append(prob_list[index][1])#index 0 为预测为0的概率 index 1 为预测为1的概率
    return result_list

#用coef预测
def predict_by_lr_coef(test_feature, lr_coef):
    """
    predict by lr_coef
    """
    sigmoid_func = np.frompyfunc(sigmoid, 1, 1)#对array每一个元素进行sigmoid操作
    return sigmoid_func(np.dot(test_feature, lr_coef))


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
    auc = (sum(pos_index)-pos_num(pos_num + 1)/2)/pos_num*neg_num#所有正样本的index求和-正样本的数目
    """
    total_list = []
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        label = test_label[index]
        total_list.append((label, predict_score))#[(0.0, 0.01294903179809861), (0.0, 0.6884381114171706), (1.0, 0.6003662665529453)]
    sorted_total_list = sorted(total_list, key = lambda ele:ele[1])#按照得分进行排序，从小到大
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


#特征，label，model，打分的函数（如果用参数模型，就要自己去乘，如果引入整个模型，则不需要，可以直接调用scikit-learn的参数）
def run_check_core(test_feature, test_label, model, score_func):
    """
    Args:
        test_feature:
        test_label:
        model: lr_coef, lr_model
        score_func: use different model to predict
    """
    predict_list = score_func(test_feature, model)
    get_auc(predict_list, test_label)
    get_accuary(predict_list, test_label)


def run_check(test_file, lr_coef_file, lr_model_file, feature_num_file):
    """
    Args:
        test_file: file to check performace
        lr_coef_file: w1,w2
        lr_model_file: dump file
        feature_num_file: file to record num of feature
    """
    test_feature, test_label = get_test_data(test_file, feature_num_file)#读取test数据
    lr_coef = np.genfromtxt(lr_coef_file, dtype=np.float32, delimiter=",")#读取模型参数
    lr_model = joblib.load(lr_model_file)#引入模型
    run_check_core(test_feature, test_label, lr_model, predict_by_lr_model)
    run_check_core(test_feature, test_label, lr_coef, predict_by_lr_coef)#两种结果准确率差距大，不能用0.5作为正负样本的界定，应该调大


if __name__ == "__main__":
    # if len(sys.argv) < 5:
    #     print("usage: python xx.py test_file coef_file model_file feature_num_file")
    #     sys.exit()
    # else:
    #     test_file = sys.argv[1]
    #     coef_file = sys.argv[2]
    #     model_file = sys.argv[3]
    #     feature_num_file = sys.argv[4]
    #     run_check(test_file, coef_file, model_file, feature_num_file)
    run_check("../data/test_file.txt", "../data/lr_coef.txt", "../data/lr_model_file", "../data/feature_num.txt")

