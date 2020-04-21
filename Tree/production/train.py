import xgboost as xgb
import get_feature_num as GF
import numpy as np
import sys
from sklearn.linear_model import LogisticRegressionCV as LRCV
from scipy.sparse import coo_matrix

#读取训练数据
def get_train_data(train_file, feature_num_file):
    """
    get train data and label for training
    """
    total_feature_num = GF.get_feature_num(feature_num_file)
    train_label = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols= -1)
    feature_list = range(total_feature_num)
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols= feature_list)
    return train_feature, train_label

#选取最优参数
def grid_search(train_mat):
    """
    Args:
        train_mat: train data and train label
    select the best parameter for training model
    """
    result_list = []
    tree_depth_list = [4, 5, 6]
    tree_num_list = [10, 50, 100]
    learning_rate_list = [0.3, 0.5, 0.7]
    for ele_tree_depth in tree_depth_list:
        for ele_tree_num in tree_num_list:
            for ele_learning_rate in learning_rate_list:
                result_list.append((ele_tree_depth, ele_tree_num, ele_learning_rate))
    para_list = result_list
    for ele in para_list:
        (tree_depth, tree_num, learning_rate) = ele
        para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent": 1}
        res = xgb.cv(para_dict, train_mat, tree_num, nfold=5, metrics={'auc'})
        auc_score = res.loc[tree_num-1, ['test-auc-mean']].values[0]#10棵树，选取最后一棵的结果
        print("tree_depth:%s,tree_num:%s, learning_rate:%s, auc:%f" \
              %(tree_depth, tree_num, learning_rate, auc_score))


#训练文件、特征维度、存储路径
def train_tree_model(train_file , feature_num_file, tree_model_file):
    """
    Args:
        train_file: data for train model
        tree_model_file: file to store model
        feature_num_file:file to record feature total num
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)#获取训练数据
    train_mat = xgb.DMatrix(train_feature, train_label)
    # grid_search(train_mat)
    tree_num = 10
    tree_depth = 4
    learning_rate = 0.3
    para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent": 1}
    #深度，步长，目标函数，不输出构造树时的信息
    bst = xgb.train(para_dict, train_mat, tree_num)
    bst.save_model(tree_model_file)


#使用稀疏矩阵存储.将数据转成lr的输入
def get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth):
    """
    Args:
        tree_leaf: prediction of the tree model
        tree_num:total_tree_num
        tree_depth:total_tree_depth
    Return:
         Sparse Matrix to record total train feature for lr part of mixed model
    """
    total_node_num = 2**(tree_depth + 1) - 1 #计算总的节点数
    yezi_num = 2**tree_depth #叶子节点数
    feiyezi_num = total_node_num - yezi_num
    total_col_num = yezi_num*tree_num
    total_row_num = len(tree_leaf)
    col = []
    row = []
    data = []
    base_row_index = 0
    #tree_leaf[0]:[15 18 15 15 23 27 13 17 28 21]10棵树共10个结果
    for one_result in tree_leaf:
        base_col_index = 0
        for fix_index in one_result:
            yezi_index = fix_index - feiyezi_num
            yezi_index  = yezi_index if yezi_index >= 0 else 0 #有些树学习不完全，叶子系数不应该小于0
            col.append(base_col_index + yezi_index)#第一棵树占据0-15维，第二棵树16-31维
            row.append(base_row_index)
            data.append(1)
            base_col_index += yezi_num
        base_row_index += 1
    total_feature_list = coo_matrix((data, (row,col)), shape=(total_row_num, total_col_num))
    return total_feature_list


#训练混合模型。特征文件，文件特征维度，混合模型树部分保存文件，混合模型lr部分保存文件
def train_tree_and_lr_model(train_file, feature_num_file, mix_tree_model_file, mix_lr_model_file):
    """
    Args:
        train_file:file for training model
        feature_num_file:file to store total feature len
        mix_tree_model_file: tree part of the mix model
        mix_lr_model_file:lr part of the mix model
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)#转成gbdt所需数据格式
    tree_num = 10
    tree_depth = 4
    learning_rate = 0.3
    para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent": 1}
    # 深度，步长，目标函数，不输出构造树时的信息
    bst = xgb.train(para_dict, train_mat, tree_num)
    bst.save_model(mix_tree_model_file)

    tree_leaf = bst.predict(train_mat, pred_leaf=True)#让树去预测样本落在哪个节点
    total_feature_list = get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth)#加工节点返回的结果得到lr训练所需的特征

    lr_clf = LRCV(Cs=[1.0], penalty='l2', dual=False, tol=0.0001, max_iter=500, cv=5).fit(total_feature_list, train_label)#训练lr
    scores = list(lr_clf.scores_.values())[0]
    print("diffC:%s" % (','.join([str(ele) for ele in scores.mean(axis=0)])))
    print("Accuracy:%f(+-%0.2f)" % (scores.mean(), scores.std() * 2))
    lr_clf = LRCV(Cs=[1.0], penalty='l2', dual=False, tol=0.0001, max_iter=500, scoring='roc_auc', cv=5).fit(
        total_feature_list, train_label)
    scores = list(lr_clf.scores_.values())[0]
    print("diffC:%s" % (','.join([str(ele) for ele in scores.mean(axis=0)])))
    print("AUC:%f,(+-%0.2f)" % (scores.mean(), scores.std() * 2))
    print('0')
    fw = open(mix_lr_model_file, "w+")
    coef = list(lr_clf.coef_)[0]
    fw.write(','.join([str(ele) for ele in coef]))
    fw.close()

if __name__ == "__main__":

    # if len(sys.argv) == 4:
    #     train_file = sys.argv[1]
    #     feature_num_file = sys.argv[2]
    #     tree_model = sys.argv[3]
    #     train_tree_model(train_file, feature_num_file, tree_model)
    # elif len(sys.argv) == 5:
    #     train_file = sys.argv[1]
    #     feature_num_file = sys.argv[2]
    #     tree_mix_model = sys.argv[3]
    #     lr_coef_mix_model = sys.argv[4]
    #     train_tree_and_lr_model(train_file,  feature_num_file, tree_mix_model, lr_coef_mix_model)
    # else:
    #     print("train gbdt model usage: python xx.py train_file feature_num_file tree_model")
    #     print("train lr_gbdt model usage: python xx.py train_file feature_num_file tree_mix_model lr_coef_mix_model")
    #     sys.exit()
    train_tree_model("../data/train_file.txt", "../data/feature_num.txt", "../data/xgb.model")
    train_tree_and_lr_model("../data/train_file.txt", "../data/feature_num.txt","../data/xgb_mix_model", "../data/lr_coef_mix_model.txt")
