import sys
from sklearn.linear_model import LogisticRegressionCV as lrcv
from sklearn.externals import joblib
import get_feature_num as gf
import numpy as np

#处理好的训练文件，lr模型每一个特征对应的参数保存的地址，直接实例化输出整个模型的地址（两种方法都可以保存模型）
def train_lr_model(train_file, model_coef, model_file, feature_num_file):
    """
    Args:
        train_file: process file for lr train
        model_coef: w1 w2...        model_file:model pkl
        feature_num_file: file to record num of feature
    """
    total_feature_num = gf.get_feature_num(feature_num_file)
    train_label = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols= -1)#读入文件，数据类型，分隔符，label为倒数第一个
    feature_list = range(total_feature_num)
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols= feature_list)#读入文件，数据类型，分隔符，使用除了label的所有数据
    lr_cf = lrcv(Cs=[1,10,100], penalty="l2", tol=0.0001, max_iter=500, cv=5).fit(train_feature, train_label)#正则化参数，支持多组参数选择。1/1,1/10,1/100，l2正则化，tol迭代停止条件，交叉验证5
    scores = list(lr_cf.scores_.values())[0]#训练的效果，发现正则化参数取1时性能最好
    print("diff:%s" % (",".join([str(ele) for ele in scores.mean(axis = 0)])))#查看每组参数的平均正确率。每组参数有5个准确率，因为用了5组交叉验证。
    print("Accuracy:%s (+-%0.4f)" %(scores.mean(), scores.std()*2))
    lr_cf = lrcv(Cs=[1,10,100], penalty="l2", tol=0.0001, max_iter=500, cv=5, scoring="roc_auc").fit(train_feature, train_label)#查看每组参数的AUC
    scores = list(lr_cf.scores_.values())[0]
    print("diff:%s" % (",".join([str(ele) for ele in scores.mean(axis=0)])))
    print("AUC:%s (+-%0.4f)" %(scores.mean(), scores.std()*2))#输出所有准确率的平均值
    coef = list(lr_cf.coef_)[0]
    fw = open(model_coef, "w+")
    fw.write(",".join(str(ele) for ele in coef))
    fw.close()
    joblib.dump(lr_cf, model_file)


if __name__ == "__main__":
    # if len(sys.argv) < 5:
    #     print("usage: python xx.py train_file coef_file model_file featuren_num_file")
    #     sys.exit()
    # else:
    #     train_file = sys.argv[1]
    #     coef_file = sys.argv[2]
    #     model_file = sys.argv[3]
    #     feature_num_file = sys.argv[4]
    #     train_lr_model(train_file, coef_file, model_file, feature_num_file)

    train_lr_model("../data/train_file.txt", "../data/lr_coef.txt", "../data/lr_model_file", "../data/feature_num.txt")