import pandas as pd
import numpy as np
import operator
import sys


#样本选择，将文件读成pd数据结构
def get_input(input_train_file, input_test_file):
    """
    Args:
        input_train_file:
        input_test_file:
    Return:
         pd.DataFrame train_data
         pd.DataFrame test_data
    """
    dtype_dict = {"age": np.int32,
                  "education-num": np.int32,
                  "capital - gain": np.int32,
                  "capital - loss": np.int32,
                  "hours - per - week": np.int32}
    use_list = [i for i in range(15)]
    use_list.remove(2)#不使用id
    train_data_df = pd.read_csv(input_train_file, sep=",", header=0, dtype=dtype_dict, na_values="?", usecols=use_list)
                                #文件，分隔符，特征说明为第一行，数据类型，缺省值为？，要用到哪些特征值
    train_data_df = train_data_df.dropna(axis=0, how="any")
                                    #去除掉所有含有？（na_values）的行
    test_data_df = pd.read_csv(input_test_file, sep=",", header=0, dtype=dtype_dict, na_values="?", usecols=use_list)
    test_data_df = test_data_df.dropna(axis=0, how="any")
    return train_data_df, test_data_df


def label_trans(x):
    """
    Args:
        x: each element in fix col of df
    """
    if x == "<=50K":
        return "0"
    if x == ">50K":
        return "1"
    return "0"


#对label离散化处理，将全部数据的label全部转为0 1
def process_label_feature(lable_feature_str, df_in):
    """
    Args:
        lable_feature_str:"label"
        df_in:DataFrameIn
    """
    df_in.loc[:, lable_feature_str] = df_in.loc[:, lable_feature_str].apply(label_trans)

#将pd数据上对应地方的特征值转化成数字
#将之前的词典转化成{"master":0,"bachelors":1,"primary_school":2}
def dict_trans(dict_in):
    """
    Args:
        dict_in: key str, value int
    Return:
        a dict, key str, value index for example 0,1,2
    """
    output_dict = {}
    index = 0
    for zuhe in sorted(dict_in.items(), key = operator.itemgetter(1), reverse= True):
        output_dict[zuhe[0]] = index
        index += 1
    return output_dict


def dis_to_feature(x, feature_dict):
    """
    Args:
        x: element
        feature_dict: pos dict
    Return:
        a str as "0,0,0,1,0,0,0" 类似于one hot编码
    """
    output_list = [0] * len(feature_dict)
    if x not in feature_dict:
        return ",".join([str(ele) for ele in output_list])
    else:
        index = feature_dict[x]
        output_list[index] = 1
    return ",".join([str(ele) for ele in output_list])

#处理离散化特征
def process_dis_feature(feature_str, df_train, df_test):
    """
    Args:
        feature_str: feature_str
        df_train: train_data_df
        df_test: test_data_df
    Return:
        the dim of the feature output
    process dis feature for lr train
    """
    origin_dict = df_train.loc[:, feature_str].value_counts().to_dict()#统计各个种类出现的次数，如教育{"master"：100,"bachelors:"75,"primary school":70}
    feature_dict = dict_trans(origin_dict)#将之前的词典转化成{"master":0,"bachelors":1,"primary_school":2}
    df_train.loc[:, feature_str] = df_train.loc[:,feature_str].apply(dis_to_feature, args= (feature_dict, ))#将pd数据上对应地方的特征值通过feature_dict词典转化成onehot
    df_test.loc[:, feature_str] = df_test.loc[:,feature_str].apply(dis_to_feature, args= (feature_dict, ))#即将数据中master转化成字符串"1,0,0"
    return len(feature_dict)#返回每个特征离散化的维度


#将describe返回的东西转成分为点对应的值
def list_trans(input_dict):
    """
    Args:
        input_dict:{'count': 30162.0, 'std': 13.134664776855985, 'min': 17.0, 'max': 90.0, '50%': 37.0,
                    '25%': 28.0, '75%': 47.0, 'mean': 38.437901995888865}
    Return:
         a list, [17,28,37,47,90]
    """
    output_list = [0]*5
    key_list = ["min", "25%","50%","75%","max"]
    for index in range(len(key_list)):
        fix_key = key_list[index]
        if fix_key not in input_dict:
            print("error")
            sys.exit()
        else:
            output_list[index] = input_dict[fix_key]
    return output_list


def con_to_feature(x, feature_list):
    """
    Args:
        x: element
        feature_list: list for feature trans
    Return:
        str, "1,0,0,0"
    """
    feature_len = len(feature_list) -1
    result = [0] * feature_len
    for index in range(feature_len):
        if x >= feature_list[index] and x <= feature_list[index + 1]:
            result[index] = 1
            return ",".join([str(ele) for ele in result])
    return ",".join([str(ele) for ele in result])


#处理连续特征
def process_con_feature(feature_str, df_train, df_test):
    """
    Args:
        feature_str: feature_str
        df_train: train_data_df
        df_test: test_data_df
    Return:
        the dim of the feature output
    process con feature for lr train
    """
    origin_dict = df_train.loc[:, feature_str].describe().to_dict()#得到连续特征的分布，用describe函数得到分布
    #包含样本数目，最大值，最小值，标准差，平均值，1/4,2/4，3/4，4/4分位点
    feature_list = list_trans(origin_dict)
    df_train.loc[:, feature_str] = df_train.loc[:, feature_str].apply(con_to_feature, args=(feature_list, ))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply(con_to_feature, args=(feature_list, ))
    return len(feature_list) -1

#train_data_df, test_data_df = get_input("../data/train.txt", "../data/train.txt")
#tmp_feature_num = process_con_feature("age", train_data_df, test_data_df)


def add(str_one, str_two):
    """
    Args:
        str_one:"0,0,1"
        str_two:"1,0,0,0"
    Return:
        str such as"0,0,0,0,0,0,0,0,1,0,0,0"
    """
    list_one = str_one.split(",")
    list_two = str_two.split(",")
    list_one_len = len(list_one)
    list_two_len = len(list_two)
    return_list = [0]*(list_one_len*list_two_len)
    try:
        index_one = list_one.index("1")
    except:
        index_one = 0
    try:
        index_two = list_two.index("1")
    except:
        index_two = 0
    return_list[index_one*list_two_len + index_two] = 1
    return ",".join([str(ele) for ele in return_list])

#用两个特征组成新的组合特征，输入：第一个特征名称，第二个特征名称，新的特征名称
def combine_feature(feature_one, feature_two, new_feature, train_data_df, test_data_df, feature_num_dict):
    """
    Args:
        feature_one:
        feature_two:
        new_feature: combine feature name
        train_data_df:
        test_data_df:
        feature_num_dict: ndim of every feature, key feature name value len of the dim
    Return:
        new_feature_num
    """
    train_data_df[new_feature] = train_data_df.apply(lambda row: add(row[feature_one], row[feature_two]), axis=1)
    test_data_df[new_feature] = test_data_df.apply(lambda row: add(row[feature_one], row[feature_two]), axis=1)
    if feature_one not in feature_num_dict:
        print("error")
        sys.exit()
    if feature_two not in feature_num_dict:
        print("error")
        sys.exit()
    return feature_num_dict[feature_one]*feature_num_dict[feature_two]#返回新特征的维度


def output_file(df_in, out_file):#将处理好的数据写进文件
    """

    write data of df_in to out_file
    """
    fw = open(out_file, "w+")
    for row_index in df_in.index:
        outline = ",".join([str(ele) for ele in df_in.loc[row_index].values])
        fw.write(outline + "\n")
    fw.close()


def ana_train_data(input_train_data, input_test_data, out_train_file, out_test_file, feature_num_file):
    """
    Args:
        input_train_data:
        input_test_data:
        out_train_file:
        out_test_file:
        feature_num_file:
    """
    train_data_df, test_data_df = get_input(input_train_data, input_test_data)
    label_feature_str = "label"
    dis_feature_list = ["workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "sex", "native-country"]
    con_feature_list = ["age","education-num","capital-gain","capital-loss","hours-per-week"]
    index_list = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    process_label_feature(label_feature_str, train_data_df)
    process_label_feature(label_feature_str, test_data_df)
    dis_feature_num = 0#统计所有离散化数据被离散成了多少维度
    con_feature_num = 0#统计所有连续数据被离散成了多少维度
    feature_num_dict = {}#组合特征的维度是两个特征维度的乘积
    for dis_feature in dis_feature_list:
        tmp_feature_num = process_dis_feature(dis_feature, train_data_df, test_data_df)
        dis_feature_num += tmp_feature_num
        feature_num_dict[dis_feature] = tmp_feature_num
    for con_feature in con_feature_list:
        tmp_feature_num = process_con_feature(con_feature, train_data_df, test_data_df)
        con_feature_num += tmp_feature_num
        feature_num_dict[con_feature] = tmp_feature_num
    print(dis_feature_num)
    print(con_feature_num)
    new_feature_len = combine_feature("age", "capital-gain", "age_gain", train_data_df, test_data_df, feature_num_dict)#输入：第一个特征名称，第二个特征名称，新的特征名称
    new_feature_len_two = combine_feature("capital-gain", "capital-loss", "loss_gain", train_data_df, test_data_df, feature_num_dict)
    train_data_df = train_data_df.reindex(columns=index_list +["age_gain","loss_gain","label"])#调整顺序，让label排在最后
    test_data_df = test_data_df.reindex(columns=index_list +["age_gain", "loss_gain", "label"])
    output_file(train_data_df, out_train_file)
    output_file(test_data_df, out_test_file)
    fw = open(feature_num_file, "w+")
    fw.write("feature_num=" + str(dis_feature_num + con_feature_num+new_feature_len+new_feature_len_two))



# if __name__ == "__main__":
#     if len(sys.argv) < 6:
#         print("usage: python xx.py origin_train origin_test train_file test_file feature_num_file")
#         sys.exit()
#     else:
#         origin_train = sys.argv[1]
#         origin_test = sys.argv[2]
#         train_file = sys.argv[3]
#         test_file = sys.argv[4]
#         feature_num_file = sys.argv[5]
#         ana_train_data(origin_train, origin_test, train_file, test_file, feature_num_file)
ana_train_data("../data/train.txt", "../data/test.txt", "../data/train_file.txt", "../data/test_file.txt", "../data/feature_num.txt")
