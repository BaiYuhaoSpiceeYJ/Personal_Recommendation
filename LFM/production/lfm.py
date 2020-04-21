import numpy as np
import read
import operator


#参数：训练样本，隐参数长度，正则化参数，学习率，模型迭代次数
def lfm_train(train_data, F, alpha, beta, step):
    """
    Args:
        train_data: train_data for lfm
        F: user vector len, item vector len
        alpha:regularization factor
        beta: learning rate
        step: iteration num
    Return:
        dict: key itemid, value:np.ndarray
        dict: key userid, value:np.ndarray
    """
    user_vec = {}
    item_vec = {}
    for step_index in range(step):
        print('step:',step_index)
        for data_instance in train_data:
            userid, itemid, label = data_instance
            if userid not in user_vec:
                user_vec[userid] = init_model(F)
            if itemid not in item_vec:
                item_vec[itemid] = init_model(F)
            delta = label - model_predict(user_vec[userid], item_vec[itemid])
            for index in range(F):
                user_vec[userid][index] += beta *(delta*item_vec[itemid][index] - alpha*user_vec[userid][index])
                item_vec[itemid][index] += beta*(delta*user_vec[userid][index] - alpha*item_vec[itemid][index])
        beta = beta * 0.9
    return user_vec, item_vec


def init_model(vector_len):
    """
    Args:
        vector_len: the len of vector
    Return:
         a ndarray
    """
    return np.random.randn(vector_len)


def model_predict(user_vector, item_vector):
    """
    user_vector and item_vector distance
    Args:
        user_vector: model produce user vector
        item_vector: model produce item vector
    Return:
         a num
    """
    res = np.dot(user_vector, item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    return res


#给用户推荐，输入为训练好的ve和userid，返回对应该用户得分最高的item的id以及得分
def give_recom_result(user_vec, item_vec, userid):
    """
    use lfm model result give fix userid recom result
    Args:
        user_vec: lfm model result
        item_vec:lfm model result
        userid:fix userid
    Return:
        a list:[(itemid, score), (itemid1, score1)]
    """
    fix_num = 10
    if userid not in user_vec:#如果用户id不在user_vec的记录中
        return []
    record = {}
    recom_list = []
    user_vector = user_vec[userid]
    for itemid in item_vec:#计算每一个item向量与该用户向量的欧氏距离
        item_vector = item_vec[itemid]
        res = np.dot(user_vector, item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
        record[itemid] = res
    for zuhe in sorted(record.items(), key= operator.itemgetter(1), reverse=True)[:fix_num]:#排序
        itemid = zuhe[0]
        score = round(zuhe[1], 3)
        recom_list.append((itemid, score))
    print(recom_list)
    return recom_list


#分析推荐结果好坏
def ana_recom_result(train_data, userid, recom_list):
    """
    debug recom result for userid
    Args:
        train_data: train data for lfm model
        userid:fix userid
        recom_list: recom result by lfm
    """
    item_info = read.get_item_info("../data/movies.txt")
    for data_instance in train_data:#分析用户对哪些item进行过喜欢的操作，并打印
        tmp_userid, itemid, label = data_instance
        if tmp_userid == userid and label == 1:
            print(item_info[itemid])
    print("recom result")
    for zuhe in recom_list:#打印推荐的电影id
        print(item_info[zuhe[0]])


def model_train_process():
    """
    test lfm model train
    """
    train_data=read.get_train_data("../data/ratings.txt")
    user_vec, item_vec = lfm_train(train_data, 50, 0.01, 0.1, 50)
    for userid in user_vec:
        recom_result = give_recom_result(user_vec, item_vec, userid)
        ana_recom_result(train_data, userid, recom_result)


if __name__ == "__main__":
    model_train_process()
