import os

#得到item的详情，电影名以及标签。对应文件movies.txt,输入文件，输出为电影名以及分类的词典
def get_item_info(input_file):
    """
    get item info:[title, genre]
    Args:
        input_file:item info file
    Return:
        a dict: key itemid, value:[title, genre]
    """
    if not os.path.exists(input_file):
        return {}
    item_info = {}
    linenum = 0
    fp = open(input_file)
    for line in fp:#跳过movies.txt的第一行
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 3:#正常情况下有两个逗号，被分成三份，有些电影名中含有逗号，为特殊情况，大于3，如果小于3则说明数据有错
            continue
        elif len(item) == 3:
            itemid, title, genre = item[0], item[1], item[2]
        elif len(item) > 3:
            itemid = item[0]
            genre = item[-1]
            title = ",".join(item[1:-1])
        item_info[itemid] = [title, genre]
    fp.close()
    return item_info


#获得item的平均评分，对应ratings.txt文件
def get_ave_score(input_file):
    """
    get item ave rating score
    Args:
        input file: user rating file
    Return:
        a dict, key:itemid, value:ave_score
    """
    if not os.path.exists(input_file):
        return {}
    linenum = 0
    record_dict = {}
    score_dict = {}
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], float(item[2])
        if itemid not in record_dict:
            record_dict[itemid] = [0, 0]#初始化
        record_dict[itemid][0] += 1#item被多少人点评过
        record_dict[itemid][1] += rating#多少分
    fp.close()
    for itemid in record_dict:
        score_dict[itemid] = round(record_dict[itemid][1]/record_dict[itemid][0], 3)#四舍五入保留三位有效数字
    return score_dict


#为lfm模型提供训练样本，输出为list，
def get_train_data(input_file):
    """
    get train data for LFM model train
    Args:
        input_file: user item rating file
    Return:
        a list:[(userid, itemid, label), (userid1, itemid1, label)]
    """
    if not os.path.exists(input_file):
        return []
    score_dict = get_ave_score(input_file)
    neg_dict = {}#<4
    pos_dict = {}#>4
    train_data = []
    linenum = 0
    score_thr = 4.0
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], float(item[2])
        if userid not in pos_dict:
            pos_dict[userid] = []
        if userid not in neg_dict:
            neg_dict[userid] = []
        if rating >= score_thr:
            pos_dict[userid].append((itemid, 1))
        else:
            score = score_dict.get(itemid, 0)#如果这个电影没有被任何人评论过，默认0分
            neg_dict[userid].append((itemid, score))
    fp.close()
    for userid in pos_dict:
        data_num = min(len(pos_dict[userid]), len(neg_dict.get(userid, [])))#为了做正负样本数量均衡
        if data_num > 0:
            train_data += [(userid, zuhe[0], zuhe[1]) for zuhe in pos_dict[userid]][:data_num]
        else:
            continue
        sorted_neg_list = sorted(neg_dict[userid], key=lambda element:element[1], reverse=True)[:data_num]#对负样本按照平均得分排序
        train_data += [(userid, zuhe[0], 0) for zuhe in sorted_neg_list]#即将这个人不喜欢且大家都不喜欢的电影写入到这个人的负样本里
    return train_data





