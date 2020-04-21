from __future__ import division

import reader
import math
import operator

def base_contribute_score():
    """
    item cf base sim contribution score by user
    """
    return 1


def update_one_contribute_score(user_total_click_num):
    """
    item cf update sim contribution score by user
    """
    return 1/math.log10(1+user_total_click_num)#对用户的总点击数惩罚


def update_two_contribute_score(click_time_one, click_time_two):
    """
    item cf update two sim contribution score by user
    """
    delata_time = abs(click_time_one - click_time_two)
    total_sec = 60 * 60 *24
    delata_time = delata_time/total_sec
    return 1/(1+delata_time)


#计算item  相似度矩阵
def cal_item_sim(user_click, user_click_time):
    """
    Args:
        user_click:dict ,key userid value [itemid1, itemid2]
    Return:
        dict, key:itemid_i, value dict, value_key itemid_j, value_value simscore
    """
    co_appear = {}#同时出现的次数矩阵
    item_user_click_time = {}#计算有多少用户点击了这个item
    for user, itemlist in user_click.items():
        for index_i in range(0, len(itemlist)):#统计这个用户点击了什么
            itemid_i = itemlist[index_i]
            item_user_click_time.setdefault(itemid_i, 0)
            item_user_click_time[itemid_i] += 1#物品的被点击总次数
            for index_j in range(index_i + 1, len(itemlist)):#统计这个用户还点击了什么，计算这两个之间的时间差
                itemid_j = itemlist[index_j]
                if user+ "_" + itemid_i not in user_click_time:
                    click_time_one = 0
                else:
                    click_time_one = user_click_time[user +"_" + itemid_i]
                if user+ "_" + itemid_j not in user_click_time:
                    click_time_two = 0
                else:
                    click_time_two = user_click_time[user +"_" + itemid_j]
                co_appear.setdefault(itemid_i, {})
                co_appear[itemid_i].setdefault(itemid_j, 0)
                #co_appear[itemid_i][itemid_j] += update_one_contribute_score(len(itemlist))
                co_appear[itemid_i][itemid_j] += update_two_contribute_score(click_time_one, click_time_two)

                co_appear.setdefault(itemid_j, {})
                co_appear[itemid_j].setdefault(itemid_i, 0)
                # co_appear[itemid_i][itemid_j] += update_one_contribute_score(len(itemlist))
                co_appear[itemid_j][itemid_i] += update_two_contribute_score(click_time_one, click_time_two)#同时出现，贡献+1/时间分数

    item_sim_score = {}#不同物品之间的相似度情况
    item_sim_score_sorted = {}

    for itemid_i, relate_item in co_appear.items():
        for itemid_j, co_time in relate_item.items():
              sim_score = co_time/math.sqrt(item_user_click_time[itemid_i]*item_user_click_time[itemid_j])
              item_sim_score.setdefault(itemid_i, {})
              item_sim_score[itemid_i].setdefault(itemid_j, 0)
              item_sim_score[itemid_i][itemid_j] = sim_score
    for itemid in item_sim_score:
        item_sim_score_sorted[itemid] = sorted(item_sim_score[itemid].items(), key = \
                                                operator.itemgetter(1), reverse=True)
    return item_sim_score_sorted


#为每个用户计算推荐结果
def cal_recom_result(sim_info, user_click):
    """
    recom by itemcf
    Args:
        sim_info: item sim dict
       user_click: user click dict
    Return:
        dict, key:userid value dict, value_key itemid , value_value recom_score
    """
    recent_click_num = 3#用用户最近点击过的3个item进行相似度推荐
    topk = 5
    recom_info = {}
    for user in user_click:
        click_list = user_click[user]#找到每一个用户的click list
        recom_info.setdefault(user, {})
        for itemid in click_list[:recent_click_num]:
            if itemid not in sim_info:
                continue
            for itemsimzuhe in sim_info[itemid][:topk]:#找到用户click过的最近3个i分别相似度最高的5个j
                 itemsimid = itemsimzuhe[0]
                 itemsimscore = itemsimzuhe[1]
                 recom_info[user][itemsimid] = itemsimscore#一共给用户推荐3*5个候选，默认用户3分以上的都为喜欢，行为得分都为1
    return recom_info


def debug_itemsim(item_info, sim_info):
    """
    show itemsim info
    Args:
        item_info: dict, key itemid value:[title, genres]
        sim_info: dict key itemid , value dict,  key [(itemid1, simscore), (itemdi2, simscore)]
    """
    fixed_itemid = "1"#打印和item1相似度最高的5个item
    if fixed_itemid not in item_info:
        print("invalid itemid")
        return
    [title_fix, genres_fix] = item_info[fixed_itemid]
    for zuhe in sim_info[fixed_itemid][:5]:
        itemid_sim = zuhe[0]
        sim_score = zuhe[1]
        if itemid_sim not in item_info:
            continue
        [title, genres] = item_info[itemid_sim]
        print(title_fix + "\t" + genres_fix + "\tsim:" + title + "\t" +  genres + "\t" + str(sim_score))


def debug_recomresult(recom_result, item_info):
    """
    debug recomresult
    Args:
        recom_result: key userid value:dict , value_key:itemid , value_value:recom_score
       item_info: dict, key itemid  value:[title, genre]

    """
    user_id = "1"
    if user_id not in recom_result:
        print("invalid result")
        return
    for zuhe in sorted(recom_result[user_id].items(), key = operator.itemgetter(1), reverse=True):
        itemid, score = zuhe
        if itemid not in item_info:
            continue
        print(",".join(item_info[itemid]) + "\t" + str(score))

def main_flow():
    """
    main flow of itemcf
    """
    user_click, user_click_time = reader.get_user_click("../data/ratings.txt")#得到用户的点击序列
    item_info = reader.get_item_info("../data/movies.txt")
    sim_info = cal_item_sim(user_click, user_click_time)#根据点击序列得到item的相似度
    debug_itemsim(item_info, sim_info)#计算推荐结果
    recom_result = cal_recom_result(sim_info, user_click)
    print(recom_result["1"])
    debug_recomresult(recom_result, item_info)

if __name__ == "__main__":
    main_flow()
