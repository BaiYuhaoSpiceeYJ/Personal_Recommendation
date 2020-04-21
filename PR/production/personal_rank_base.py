import read
import operator


#二分图，固定定点，alpha概率，迭代次数，给固定定点推荐的数目
#输出词典，key为item，value是pr值，长度是推荐数目
def personal_rank(graph, root, alpha, iter_num, recom_num= 10):
    """
    Args
        graph: user item graph 
        root: the  fixed user for which to recom 
        alpha: the prob to go to random walk 
        iter_num:iteration num 
        recom_num: recom item num 
    Return:
        a dict, key itemid, value pr
    """

    rank = {point:0 for point in graph}#除root定点，其余的顶点初始pr值都为0
    rank[root] = 1
    recom_result = {}
    for iter_index in range(iter_num):
        print(iter_index)
        tmp_rank = {point:0 for point in graph}#存储其余顶点对该顶点的pr值
        for out_point, out_dict in graph.items():
                for inner_point, value in graph[out_point].items():
                    tmp_rank[inner_point] += round(alpha*rank[out_point]/len(out_dict), 4)
                    if inner_point == root:
                        tmp_rank[inner_point] += round(1-alpha, 4)
        if tmp_rank == rank:#如果收敛，提前结束
            print("out" + str(iter_index))
            break
        rank = tmp_rank#如果没收敛，继续迭代
    right_num = 0
    #让rank数据结构按照pr值进行排序，并过滤掉user顶点和root顶点已经行为过的item
    for zuhe in sorted(rank.items(), key = operator.itemgetter(1), reverse=True):
        point, pr_score = zuhe[0], zuhe[1]
        if len(point.split('_')) < 2:#如果不是item顶点，过滤掉
            continue
        if point in graph[root]:#如果该顶点在root行为里，过滤掉
            continue
        recom_result[point] = round(pr_score,4)#输出数据结构
        right_num += 1
        if right_num > recom_num:#如果推荐的数目达到要求的数目，就可以提前终止并返回
            break
    return recom_result


def get_one_user_recom():
    """
    give one fix_user recom result
    """
    user = "1"
    alpha = 0.6
    graph = read.get_graph_from_data("../data/ratings.txt")
    iter_num = 100
    recom_result = personal_rank(graph, user, alpha, iter_num, 100)
    #print(recom_result)
    return recom_result

    #打印推荐的id的电影信息
    '''
    item_info = read.get_item_info("../data/movies.txt")
    for itemid in graph[user]:#打印用户感兴趣的item
        pure_itemid = itemid.split("_")[1]
        print(item_info[pure_itemid])
    print("result---")
    for itemid in recom_result:#打印推荐的item
        pure_itemid = itemid.split("_")[1]
        print(item_info[pure_itemid])
        print(recom_result[itemid])    '''


if __name__ == "__main__":
    recom_result_base = get_one_user_recom()



