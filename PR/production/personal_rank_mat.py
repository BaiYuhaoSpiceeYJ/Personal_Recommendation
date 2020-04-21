import read
import operator
import mat_util
from scipy.sparse.linalg import gmres
import numpy as np
#import personal_rank_base
from personal_rank_base import get_one_user_recom

#二分图，固定定点，alpha概率，给固定定点推荐的数目
#输出词典，key为item，value是pr值，长度是推荐数目
def personal_rank_mat(graph, root, alpha, recom_num = 10):
    """
    Args:
        graph:user item graph
        root:the fix user to recom
        alpha:the prob to random walk
        recom_num:recom item num
    Return:
        a dict, key: itemid, value: pr score
    A*r = r0
    """
    m, vertex, address_dict = mat_util.graph_to_m(graph)
    if root not in address_dict:
        return {}
    score_dict = {}
    recom_dict = {}
    mat_all = mat_util.mat_all_point(m, vertex, alpha)#A矩阵
    index = address_dict[root]
    initial_list = [[0] for row in range(len(vertex))]#r0矩阵，为一个列向量，除了根节点为1，其余均为0
    initial_list[index] = [1]
    r_zero = np.array(initial_list)
    res = gmres(mat_all, r_zero, tol=1e-8)[0]#tol为误差，gmres用来解Ax=b，即Ar=r0
    for index in range(len(res)):
        point = vertex[index]
        if len(point.strip().split("_")) < 2:#过滤掉user顶点
            continue
        if point in graph[root]:#过滤掉根借点user行为过的item
            continue
        score_dict[point] = round(res[index], 3)
    for zuhe in sorted(score_dict.items(), key = operator.itemgetter(1), reverse= True)[:recom_num]:
        point, score = zuhe[0], zuhe[1]
        recom_dict[point] = score
    return recom_dict


def get_one_user_by_mat():
    """
    give one fix user by mat
    """
    user = "1"
    alpha = 0.8
    graph = read.get_graph_from_data("../data/ratings.txt")
    recom_result = personal_rank_mat(graph, user, alpha, 100)
    print(recom_result)
    return recom_result


if __name__ == "__main__":
    recom_result_mat = get_one_user_by_mat()
    recom_result_base = get_one_user_recom()
    num = 0
    for ele in recom_result_base:
        num += ele in recom_result_mat

    print(num)


