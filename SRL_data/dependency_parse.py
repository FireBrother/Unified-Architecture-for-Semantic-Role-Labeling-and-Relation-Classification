import pyltp
import os
LTP_DATA_DIR = '/Users/menrui/PycharmProjects/ltp_data'  # ltp模型目录的路径
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
from pyltp import Parser
from pyltp import Postagger
import json
parser = Parser() # 初始化实例
parser.load(par_model_path)  # 加载模型
posttagger = Postagger()
posttagger.load(pos_model_path)
depend_dict = {}
file_list = ['cpbdev','cpbtest','cpbtrain']
for file_name in file_list:
    output = open('data/'+file_name+'_tree.txt','w',encoding='utf-8')
    print(file_name)
    f = open('data/'+file_name+'.txt','r',encoding='utf-8')

    for line in f.readlines():
        tokens = line.strip().split()
        word_seq = []
        pos_seq = []
        for i, token in enumerate(tokens):
            v = token.split('/')
            word_seq.append(v[0])
            pos_seq.append(v[1])
        pos_seq = posttagger.postag(word_seq)
        arcs = parser.parse(word_seq,pos_seq)
        if file_name == 'cpbtrain':
            for arc in arcs:
                if not arc.relation in depend_dict:
                    depend_dict[arc.relation] = 0
                depend_dict[arc.relation]+=1

        #print ("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        output.write("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)+'\n')
    output.close()
    f.close()
json.dump(depend_dict,open('data/depend_dict.json','w'))
parser.release()