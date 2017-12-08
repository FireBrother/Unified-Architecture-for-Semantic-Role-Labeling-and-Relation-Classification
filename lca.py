class Tree():
    def __init__(self,node_list,rel_index):
        self.root = {'index':0,'father':-1,'type':'ROOT','children':[]}
        self.nodes = [self.root]
        node_list = node_list.split('\t')
        for index,node_info in enumerate(node_list):
            node_info = node_info.split(':')
            node = {'index':index,'father':int(node_info[0]),'type':node_info[1],'children':[]}
            self.nodes.append(node)
        for index,node in enumerate(self.nodes):
            if node['father']>=0:
                self.nodes[node['father']]['children'].append(index)
        self.rel_index = rel_index
        self.size = len(self.nodes)
        self.rel_token_path = []
        self.rel_depend_path = []
        tmp = rel_index
        while tmp>=0 :
            self.rel_token_path.append(tmp)
            self.rel_depend_path.append(self.nodes[tmp]['type'])
            tmp = self.nodes[tmp]['father']

    def cal_lca(self,node_index):
        tmp = node_index
        token_path = [tmp]
        depend_path = [self.nodes[tmp]['type']]
        while not tmp in self.rel_token_path:
            tmp = self.nodes[tmp]['father']
            token_path.append(tmp)
            depend_path.append(self.nodes[tmp]['type'])
        rel_path_index = self.rel_token_path.index(tmp)
        return [token_path,depend_path,self.rel_token_path[:rel_path_index+1],self.rel_depend_path[:rel_path_index+1]]


if __name__=='__main__':
    tree = Tree('3:ATT	3:ATT	4:SBV	0:HED	6:ADV	4:VOB',6)
    for i in range(tree.size):
        if not i == 4:
            print(tree.cal_lca(i))





