import random, math

class Node():
    def __init__(self, chromosome, fitness):
        self.fitness = fitness
        self.chromosome = chromosome
        self.parent = None
        self.children = []
class TernaryTreePopulation():
    def __init__(self, max_capacity = 13):
        self.root = None
        self.max_capacity = max_capacity
        self._max_levels = math.log((self.max_capacity*2) + 1, 3)
    
    def get_all_nodes(self):
        if not self.root: return []
        nodes, queue = [], [self.root]
        while queue:
            curr = queue.pop(0)
            nodes.append(curr)
            queue.extend(curr.children)
        return nodes

    def insert_node(self, node):
        if self.root == None:
            self.root = node
            node.parent = None
            return True

        current = [self.root]
        current_level = 0

        inserted = False
        while current_level < self._max_levels:
            for nd in current:
                if nd.fitness > node.fitness:
                    node.chromosome, nd.chromosome =  nd.chromosome, node.chromosome
                    node.fitness, nd.fitness =  nd.fitness, node.fitness

            new = []
            for nd in current:
                if not inserted and len(nd.children) < 3:
                    inserted = True
                    nd.children.append(node)
                    node.parent = nd
                new = new + nd.children

            self._bubble_sort(new)
            if inserted: break

            current_level+=1
            current = new

        return inserted   
                    
    def _bubble_sort(self, list):
        sorted = False
        while(not sorted):
            sorted = True
            for i in range(0, len(list)-1):
                if list[i].fitness > list[i+1].fitness:
                    sorted = False
                    list[i].chromosome, list[i+1].chromosome =  list[i+1].chromosome, list[i].chromosome
                    list[i].fitness, list[i+1].fitness =  list[i+1].fitness, list[i].fitness
    
    def print_tree(self, raiz, nivel = 0, prefixo="Raiz: "):
        if raiz is not None:
            if len(raiz.children) > 0:
                self.print_tree(raiz.children[0], nivel + 1, "┌── (D) ")
            if len(raiz.children) > 1:   
                self.print_tree(raiz.children[1], nivel + 1, "├── (M) ")
            
            espacos = "│   " * nivel if nivel > 0 else ""
            if nivel > 0:
                espacos = espacos[:-4] + "│   "
            print(f"{'    ' * nivel}{prefixo}{raiz.fitness}")
            if len(raiz.children) == 3:
                self.print_tree(raiz.children[2], nivel + 1, "└── (E) ")

    def best_nodes(self, n_best):
        nodes = self.get_all_nodes()
        nodes = [n for n in nodes if n.fitness is not None]
        return sorted(nodes, key=lambda n: n.fitness)[:n_best]

    def random_leader_supporter(self):
        leaders = [self.root] + self.root.children
        idx = random.randint(0, len(leaders)-1)
        leader = leaders[idx]
        idx = random.randint(0, len(leader.children)-1)
        return leader, leader.children[idx]

if __name__ == "__main__":
    tree = TernaryTreePopulation()
    fitness = [27,32,34,35,39,42,43,50,51,53,59,64,67]
    # fitness = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    fitness.reverse()
    #fitness = [(0,2), (1,2), (0,1), (1,1)]
    for fit in fitness:
        node = Node([], fit)
        tree.insert_node(node)
        nodes = tree.get_all_nodes()
        for n in nodes:
            print(n.fitness, end= " ")
        print()    