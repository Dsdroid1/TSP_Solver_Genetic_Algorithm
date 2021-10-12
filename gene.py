"""
Class to store the details about the path in TSP genetic algorithm
"""
import random

class Gene:

    # Constructor
    def __init__(self, nodeList, graph):
        self.path = nodeList.copy()
        # This is an indicator to check if the given ordering of nodes produces a valid tour
        self.valid = self.valid_tour(graph)
        # Get the fitness here
        self.fitness = self.fitness_fn(graph)

    def fitness_fn(self, graph):
        # We have to also check if the total path exists or not
        local_path = self.path.copy()
        num_of_nodes = len(self.path)
        prev = local_path.pop(0)
        cost = 0
        invalid = False
        for node in local_path:
            # Add the cost of prev-to-node to our cost
            if graph[prev].get(node) is not None:
                cost += graph[prev][node]
                prev = node
            else:
                # Since this is a path that cannot really be completed, we have to assign it a minimum fitness
                invalid = True
                break

        # Now check the tour completion condition
        if invalid == False and graph[self.path[num_of_nodes-1]].get(self.path[0]) is not None:
            cost += graph[self.path[num_of_nodes-1]][self.path[0]]
        else:
            invalid = True
            # Not a valid tour, do something
        if invalid:
            # Assign the fitness a minimum value
            max_edge_cost = 0
            for node1 in graph.keys():
                for node2 in graph[node1].keys():
                    max_edge_cost = max(graph[node1][node2],max_edge_cost)
            # Cost greater than any possible path cost, to denote a bad state
            cost = max_edge_cost * (len(graph.keys())+1)
        # print(cost)
        return 1/cost

    def valid_tour(self, graph):
        # Check of the given path is actually possible in the graph
        local_path = self.path.copy()
        prev = local_path.pop(0)
        for node in local_path:
            if graph[prev].get(node) is None:
                return False
            prev = node

        if graph[self.path[-1]].get(self.path[0]) is None:
            return False
        return True

    def mutate(self, graph):
        # Exchange 2 random positions in the path
        num_of_nodes = len(graph.keys())

        position1 = random.randint(0, num_of_nodes-1)
        position2 = random.randint(0, num_of_nodes-1)

        # Swap these positions
        temp = self.path[position1]
        self.path[position1] = self.path[position2]
        self.path[position2] = temp

        self.fitness = self.fitness_fn(graph)
        self.valid = self.valid_tour(graph)

    def display(self):
        print('Gene Info:')
        print(f'Path taken:{self.path}')
        print(f'Fitness value: {self.fitness}')
        print(f'Complete tour possible: {self.valid}')

    def get_path(self):
        return self.path

    def get_fitness(self):
        return self.fitness

    def is_valid(self):
        return self.valid