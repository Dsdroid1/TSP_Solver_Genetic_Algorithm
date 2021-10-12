"""
AI Assignment 3 by BT18CSE046
TSP Solver using genetic algorithm
"""
import numpy as np
import math
import random
from gene import Gene

# Function to read the graph as an input file
def read_input(filename):
    """
    The format of the input file should be:
    (i) The first line should contain the number of nodes in the graph(will be numbered from 1 to n)
    (ii) The subsequent lines contain info about the edges of the graph, in the format:
        i j cost or i,j,cost if the file extension is csv
        where i and j represent the nodes involved in the edge and cost represents the path coset for this edge
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        no_of_nodes = int(lines.pop(0).strip())
        
        # Create the adjacency list using dictionary
        graph = {}
        for i in range(1,no_of_nodes+1):
            graph[i] = {}

        separator = " "
        dot_pose = filename.find('.')
        if dot_pose != -1:
            file_extension = filename[dot_pose+1:]
            if file_extension == 'csv':
                separator = ','

        for line in lines:
            i,j,cost = list(map(int,line.strip().split(separator)))
            if i!=j:
                # Assuming undirected graph
                graph[i][j] = cost
                graph[j][i] = cost
        
        return graph 


"""
Our genetic representation of the state will be an ordered list of states to visit.
The fitness function can be evaluated as inversely proportional to the tour cost of the given path.
"""

def breed(gene1, gene2, graph):
    # We pick some nodes from gene1 and some from gene2 randomly, preserving some order
    nodes = list(graph.keys())
    crossover_point = random.randint(0, len(nodes)-1)
    new_list1 = []
    # Get all nodes till the crossover point from gene1
    new_list1 = gene1.get_path()[:crossover_point].copy()
    # print(crossover_point)
    new_list2 = []
    # Get the remaining nodes from gene2, preserving their order from gene2
    visited1 = set(new_list1)
    # Second child will be the those nodes from gene2 which were picked from gene1 in child 1
    visited2 = set(nodes)-visited1
    for node in gene2.get_path():
        if node not in visited1:
            visited1.add(node)
            new_list1.append(node)
        if node not in visited2:
            visited2.add(node)
            new_list2.append(node)
    new_list2.extend(gene1.get_path()[crossover_point:].copy())

    # Generate a gene for this new state
    child1 = Gene(new_list1,graph)
    child2 = Gene(new_list2,graph)
    # Mutate the child gene if probability of mutation is reached
    mutation_prob = 0.35
    chance1 = random.uniform(0,1)
    if chance1 < mutation_prob:
        # print("Mutated child 1")
        child1.mutate(graph)
    chance2 = random.uniform(0,1)
    if chance2 < mutation_prob:
        # print("Mutated child 2")
        child2.mutate(graph)
    return child1,child2

def generate_population(graph):
    # Generate the initial permutations for the inital population
    population_size = 5
    nodes = list(graph.keys())
    population = []
    print("Generating population...")
    for i in range(population_size):
        population.append(Gene(list(np.random.permutation(nodes)),graph))
    print("Done!")
    return population

def choose_parents(population,graph):
    # Choose parenst based on fitness values
    weights = [gene.get_fitness() for gene in population]
    parents = random.choices(population, weights=weights,k=2) # 2 parents
    return parents

def make_new_generation(previous_generation, graph):
    next_generation = []
    population_count = len(previous_generation)
    for i in range(math.ceil(population_count/2)):
        parents = choose_parents(previous_generation,graph)
        child1, child2 = breed(parents[0],parents[1],graph)
        next_generation.append(child1)
        next_generation.append(child2)
    return next_generation[:population_count]

def genetic_algorithm(graph):
    best_gene_yet = None # Keep track of the best solution generated
    current_population = generate_population(graph)
    iters = 100 # Fix the number of generations to be generated
    for i in range(iters):
        # Get the best gene from the current population
        for gene in current_population:
            if best_gene_yet is not None and best_gene_yet.get_fitness() < gene.get_fitness():
                best_gene_yet = gene
            elif best_gene_yet == None:
                best_gene_yet = gene
        # Generate the next population
        next_gen = make_new_generation(current_population,graph)
        current_population = next_gen
    
    # Get the best gene from the current population
    for gene in current_population:
        if best_gene_yet is not None and best_gene_yet.get_fitness() < gene.get_fitness():
            best_gene_yet = gene
        elif best_gene_yet == None:
            best_gene_yet = gene

    if best_gene_yet.is_valid() == True:
        # Solution has been found
        # best_gene_yet.display()
        print('Solution found')
    else:
        # No solution found uptil now, print that a restart should be done
        best_gene_yet = None
        print('No solution found, please try to run the program again[hope for the best]')
    return best_gene_yet

if __name__ == "__main__":
    # graph  = read_input('test.txt')
    graph = read_input('test1.csv')
    solution = genetic_algorithm(graph)
    if solution is not None:
        # Print the path
        path_list = solution.get_path()
        path_string = f'{path_list[0]}'
        for node in path_list[1:]:
            path_string += f'-{node}'
        if len(graph.keys()) > 1:
            path_string += f'-{path_list[0]}'
        print(f'Best Path Found:{path_string}')
        print(f'Cost: {1/solution.get_fitness()}')
    # population = generate_population(graph)
    # for gene in population:
    #     gene.display()
    # print("")
    # next_gen = make_new_generation(population,graph)
    # for gene in next_gen:
    #     gene.display()
    # parents = choose_parents(population,graph)
    # for parent in parents:
    #     parent.display()
    # parents = choose_parents(population,graph)
    # for parent in parents:
    #     parent.display()
    # for gene in population:
    #     gene.display()
    #     print('')
    # gene1 = Gene([2,3,4,5,1],graph=graph)
    # gene1.display()
    # gene2 = Gene([1,2,5,3,4],graph=graph)
    # gene2.display()
    # child1,child2 = breed(gene1,gene2,graph=graph)
    # child1.display()
    # child2.display()
