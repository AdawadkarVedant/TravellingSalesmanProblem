#import required libraries
import numpy as np
import random as rd
from bs4 import BeautifulSoup as bs
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# Region 'Mutation Operators'
def doSingleGeneSwapMutation(parent_chromosomes, cost_matrix, matrix_length):
    children = np.zeros((2, matrix_length + 1))
    gene_index_1, gene_index_2 = sorted(rd.sample(range(1, matrix_length), 2))
    for index in range(len(children)):
        updated_parent_chromosome = parent_chromosomes[index][:-1]
        updated_parent_chromosome[gene_index_1], updated_parent_chromosome[gene_index_2] = updated_parent_chromosome[
            gene_index_2], updated_parent_chromosome[gene_index_1]
        child = calculateFitness(updated_parent_chromosome, cost_matrix)
        children[index] = child
    return children


def doInversionSwapMutation(parent_chromosomes, cost_matrix, matrix_length):
    start_pos, end_pos = sorted(rd.sample(range(0, matrix_length), 2))
    children = np.zeros((2, matrix_length + 1))
    for index in range(len(children)):
        updated_parent_chromosome = parent_chromosomes[index][:-1]
        updated_parent_chromosome[start_pos:end_pos + 1] = updated_parent_chromosome[start_pos:end_pos + 1][::-1]
        child = calculateFitness(updated_parent_chromosome, cost_matrix)
        children[index] = child
    return children
# End Region

# Region 'Crossover Operators'
def doPMXCrossover(parent_chromosomes, cost_matrix, matrix_length):
    first_crossover_point, second_crossover_point = sorted(rd.sample(range(0, matrix_length), 2))
    crossover_children = np.zeros((2, matrix_length + 1))
    parent_1 = parent_chromosomes[0][:-1]
    parent_2 = parent_chromosomes[1][:-1]
    crossover_child_1 = []
    count = 0
    for child in parent_1:
        if count == first_crossover_point:
            break
        if child not in parent_2[first_crossover_point:second_crossover_point]:
            crossover_child_1.append(child)

    crossover_child_1.extend(
        parent_2[first_crossover_point:second_crossover_point]
    )
    crossover_child_1.extend([x for x in parent_1 if x not in crossover_child_1])

    count = 0
    crossover_child_2 = []
    for child in parent_2:
        if count == second_crossover_point:
            break
        if child not in parent_1[first_crossover_point:second_crossover_point]:
            crossover_child_2.append(child)

    crossover_child_2.extend(
        parent_1[first_crossover_point:second_crossover_point]
    )
    crossover_child_2.extend([x for x in parent_2 if x not in crossover_child_2])
    updated_crossover_child_1 = calculateFitness(np.array(crossover_child_1), cost_matrix)
    updated_crossover_child_2 = calculateFitness(np.array(crossover_child_2), cost_matrix)
    crossover_children[0] = np.array(updated_crossover_child_1)
    crossover_children[1] = np.array(updated_crossover_child_2)
    return crossover_children
# End Region


# Region 'Replacement Methods'
def doPopulationReplacement(operator_children, population_fitness_matrix):
    for operator_child in operator_children:
        worst_fit_chromosome, worst_fit_chromosome_index = getWorstFitChromosome(population_fitness_matrix)
        if worst_fit_chromosome[-1] >= operator_child[-1]:
            population_fitness_matrix[worst_fit_chromosome_index] = operator_child
# End Region

# Region 'Calculate Fitness Methods'
def calculateFitness(single_chromosome, cost_matrix):
    chromosome_pairs = [(single_chromosome[index], single_chromosome[(index + 1) % len(single_chromosome)]) for index in
                        range(len(single_chromosome))]
    updated_chromosome = np.append(single_chromosome,
                                   float(sum(cost_matrix[int(cp[0]), int(cp[1])] for cp in chromosome_pairs)))
    return updated_chromosome

def getBestFitChromosome(population_fitness_matrix):
    best_fit_value = np.min(population_fitness_matrix[:, -1])
    best_fit_row_index = np.where(best_fit_value == population_fitness_matrix)[0][0]
    best_fit_row = population_fitness_matrix[best_fit_row_index]
    return best_fit_row, best_fit_row_index

def getWorstFitChromosome(population_fitness_matrix):
    worst_fit_value = np.max(population_fitness_matrix[:, -1])
    worst_fit_row_index = np.where(worst_fit_value == population_fitness_matrix)[0][0]
    worst_fit_row = population_fitness_matrix[worst_fit_row_index]
    return worst_fit_row, worst_fit_row_index
# End Region

# Regions 'Other Methods'
def callXMLParser(filename):
    with open(filename, "r") as file:
        xml_file = file.read()
        xml_file_data = bs(xml_file, "xml")
    return xml_file_data

def generateCostMatrix(xmlData):
    vertex_length = int(len(xmlData.find_all('vertex')))
    edge_costs_arr = np.zeros((vertex_length, vertex_length))
    vertex_index = 0
    for vertex in xmlData.find_all('vertex'):
        edges = vertex.find_all('edge')
        for edge in edges:
            edge_index = int(edge.text)
            cost = float(edge.get('cost'))
            edge_costs_arr[vertex_index][edge_index] = cost
        vertex_index += 1
    return vertex_length, edge_costs_arr

def generatePopulationFitness(cost_matrix, matrix_length, no_population):
    rand_population = np.zeros((no_population, matrix_length + 1))
    for num in range(no_population):
        chromosome = np.arange(matrix_length)
        np.random.shuffle(chromosome)
        chromosome = calculateFitness(chromosome, cost_matrix)
        rand_population[num] = chromosome
    return rand_population

def generateParents(tournament_size, population_fitness_matrix, matrix_length):
    parents = np.zeros((2, matrix_length + 1))
    for i in range(2):
        random_indices = np.random.choice(len(population_fitness_matrix) - 1, size=tournament_size, replace=False)
        tournament_players = population_fitness_matrix[random_indices]
        best_fit_chromosome, _ = getBestFitChromosome(tournament_players)
        parents[i] = best_fit_chromosome
    return parents
# End Region

# Region 'Genetic Algorithm'
# Set numpy print options so as not to show exponential
np.set_printoptions(suppress=True)

# Store file names in a list and call the parser
file_names = ['burma14.xml', 'brazil58.xml']

# Declare different kinds of parameters used in the experiments
population_size = [50, 100, 150, 200]
tournament_size = [5, 10, 15, 20]
operator_Parameters = {1: 'Single Gene Swap Mutation', 2: 'Inversion Mutation', 3: 'PMX Crossover',
                       4: 'PMX Crossover With Single Gene Swap Mutation', 5: 'PMX Crossover With Inversion Mutation'}

# Set Experiment dictionary where every one parameter is changed throughout the experiment
experiment_params = {}
experiment_count = 1
for pop_size in population_size:
    exp_pop_size = pop_size
    for tour_size in tournament_size:
        exp_tour_size = tour_size
        for key, values in operator_Parameters.items():
            exp_operator = {key: values}
            experiment_params[experiment_count] = {'experiment_pop_size': exp_pop_size,
                                                   'experiment_tournament_size': exp_tour_size,
                                                   'experiment_operator': exp_operator}
            experiment_count += 1

# Time to run the experiments for each file!
for file_name in file_names:

    # Parse the XML File
    xml_data = callXMLParser(file_name)

    # Generate cost matrix
    data_length, data_matrix = generateCostMatrix(xml_data)

    # Best fit, Worst fit, Execution time dataframes that will be used to create a csv file and plot the graphs.
    fitness_scores_df = pd.DataFrame(
        columns=['Trial', 'Operator Name', 'Population Size', 'Tournament Size', 'Best Fitness', 'Worst Fitness',
                 'Execution Time'])

    fitness_scores_list = []
    iteration_list = []

    # Create a directory if it does not exist
    dir_name = file_name.replace('.xml', '')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(dir_name + '\\' + 'Iterations'):
        os.makedirs(dir_name + '\\' + 'Iterations')

    # Iterate over each experiment with its set parameters
    for key, values in experiment_params.items():
        experiment_number = key
        selected_population_size = values.get('experiment_pop_size')
        selected_tournament_size = values.get('experiment_tournament_size')
        selected_operator = list(values.get('experiment_operator').items())

        # Run 10 trials for each experiment, each time with new population being generated
        for trial_num in range(10):
            start_time = time.time()
            # Generate population sample matrix
            population_matrix = generatePopulationFitness(data_matrix, data_length, selected_population_size)

            if trial_num == 0:
                # Get chromosome with the best fitness and the worst fitness from the generated population
                best_fit_chromosome, best_fit_chromosome_index = getBestFitChromosome(population_matrix)
                worst_fit_chromosome, worst_fit_chromosome_index = getWorstFitChromosome(population_matrix)

            # Set iteration counter for running 10,000 iterations
            # Run through the   iterations
            for iter_count in range(1, 10000):

                # Create an empty numpy array as a placeholder
                children = np.zeros((2, data_length + 1))

                # Based on the set tournament_size generate tournament which will find the winners.
                # The winners will be set as parents which will be passed to the operator functions
                parents = generateParents(selected_tournament_size, population_matrix, data_length)

                # Identify which operator is selected in the experiment parameter and call the operator method to perform the operations
                operator_to_run = selected_operator[0][0]

                if operator_to_run == 1:
                    # Single Gene Swap Mutation
                    children = doSingleGeneSwapMutation(parents, data_matrix, data_length)
                elif operator_to_run == 2:
                    # Inversion Mutation
                    children = doInversionSwapMutation(parents, data_matrix, data_length)
                elif operator_to_run == 3:
                    # PMX Crossover
                    children = doPMXCrossover(parents, data_matrix, data_length)
                elif operator_to_run == 4:
                    # PMX Crossover With Single Gene Swap Mutation
                    crossover_children = doPMXCrossover(parents, data_matrix, data_length)
                    children = doSingleGeneSwapMutation(crossover_children, data_matrix, data_length)
                elif operator_to_run == 5:
                    # PMX Crossover With Inversion Mutation
                    crossover_children = doPMXCrossover(parents, data_matrix, data_length)
                    children = doInversionSwapMutation(crossover_children, data_matrix, data_length)
                doPopulationReplacement(children, population_matrix)

                # Get chromosome with the best fitness from the generated population
                best_fit_chromosome, best_fit_chromosome_index = getBestFitChromosome(population_matrix)
                worst_fit_chromosome, worst_fit_chromosome_index = getBestFitChromosome(population_matrix)
                iteration_list.append([file_name.replace('.xml', ''), trial_num, selected_operator[0][1], iter_count,
                                       selected_population_size, selected_tournament_size, best_fit_chromosome[-1],
                                       worst_fit_chromosome[-1]])
            trial_num += 1

            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_time_min = elapsed_time / 60

            # Get chromosome with the best fitness and the worst fitness from the generated population
            best_fit_chromosome, best_fit_chromosome_index = getBestFitChromosome(population_matrix)
            worst_fit_chromosome, worst_fit_chromosome_index = getWorstFitChromosome(population_matrix)

            # Store results in a list for future references
            fitness_scores_list.append(
                [trial_num, selected_operator[0][1], selected_population_size, selected_tournament_size,
                 best_fit_chromosome[-1], worst_fit_chromosome[-1], elapsed_time_min])

    # Convert the lists into a dataframe and store the dataframe in a csv file
    fitness_scores_df = pd.DataFrame(fitness_scores_list,
                                     columns=['Trial', 'Operator Name', 'Population Size', 'Tournament Size',
                                              'Best Fitness', 'Worst Fitness', 'Execution Time'])

    if os.path.exists(dir_name):
        # Store the dataframes to CSV files
        fitness_scores_df.to_csv(dir_name + '\\' + 'Fitness_Score.csv', index=False)

        iteration_df = pd.DataFrame(iteration_list,
                                    columns=['File Name', 'Trial', 'Experiment', 'Iteration Count', 'Population Size',
                                             'Tournament Size', 'Best Fitness', 'Worst Fitness'])
        iteration_df.to_csv(dir_name + '\\Iterations\\' + 'Iteration_Fitness.csv', index=False)
# End Region

# Region 'Visualisation'
country_lists = ['burma14', 'brazil58']
for country in country_lists:
    fitness_dataframe = pd.read_csv(f'{country}\\Fitness_Score.csv')
    # Find the combination with the lowest 'Best Fitness' for each set of parameters
    best_fitness_combinations = fitness_dataframe.groupby(['Operator Name', 'Population Size', 'Tournament Size'])[
        'Best Fitness'].min().reset_index()

    # Pivot the DataFrame to prepare for the heatmap
    pivot_df = best_fitness_combinations.pivot(index='Operator Name', columns=['Population Size', 'Tournament Size'],
                                               values='Best Fitness')

    # Create a heatmap using Seaborn
    plt.figure(figsize=(16, 10))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='g', cbar=True)
    plt.title(
        f'{country}: Best Fitness Values for Different Combinations of Operator, Population Size, and Tournament Size')
    plt.xlabel('Population Size - Tournament Size')
    plt.ylabel('Operator Name')
    plt.tick_params(axis='y')
    # plt.tight_layout()
    plt.savefig(f'{country}\\{country}_best_fitness.png', bbox_inches='tight')

    # Find the combination with the lowest 'Best Fitness' for each set of parameters
    worst_fitness_combinations = fitness_dataframe.groupby(['Operator Name', 'Population Size', 'Tournament Size'])[
        'Worst Fitness'].max().reset_index()

    # Pivot the DataFrame to prepare for the heatmap
    pivot_df = worst_fitness_combinations.pivot(index='Operator Name', columns=['Population Size', 'Tournament Size'],
                                                values='Worst Fitness')

    # Create a heatmap using Seaborn
    plt.figure(figsize=(16, 10))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='g', cbar=True)
    plt.title(
        f'{country}: Worst Fitness Values for Different Combinations of Operator, Population Size, and Tournament Size')
    plt.xlabel('Population Size - Tournament Size')
    plt.ylabel('Operator Name')
    plt.tick_params(axis='y')
    plt.savefig(f'{country}\\{country}_worst_fitness.png', bbox_inches='tight')

    # Find the combination with the lowest 'Execution Time' for each set of parameters
    exec_time_combinations = fitness_dataframe.groupby(['Operator Name', 'Population Size', 'Tournament Size'])[
        'Execution Time'].min().reset_index()

    # Pivot the DataFrame to prepare for the heatmap
    pivot_df = exec_time_combinations.pivot(index='Operator Name', columns=['Population Size', 'Tournament Size'],
                                            values='Execution Time')

    # Create a heatmap using Seaborn
    plt.figure(figsize=(24, 10))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='g', cbar=True)
    plt.title(
        f'{country}: Execution Time for Different Combinations of Operator, Population Size, and Tournament Size')
    plt.xlabel('Population Size - Tournament Size')
    plt.ylabel('Operator Name')
    plt.tick_params(axis='y')
    plt.savefig(f'{country}\\{country}_execution_time.png', bbox_inches='tight')
# End Region
