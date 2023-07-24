# -*- coding: utf-8 -*-
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools


cities = ["Москва", "Казань", "Саратов", "Сочи", "Санкт Петербург"]

cost_matrix = [
    [0, 5000, 4000, 7000, 8000],
    [5000, 0, 3000, 10000, 6000],
    [4000, 3000, 0, 6000, 9000],
    [7000, 10000, 6000, 0, 5000],
    [8000, 6000, 9000, 5000, 0]
]

time_matrix = [
    [0, 3, 2, 5, 6],
    [3, 0, 2, 6, 4],
    [2, 2, 0, 4, 5],
    [5, 6, 4, 0, 3],
    [6, 4, 5, 3, 0]
]


creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(cities)), len(cities))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    distance = 0
    time = 0
    for i in range(len(individual)):
        from_city = individual[i]
        to_city = individual[(i + 1) % len(individual)]
        distance += cost_matrix[from_city][to_city]
        time += time_matrix[from_city][to_city]
    return distance, time


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def visualize_route(individual):
    G = nx.DiGraph()
    labels = {}
    for i in range(len(individual)):
        from_city = individual[i]
        to_city = individual[(i + 1) % len(individual)]
        G.add_edge(cities[from_city], cities[to_city])
        labels[(cities[from_city], cities[to_city])] = f"Transport: {get_transport_type(individual[i])}"
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_color='black', font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.axis('off')
    plt.show()

def get_transport_type(index):
    if index == 0:
        return "Автомобиль"
    elif index == 1:
        return "Поезд"
    elif index == 2:
        return "Самолет"

def main():
    random.seed(42)

    pop_size = 100
    cx_prob = 0.8
    mut_prob = 0.2
    num_generations = 100

    pop = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "min_distance", "min_time"

    best_distance = float('inf')
    best_time = float('inf')
    best_individual = None
    num_iterations = 0

    for gen in range(num_generations):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob)
        fitnesses = toolbox.map(toolbox.evaluate, offspring)

        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))

        record = stats.compile(pop)
        logbook.record(gen=gen, **record)

        if record["min"][0] < best_distance and record["min"][1] < best_time:
            best_distance = record["min"][0]
            best_time = record["min"][1]
            best_individual = pop[np.argmin([toolbox.evaluate(ind) for ind in pop])]
            num_iterations = gen

    print("Optimal route found:")
    optimal_route = [cities[i] for i in best_individual]
    optimal_route.append(cities[best_individual[0]])  # Добавляем начальную вершину в конец маршрута
    print(optimal_route)
    print("Number of iterations:", num_iterations)

    visualize_route(best_individual)

if __name__ == "__main__":
    main()
