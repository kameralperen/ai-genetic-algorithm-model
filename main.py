from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import partial

# Genetik algoritma parametreleri
population_size = 150  # Popülasyon büyüklüğü
num_generations = 30  # Nesil sayısı
mutation_rate = 0.2   # Mutasyon oranı (%20)


def eval_fitness(individual):
    total_area = 0
    penalty = 0
    for i, (x, y, w, h) in enumerate(individual):
        if x is None or y is None:  # Yerleştirilemeyen dikdörtgenleri atla
            continue
        if x + w > plate_size[0] or y + h > plate_size[1]:
            penalty += 10000
            continue
        total_area += w * h
        for j, (other_x, other_y, other_w, other_h) in enumerate(individual):
            if i != j and (other_x is not None and other_y is not None) and check_overlap((x, y, w, h), (other_x, other_y, other_w, other_h)):
                penalty += 10000
    return total_area - penalty,

# Çakışma kontrol
def check_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

# Birey oluşturma
def create_individual(rectangles):
    individual = []
    for w, h in rectangles:
        placed = False
        for _ in range(100):  # Yeniden deneme sayısı
            x, y = random.randint(0, plate_size[0] - w), random.randint(0, plate_size[1] - h)
            rect = (x, y, w, h)
            if not any(check_overlap(rect, existing_rect) for existing_rect in individual if None not in existing_rect):
                individual.append(rect)
                placed = True
                break
        if not placed:
            individual.append((None, None, None, None))  # Yerleştirilemeyen dikdörtgen
    return individual

# Gelişmiş çaprazlama stratejisi
def crossover(ind1, ind2):
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

# Gelişmiş mutasyon stratejisi
def mutate(individual, rectangles):
    for i, (x, y, w, h) in enumerate(individual):
        if x is None or y is None:  # Yerleştirilemeyen dikdörtgenleri atla
            continue
        if random.random() < mutation_rate:
            new_x = random.randint(0, plate_size[0] - rectangles[i][0])
            new_y = random.randint(0, plate_size[1] - rectangles[i][1])
            individual[i] = (new_x, new_y, w, h)
    return individual,

# DEAP araçlarının tanımlanması
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("evaluate", eval_fitness)
toolbox.register("mate", crossover)
toolbox.register("select", tools.selTournament, tournsize=3)

# Üç problem için dikdörtgen boyutlarını ve plaka boyutlarını tanımlıyoruz
problems = {
    "problem1": ([(2, 12), (7, 12), (8, 6), (3, 6), (3, 5), (5, 5), (3, 12), (3, 7), (5, 7),
                 (2, 6), (3, 2), (4, 2), (3, 4), (4, 4), (9, 2), (11, 2)], (20, 20)),
    "problem2": ([(4, 1), (4, 5), (9, 4), (3, 5), (3, 9), (1, 4), (5, 3), (4, 1), (5, 5),
                 (7, 2), (9, 3), (3, 13), (2, 8), (15, 4), (5, 4), (10, 6), (7, 2)], (20, 20)),
    "problem3": ([(4, 14), (5, 2), (2, 2), (9, 7), (5, 5), (2, 5), (7, 7), (3, 5), (6, 5),
                 (3, 2), (6, 2), (4, 6), (6, 3), (10, 3), (6, 3), (10, 3)], (20, 20))
}

# En iyi bireyin yerleşiminin görselleştirilmesi için fonksiyon
def plot_individual(individual, plate_size):
    fig, ax = plt.subplots()
    ax.set_xlim(0, plate_size[0])
    ax.set_ylim(0, plate_size[1])
    for x, y, width, height in individual:
        if x is None or y is None:  # Yerleştirilemeyen dikdörtgenleri atla
            continue
        rect = patches.Rectangle((x, y), width, height, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(rect)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# genetik algoritma fonksiyonu
for problem_name, (problem_rectangles, plate_size) in problems.items():
    # Birey oluşturma fonksiyonu için argümanları ayarla
    create_ind = partial(create_individual, problem_rectangles)

    # Fonksiyonları problem spesifik hale getir
    toolbox.register("individual", tools.initIterate, creator.Individual, create_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", mutate, rectangles=problem_rectangles)

    # Popülasyon oluştur ve genetik algoritmayı çalıştır
    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    result, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, stats=stats, halloffame=hof, verbose=True)

    # En iyi bireyi bul ve görselleştir
    best_individual = hof[0]
    best_fitness = eval_fitness(best_individual)[0]
    print(f"{problem_name} - En İyi Bireyin Uygunluk Değeri:", best_fitness)
    plot_individual(best_individual, plate_size)
