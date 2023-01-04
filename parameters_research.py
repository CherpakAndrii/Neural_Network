import multiprocessing
from MKR_Cherpak import get_nn_results
from random import randint as rand
from os import system

iterations_boundaries = (1, 50)
generation_boundaries = (100, 1000)
solution_boundaries = (5, 100)
parents_boundaries = (5, 100)
range_boundaries = (5, 50)
param_vector = tuple[int, int, int, int, int]


def sort_arrs(arr_by: list[float], arr_dependent: list[param_vector]) -> (list[param_vector], list[float]):
    def get_key(elem: tuple[param_vector, float]) -> float:
        return elem[1]

    res = sorted(zip(arr_dependent, arr_by), key=get_key)
    res_by = [el[1] for el in res]
    res_dependent = [el[0] for el in res]

    return res_dependent, res_by


def get_rates(population: list[param_vector]) -> list[float]:
    with multiprocessing.Pool() as pool:
        rates = pool.map(get_nn_results, population)
    return rates


def generate_start_population(quantity: int) -> list[param_vector]:
    def rel_ind(boundaries, index):
        return int((boundaries[1] - boundaries[0]) / quantity * index)

    population = [(rel_ind(iterations_boundaries, i),
                   rel_ind(generation_boundaries, i),
                   rel_ind(solution_boundaries, i),
                   rel_ind(parents_boundaries, i),
                   rel_ind(range_boundaries, i)) for i in range(quantity)]

    return population


def check_for_viability(child: param_vector) -> bool:
    return iterations_boundaries[0] <= child[0] <= iterations_boundaries[1] and \
           generation_boundaries[0] <= child[1] <= generation_boundaries[1] and \
           parents_boundaries[0] <= child[2] <= parents_boundaries[1] and \
           solution_boundaries[0] <= child[3] <= solution_boundaries[1] and \
           range_boundaries[0] <= child[4] <= range_boundaries[1] and child[2] <= child[3]


def get_children(parents: tuple[param_vector, param_vector]) -> list[param_vector]:
    point1 = rand(1, len(parents[0]))
    point2 = rand(1, len(parents[0]))
    p1, p2 = min(point1, point2), max(point1, point2)
    src1, src2 = list(parents[0]), list(parents[1])
    child1 = src1[:p1] + src2[p1:p2] + src2[p2:]
    child2 = src2[:p1] + src1[p1:p2] + src2[p2:]
    child3, child4 = [el for el in child1], [el for el in child2]
    for ch in [child3, child4]:
        for i in len(ch):
            ch[i] = int(ch[i] * (rand(90, 110) / 100 if rand(0, 10) < 4 else 1))

    return [tuple(ch) for ch in [child1, child2, child3, child4] if check_for_viability(ch)]


def get_next_generation(population, num_of_parents) -> list[param_vector]:
    parents = population[:num_of_parents]
    with multiprocessing.Pool() as pool:
        children_arrs = pool.map(get_children, [(parents[i], parents[i + 1]) for i in range(0, len(parents), 2)])
    children = []
    for arr in children_arrs:
        children += arr
    return children


def update_population(population: list[param_vector], rates: list[float], size: int, parent_num: int) -> (list[param_vector], list[float]):
    children = get_next_generation(population, parent_num)
    children_rates = get_rates(children)
    population, rates = sort_arrs(rates + children_rates, population + children)
    return population[:size], rates[:size]


def get_best_params(population_size=10, iterations=5) -> param_vector:
    population = generate_start_population(population_size)
    rates = get_rates(population)
    population, rates = sort_arrs(rates, population)
    num_of_parents = population_size // 4 * 2
    for i in range(iterations):
        population, rates = update_population(population, rates, population_size, num_of_parents)
        log(f'Iteration {i+1} done. Best square mean: {rates[0]}')
        system('CLS')
        print(f'Iteration {i+1} done. Best square mean: {rates[0]}')

    return population[0]


def log(message: str):
    handle = open('res.txt', 'a+')
    handle.write(message+'\n')
    handle.close()


if __name__ == '__main__':
    params = get_best_params()
    log(f'number_of_iterations: {params[0]}\nnum_generations: {params[1]}\nnum_parents_mating: {params[2]}\nnum_solutions: {params[3]}\nfunction_range: {params[4]}')
