import numpy as np
import pygad.gann
from math import sin, cos
import torch.nn as t
from torch import tensor
import pygad.torchga as torchga
import matplotlib.pyplot as plt


def z(x, y):
    return float(x * cos(x) + sin(y))


def start_nn(number_of_iterations, num_generations, num_parents_mating, num_solutions, function_range):
    nn_architecture = [2, 2, 8, 6, 7, 2, 1]
    global model, loss_fun
    model = get_model(architecture=nn_architecture)
    loss_fun = t.L1Loss()
    print(model)
    (X, Y, Z) = get_data(function_range)
    solution, solution_fitness, solution_idx, predictions, abs_error = run_algo(X, Y, Z, (number_of_iterations, num_generations, num_parents_mating, num_solutions))
    print_results(solution_fitness, solution_idx, predictions, abs_error)
    get_plot(X, Y, predictions)
    return abs_error


def run_algo(X, Y, Z, params: tuple[int, int, int, int]): #(number_of_iterations, num_generations, num_parents_mating, num_solutions)
    global data_inputs, data_outputs
    data_inputs = tensor([[x, y] for (x, y) in zip(X, Y)])
    data_outputs = tensor(Z)
    solution, solution_fitness, solution_idx, abs_error, predictions = 0, 0, 0, 0, []

    genetic_algo = pygad.GA(num_generations=params[1],
                            num_parents_mating=params[2],
                            initial_population=torchga.TorchGA(model, params[3]).create_population(),
                            fitness_func=fitness_func,
                            on_generation=callback_generation)

    for i in range(params[0]):
        print(f"Iteration {i + 1}:")
        genetic_algo.run()
        solution, solution_fitness, solution_idx = genetic_algo.best_solution()
        predictions = torchga.predict(model=model,
                                      solution=solution,
                                      data=data_inputs)
        abs_error = loss_fun(predictions, data_outputs)
        print(f'\tAbsolute error: {abs_error}')

    return solution, solution_fitness, solution_idx, predictions, abs_error


def get_model(architecture) -> t.modules.container.Sequential:
    linear_layers = [t.Linear(a, b) for (a, b) in zip(architecture[:-1], architecture[1:])]
    relu_layer = t.ReLU()
    softmax_layer = t.Softmax()
    model_layers = [linear_layers[0]]
    for lin_layer in linear_layers[1:]:
        model_layers.append(softmax_layer)
        model_layers.append(lin_layer)

    model_layers[-2] = relu_layer
    return t.Sequential(*model_layers)


def fitness_func(solution, sol_idx):
    predictions = torchga.predict(model=model, solution=solution, data=data_inputs)
    abs_error = loss_fun(predictions, data_outputs).detach().numpy() + 0.00000001
    solution_fitness = 1.0 / abs_error
    return solution_fitness


def callback_generation(genetic_algo):
    print(f"\rGeneration = {genetic_algo.generations_completed}\tFitness = {genetic_algo.best_solution()[1]}", end='')


def get_data(dist=5) -> (list[float], list[float], list[float]):
    X = np.linspace(0, dist, dist * 8)
    Y = np.linspace(0, dist, dist * 8)
    Z = []
    for i in range(len(X)):
        Z.append([float(z(X[i], Y[i]))])
    return (X.tolist(), Y.tolist(), Z)


def print_results(solution_fitness, solution_idx, predictions, abs_error):
    print(f"Fitness-цінність найкращого рішення = {solution_fitness}")
    print(f"Індекс найкращого рішення: {solution_idx}")
    print("Прогнози: \n", predictions.detach().numpy())
    print("Абсолютна помилка: ", abs_error.detach().numpy())


def get_plot(X, Y, predictions):
    ax = plt.axes(projection='3d')
    plot_X = []
    plot_Y = []
    plot_Z_pred = []
    plot_Z = []
    for i in range(len(X)):
        plot_X.append(X[i])
        plot_Y.append(Y[i])
        plot_Z_pred.append(predictions.detach().numpy().tolist()[i][0])
        plot_Z.append(z(X[i], Y[i]))
    ax.plot3D(plot_X, plot_Y, plot_Z, 'blue')
    ax.plot3D(plot_X, plot_Y, plot_Z_pred, 'red')
    plt.show()


def get_nn_results(params):
    return start_nn(*params)


if __name__ == '__main__':
    start_nn(10, 800, 5, 10, 7)
