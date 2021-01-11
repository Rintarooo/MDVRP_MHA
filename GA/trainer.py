import random
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import numpy as np
from time import time

from classes import Depot, Customer


population_size = 25

depots = None
customers = None
population = None


def load_problem(path):
    global depots, customers
    depots = []
    customers = []

    with open(path) as f:
        max_vehicles, num_customers, num_depots = tuple(map(lambda z: int(z), f.readline().strip().split()))

        for i in range(num_depots):
            max_duration, max_load = tuple(map(lambda z: int(z), f.readline().strip().split()))
            depots.append(Depot(max_vehicles, max_duration, max_load))

        for i in range(num_customers):
            vals = tuple(map(lambda z: int(z), f.readline().strip().split()))
            cid, x, y, service_duration, demand = (vals[j] for j in range(5))
            customers.append(Customer(cid, x, y, service_duration, demand))

        for i in range(num_depots):
            vals = tuple(map(lambda z: int(z), f.readline().strip().split()))
            cid, x, y = (vals[j] for j in range(3))
            depots[i].pos = (x, y)

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def find_closest_depot(pos):
    closest_depot = None
    closest_distance = -1
    for i, depot in enumerate(depots):
        d = distance(depot.pos, pos)
        if closest_depot is None or d < closest_distance:
            closest_depot = (depot, i)
            closest_distance = d

    return closest_depot[0], closest_depot[1], closest_distance


def is_consistent_route(route, depot, include_reason=False):
    # print('route:', route)
    route_load = 0
    route_duration = 0
    last_pos = depot.pos
    for c in route:
        customer = customers[c - 1]
        route_load += customer.demand
        route_duration += distance(last_pos, customer.pos) + customer.service_duration
        last_pos = customer.pos
    # route_duration += find_closest_depot(last_pos)[2]
    route_duration += distance(last_pos, depot.pos)

    if include_reason:
        if route_load > depot.max_load:
            return False, 1
        if depot.max_duration != 0 and route_duration > depot.max_duration:
            return False, 2
        return True, 0
    return route_load <= depot.max_load and (depot.max_duration == 0 or route_duration <= depot.max_duration)


def is_consistent(chromosome):
    for c in customers:
        if c.id not in chromosome:
            return False

    routes = decode(chromosome)
    for d in range(len(routes)):
        depot = depots[d]
        if len(routes[d]) > depot.max_vehicles:
            return False
        for route in routes[d]:
            if not is_consistent_route(route, depot):
                return False
    return True


def encode(routes):
    chromosome = []
    for d in range(len(routes)):
        if d != 0:
            chromosome.append(-1)
        for r in range(len(routes[d])):
            if r != 0:
                chromosome.append(0)
            chromosome.extend(routes[d][r])
    return chromosome


def decode(chromosome):
    routes = [[[]]]
    d = 0
    r = 0
    for i in chromosome:
        if i < 0:
            routes.append([[]])
            d += 1
            r = 0
        elif i == 0:
            routes[d].append([])
            r += 1
        else:
            routes[d][r].append(i)
    return routes

# calculate and return fitness
def evaluate(chromosome, return_distance=False):
    for c in customers:
        if c.id not in chromosome:
            if return_distance:
                return math.inf
            return 0

    routes = decode(chromosome)
    score = 0
    for depot_index in range(len(routes)):
        depot = depots[depot_index]
        for route in routes[depot_index]:
            route_length, route_load = evaluate_route(route, depot, True)

            score += route_length

            if depot.max_duration and route_length > depot.max_duration:
                score += (route_length - depot.max_duration) * 20
            if route_load > depot.max_load:
                score += (route_load - depot.max_load) * 50
    if return_distance:
        return score
    return 1/score


def evaluate_route(route, depot, return_load=False):
    if len(route) == 0:
        if return_load:
            return 0, 0
        return 0
    route_load = 0
    route_length = 0
    customer = None
    last_pos = depot.pos
    for cid in route:
        customer = customers[cid - 1]
        route_load += customer.demand
        route_length += distance(last_pos, customer.pos)
        route_length += customer.service_duration# = 0
        last_pos = customer.pos
    # route_length += find_closest_depot(customer.pos)[1]
    # route_length += find_closest_depot(customer.pos)[2]
    route_length += distance(last_pos, depot.pos)

    if return_load:
        return route_length, route_load
    return route_length


def schedule_route(route):
    if not len(route):
        return route
    new_route = []
    prev_cust = random.choice(route)
    route.remove(prev_cust)
    new_route.append(prev_cust)

    while len(route):
        prev_cust = min(route, key=lambda x: distance(customers[x - 1].pos, customers[prev_cust - 1].pos))
        route.remove(prev_cust)
        new_route.append(prev_cust)
    return new_route


def create_heuristic_chromosome(groups):
    # Group customers in routes according to savings
    routes = [[] for i in range(len(depots))]
    missing_customers = list(map(lambda x: x.id, customers))
    for d in range(len(groups)):
        depot = depots[d]
        savings = []
        for i in range(len(groups[d])):
            ci = customers[groups[d][i] - 1]
            savings.append([])
            for j in range(len(groups[d])):
                if j <= i:
                    savings[i].append(0)
                else:
                    cj = customers[groups[d][j] - 1]
                    savings[i].append(distance(depot.pos, ci.pos) + distance(depot.pos, cj.pos) -
                                      distance(ci.pos, cj.pos))
        savings = np.array(savings)
        order = np.flip(np.argsort(savings, axis=None), 0)

        for saving in order:
            i = saving // len(groups[d])
            j = saving % len(groups[d])

            ci = groups[d][i]
            cj = groups[d][j]

            ri = -1
            rj = -1
            for r, route in enumerate(routes[d]):
                if ci in route:
                    ri = r
                if cj in route:
                    rj = r

            route = None
            if ri == -1 and rj == -1:
                if len(routes[d]) < depot.max_vehicles:
                    route = [ci, cj]
            elif ri != -1 and rj == -1:
                if routes[d][ri].index(ci) in (0, len(routes[d][ri]) - 1):
                    route = routes[d][ri] + [cj]
            elif ri == -1 and rj != -1:
                if routes[d][rj].index(cj) in (0, len(routes[d][rj]) - 1):
                    route = routes[d][rj] + [ci]
            elif ri != rj:
                route = routes[d][ri] + routes[d][rj]

            if route:
                if is_consistent_route(route, depot, True)[1] == 2:
                    route = schedule_route(route)
                if is_consistent_route(route, depot):
                    if ri == -1 and rj == -1:
                        routes[d].append(route)
                        missing_customers.remove(ci)
                        if ci != cj:
                            missing_customers.remove(cj)
                    elif ri != -1 and rj == -1:
                        routes[d][ri] = route
                        missing_customers.remove(cj)
                    elif ri == -1 and rj != -1:
                        routes[d][rj] = route
                        missing_customers.remove(ci)
                    elif ri != -1 and rj != -1:
                        if ri > rj:
                            routes[d].pop(ri)
                            routes[d].pop(rj)
                        else:
                            routes[d].pop(rj)
                            routes[d].pop(ri)
                        routes[d].append(route)


    # Order customers within routes
    for i, depot_routes in enumerate(routes):
        for j, route in enumerate(depot_routes):
            new_route = schedule_route(route)
            routes[i][j] = new_route

    chromosome = encode(routes)
    chromosome.extend(missing_customers)
    return chromosome


def create_random_chromosome(groups):
    routes = []
    for d in range(len(groups)):
        depot = depots[d]
        group = groups[d][:]
        random.shuffle(group)
        routes.append([[]])

        r = 0
        route_cost = 0
        route_load = 0
        last_pos = depot.pos
        for c in group:
            customer = customers[c - 1]
            # cost = distance(last_pos, customer.pos) + customer.service_duration + find_closest_depot(customer.pos)[2]
            cost = distance(last_pos, customer.pos) + customer.service_duration# + find_closest_depot(customer.pos)[2]
            if route_cost + cost > depot.max_duration or route_load + customer.demand > depot.max_load:
                r += 1
                routes[d].append([])
            routes[d][r].append(c)

    return encode(routes)


def initialize(random_portion=0):
    global population
    population = []
    groups = [[] for i in range(len(depots))]

    # Group customers to closest depot
    for c in customers:
        depot, depot_index, dist = find_closest_depot(c.pos)
        groups[depot_index].append(c.id)

    for z in range(int(population_size * (1 - random_portion))):
        chromosome = create_heuristic_chromosome(groups)
        population.append((chromosome, evaluate(chromosome)))

    for z in range(int(population_size * random_portion)):
        chromosome = create_random_chromosome(groups)
        population.append((chromosome, evaluate(chromosome)))


def select(portion, elitism=0):
    total_fitness = sum(map(lambda x: x[1], population))
    weights = list(map(lambda x: (total_fitness - x[1])/(total_fitness * (population_size - 1)), population))
    selection = random.choices(population, weights=weights, k=int(population_size*portion - elitism))
    population.sort(key=lambda x: -x[1])
    if elitism > 0:
        selection.extend(population[:elitism])
    return selection


def crossover(p1, p2):
    protochild = [None] * max(len(p1), len(p2))
    cut1 = int(random.random() * len(p1))
    cut2 = int(cut1 + random.random() * (len(p1) - cut1))
    substring = p1[cut1:cut2]

    for i in range(cut1, cut2):
        protochild[i] = p1[i]

    p2_ = list(reversed(p2))
    for g in substring:
        if g in p2_:
            p2_.remove(g)
    p2_.reverse()

    j = 0
    for i in range(len(protochild)):
        if protochild[i] is None:
            if j >= len(p2_):
                break
            protochild[i] = p2_[j]
            j += 1

    i = len(protochild) - 1
    while protochild[i] is None:
        protochild.pop()
        i -= 1

    population.append((protochild, evaluate(protochild)))


def heuristic_mutate(p):
    g = []
    for i in range(3):
        g.append(int(random.random() * len(p)))

    offspring = []
    for i in range(len(g)):
        for j in range(len(g)):
            if g == j:
                continue
            o = p[:]
            o[g[i]], o[g[j]] = o[g[j]], o[g[i]]
            offspring.append((o, evaluate(o)))

    selected_offspring = max(offspring, key=lambda o: o[1])
    population.append(selected_offspring)


def inversion_mutate(p):
    cut1 = int(random.random() * len(p))
    cut2 = int(cut1 + random.random() * (len(p) - cut1))

    if cut1 == cut2:
        return
    if cut1 == 0:
        child = p[:cut1] + p[cut2 - 1::-1] + p[cut2:]
    else:
        child = p[:cut1] + p[cut2 - 1:cut1 - 1:-1] + p[cut2:]
    population.append((child, evaluate(child)))


def best_insertion_mutate(p):
    g = int(random.random() * len(p))

    best_child = None
    best_score = 0

    for i in range(len(p) - 1):
        child = p[:]
        gene = child.pop(g)
        child.insert(i, gene)
        score = evaluate(child)
        if score > best_score:
            best_score = score
            best_child = child

    population.append((best_child, best_score))


def depot_move_mutate(p):
    if -1 not in p:
        return
    i = int(random.random() * len(p))
    while p[i] != -1:
        i = (i + 1) % len(p)

    move_len = int(random.random() * 10) - 5
    new_pos = (i + move_len) % len(p)

    child = p[:]
    child.pop(i)
    child.insert(new_pos, -1)
    population.append((child, evaluate(child)))


def route_merge(p):
    routes = decode(p)

    d1 = int(random.random() * len(routes))
    r1 = int(random.random() * len(routes[d1]))
    d2 = int(random.random() * len(routes))
    r2 = int(random.random() * len(routes[d2]))

    if random.random() < 0.5:
        limit = int(random.random() * len(routes[d2][r2]))
    else:
        limit = len(routes[d2][r2])

    reverse = random.random() < 0.5

    for i in range(limit):
        if reverse:
            routes[d1][r1].append(routes[d2][r2].pop(0))
        else:
            routes[d1][r1].append(routes[d2][r2].pop())
    routes[d1][r1] = schedule_route(routes[d1][r1])
    routes[d2][r2] = schedule_route(routes[d2][r2])
    child = encode(routes)
    population.append((child, evaluate(child)))


def train(generations, crossover_rate, heuristic_mutate_rate, inversion_mutate_rate,
          depot_move_mutate_rate, best_insertion_mutate_rate, route_merge_rate, t1,
          intermediate_plots=False, log=True):
    global population
    for g in range(generations):
        if log and g % 10 == 0:
            best = max(population, key=lambda x: x[1])
            print(f'[Generation {g}] Best score: {best[1]} Consistent: {is_consistent(best[0])}')

        # plottime
        if intermediate_plots and g % 500 == 0:
            if g != 0:
                population.sort(key=lambda x: -x[1])
                plot(population[0][0])

        selection = select(heuristic_mutate_rate + inversion_mutate_rate
                           + crossover_rate + depot_move_mutate_rate + best_insertion_mutate_rate
                           + route_merge_rate)
        selection = list(map(lambda x: x[0], selection))

        offset = 0
        for i in range(int((population_size * crossover_rate) / 2)):
            p1, p2 = selection[2*i + offset], selection[2*i + 1 + offset]
            crossover(p1, p2)
            crossover(p2, p1)
        offset += int(population_size * crossover_rate)

        for i in range(int(population_size * heuristic_mutate_rate)):
            heuristic_mutate(selection[i + offset])
        offset += int(population_size * heuristic_mutate_rate)

        for i in range(int(population_size * inversion_mutate_rate)):
            inversion_mutate(selection[i + offset])
        offset += int(population_size * inversion_mutate_rate)

        for i in range(int(population_size * depot_move_mutate_rate)):
            depot_move_mutate(selection[i + offset])
        offset += int(population_size * depot_move_mutate_rate)

        for i in range(int(population_size * best_insertion_mutate_rate)):
            best_insertion_mutate(selection[i + offset])
        offset += int(population_size * best_insertion_mutate_rate)

        for i in range(int(population_size * route_merge_rate)):
            route_merge(selection[i + offset])
        offset += int(population_size * route_merge_rate)

        population = select(1.0, elitism=4)

    population.sort(key=lambda x: -x[1])
    print("\n\nFinished training")

    best_solution = None
    if is_consistent(population[0][0]):
        print(f'Best score: {population[0][1]}, best distance: {evaluate(population[0][0], True)}')
        best_solution = population[0][0]
    else:
        for c in population:
            if is_consistent(c[0]):
                print(f'Best score: {c[1]}, best distance: {evaluate(c[0], True)}')
                best_solution = c[0]
                break
        else:
            print('Found no consistent solutions.')
    print(f'inference time: {time()-t1}s')
    if best_solution:
        plot(best_solution)
    return best_solution


def plot_map(show=True, annotate=True):
    depot_positions = np.array(list(map(lambda x: x.pos, depots)))
    customer_positions = np.array(list(map(lambda x: x.pos, customers)))

    depot_ids = np.arange(1, len(depots) + 1)
    customer_ids = np.arange(1, len(customers) + 1)

    depot_positions = np.array(list(map(lambda x: x.pos, depots)))
    customer_positions = np.array(list(map(lambda x: x.pos, customers)))
    plt.scatter(depot_positions[:, 0], depot_positions[:, 1], c='r', s=60, zorder=10)
    plt.scatter(customer_positions[:, 0], customer_positions[:, 1], c='k', s=20, zorder=20)

    if annotate:
        for i, id in enumerate(depot_ids):
            plt.annotate(id, depot_positions[i], zorder=30)
        for i, id in enumerate(customer_ids):
            plt.annotate(id, customer_positions[i], zorder=30)

    if show:
        plt.show()


def plot(chromosome):
    r = decode(chromosome)
    print('depot No., visit route')
    for d, routes in enumerate(r):
        depot = depots[d]
        for route in routes:
            positions = [depot.pos]
            last_pos = depot.pos
            for cid in route:
                last_pos = customers[cid - 1].pos
                positions.append(last_pos)
            # positions.append(find_closest_depot(last_pos)[0].pos)
            positions.append(depot.pos)

            positions = np.array(positions)
            plt.plot(positions[:, 0], positions[:, 1], zorder=0)
            print(d+1, route)

    plot_map(False)

    plt.show()


def save_solution(chromosome, path):
    routes = decode(chromosome)
    total_duration = evaluate(chromosome, True)

    with open(path, 'w') as f:
        f.write(f'{total_duration:.2f}\n')

        for d, depot in enumerate(depots):
            for r, route in enumerate(routes[d]):
                route_length, route_load = evaluate_route(route, depot, True)
                f.write(f'{d + 1}\t{r + 1}\t{route_length:.2f}\t{route_load}\t')
                end_depot = find_closest_depot(customers[route[-1] - 1].pos)[1]
                f.write(f'{end_depot + 1}\t')

                f.write(' '.join([str(c) for c in route]))
                f.write('\n')
