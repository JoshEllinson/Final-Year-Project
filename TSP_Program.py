# LJMU Travelling Salesman Problem Code Project
# Implements and compares exact and approimxation algorithms for the Travelling Salesman Problem

#Library Imports

import math
import time
import random
import tsplib95
import matplotlib.pyplot as plt
import csv
import os

#------------------ Data loading and exporting with TSPLIB and CSV -------------------- #

#Function for loading a .tsp file and returning the contents as a list of cities [id, x, y]
def loadTSPLib(filepath):

    problem = tsplib95.load(filepath)
    #Error checking for debugging the loading of the file 
    if not problem.node_coords:
        raise ValueError("Error loading coordinates")
    
    cities = []
    for node_id, (x, y) in problem.node_coords.items():
        cities.append((node_id, x, y))
    
    #Sorts the cities list to ensure indexing is consistent
    cities.sort(key=lambda c: c[0])
    return cities, problem

#Function for saving the results into a csv file
def saveResults(rows, filename="TSP_Results.csv"):
    fieldnames = ["dataset", "file", "n", "reference_type", "reference_distance", "algorithm", "distance", "runtime_s", "accuracy"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    
    print(f"\n Saved results to CSV: {filename}")

#------------------ Functions for distance calculations -----------------#
#Function calculates the euclidean distance of cities
def euclideanDis(a, b):
    return math.sqrt((a[1] - b[1])**2 + (a[2]-b[2])**2)

#Function calculates the total length of the travelling salesman
def totalDistance(tour):
    distance = 0.0
    for i in range(len(tour) - 1):
        distance += euclideanDis(tour[i], tour[i+1])
    return distance

#Function generates a random TSP tour (that is valid) 
def makeRandomTour(cities):
    start = cities[0]
    middle = cities[1:].copy()
    random.shuffle(middle)
    tour = [start] + middle + [start]
    return tour

#Function that performs an edge swap between two indicies (i and j) 
def twoOptSwap(tour, i, j):
    newTour = tour[:]
    newTour[i:j] = reversed(newTour[i:j])
    return newTour

#------------------- Functions for each algorithm --------------------#

#Function for the Held-Karp algorithm (EXACT ALGORITHM)
def heldKarp(cities):

    n= len(cities)
    cityDistance = [[euclideanDis(cities[i], cities[j]) for j in range(n)] for i in range(n)]

    #dp represents the minimum cost to reach the city "j" after visiting every city subset
    dp = {}
    for i in range (1, n):
        dp[(1 << i, i)] = cityDistance[0][i]
    
    for subset_size in range(2, n):
        for subset in range(1 << n):
            if subset & 1:
                continue
            if bin(subset).count("1") != subset_size:
                continue
            for j in range(n):
                if subset & (1 << j):
                    dp[(subset, j)] = min(dp[(subset ^ (1 << j), k)] + cityDistance[k][j] for k in range(n) if subset & (1 << k) and k != j)

    full = (1 << n) - 2
    best = min(dp[(full, j)] + cityDistance[j][0] for j in range(1, n))
    return best

#Function for Branch and Bound algorithm (EXACT ALGORITHM) 
#A time limit has been implemented to prevent the program hanging and to provide results in a relatively fast time
def branchAndBound(cities, time_limit=None):
    n = len(cities)
    dist = [[euclideanDis(cities[i], cities[j]) for j in range(n)] for i in range(n)]

    min_out = [min(dist[i][j] for j in range(n) if j != i) for i in range(n)]
    visited = [False] * n
    visited[0] = True
    path = [0]

    #Greedy heuristic to make an upper bound for pruning
    def greedyUpperBound():
        unvis = set(range(n))
        cur = 0
        unvis.remove(cur)
        cost = 0.0
        while unvis:
            nxt = min(unvis, key=lambda j: dist[cur][j])
            cost += dist[cur][nxt]
            cur = nxt
            unvis.remove(cur)
        cost += dist[cur][0]
        return cost

    bestDistance = greedyUpperBound()
    nodes_expanded = 0
    start_time = time.perf_counter()

    #Function for computing a lower bound estimate for the current path
    #Is used to prune branches that cannot improve solution accuracy
    def lower_bound(cost):
        unvisited = [i for i in range(n) if not visited[i]]
        if not unvisited:
            return cost + dist[path[-1]][0]

        last = path[-1]
        min_leave = min(dist[last][j] for j in unvisited)
        min_return = min(dist[j][0] for j in unvisited)

        
        return cost + min_leave + min_return + sum(min_out[j] for j in unvisited)

    def dfs(cost):
        nonlocal bestDistance, nodes_expanded

        # Time check to help prevent program hanging
        if time_limit is not None and (time.perf_counter() - start_time) >= time_limit:
            return

        nodes_expanded += 1
        if nodes_expanded % 20000 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"Expanded {nodes_expanded} nodes | depth={len(path)} | best={bestDistance:.2f} | t={elapsed:.2f}s")

        #Branch is pruned if either the cost or bound exceed the current best solution
        if cost >= bestDistance:
            return
        if lower_bound(cost) >= bestDistance:
            return

        #Displays information about the function as it is running to help with debugging (Should I remove it before submission?)
        if len(path) == n:
            totalCost = cost + dist[path[-1]][0]
            if totalCost < bestDistance:
                bestDistance = totalCost
                elapsed = time.perf_counter() - start_time
                print(f"NEW BEST: {bestDistance:.2f}  (nodes={nodes_expanded}, t={elapsed:.2f}s)")
            return

        last = path[-1]
        candidates = [i for i in range(n) if not visited[i]]
        candidates.sort(key=lambda i: dist[last][i])

        for nxt in candidates:
            visited[nxt] = True
            path.append(nxt)
            dfs(cost + dist[last][nxt])
            path.pop()
            visited[nxt] = False

    dfs(0.0)
    return bestDistance


#Function for Nearest Neighbour (APPROXIMATION ALGORITHM)    
def nearestNeighbour(cities):
    unvisited = cities.copy()
    tour = [unvisited.pop(0)]

    #Selects the closest unvisited city at each step of the route
    while unvisited:
        last = tour[-1]
        nextCity = min(unvisited, key=lambda c: euclideanDis(last, c))
        tour.append(nextCity)
        unvisited.remove(nextCity)

    tour.append(tour[0])
    return tour, totalDistance(tour)    

#Function for 2-Opt algorithm to increase the accuracy of nearest neighbour
def twoOpt(tour):
    improvement = True
    bestTour = tour
    bestDistance = totalDistance(bestTour)

    #Uses nearest neighbour as a benchmark and removes edge crossings by swapping them
    while improvement:
        improvement = False
        for i in range(1, len(bestTour) - 2):
            for j in range(i + 1, len(bestTour) - 1):
                if j - i == 1:
                    continue
                
                newTour = bestTour[:]
                newTour[i:j] = reversed(bestTour[i:j])

                newDistance = totalDistance(newTour)
                if newDistance < bestDistance:
                    bestTour = newTour
                    bestDistance = newDistance
                    improvement = True
    #Returns the altered distance and tours for display 
    return bestTour, bestDistance

#Function for Simulated Annealing algorithim (Approximation) 
def simulatedAnnealing(cities, time_limit=5.0, start_temp=5000.0, cooling=0.999, iter_per_temp=200):
    startTime = time.perf_counter()

    # Uses the result of Nearest Neighbour as a starting point to attempt to improve accuracy
    tour, currentDist = nearestNeighbour(cities)

    bestTour = tour[:]
    bestDist = currentDist

    T = start_temp
    Tmin = 1e-9

    #Will loop until time limit is met or the temp becomes too small
    while (time.perf_counter() - startTime) < time_limit and T > Tmin:
        for _ in range(iter_per_temp):
            i = random.randint(1, len(tour) - 3)
            j = random.randint(i + 1, len(tour) - 2)

            candidate = twoOptSwap(tour, i, j)
            candDist = totalDistance(candidate)

            delta = candDist - currentDist

            #Will accept the candidate if it is either better or probabilistically accepts if it is worse
            if delta < 0 or random.random() < math.exp(-delta / T):
                tour = candidate
                currentDist = candDist

                #Tracks the current best solution so far
                if currentDist < bestDist:
                    bestDist = currentDist
                    bestTour = tour[:]
        #Will cool the temp to reduce the probability of a worse solution being accepted
        T *= cooling

    return bestTour, bestDist

#Functions for use in the genetic algorithm
# Function converts the city order into a tour distance
def routeDistanceOrder(order, cities):
    tour = [cities[0]] + [cities[i] for i in order] + [cities[0]]
    return totalDistance(tour)

# Function converts the city order into a tour list for use in plotting
def orderToTour(order, cities):
    return [cities[0]] + [cities[i] for i in order] + [cities[0]]

#Function selects parents using a tournament selection i.e: lower distance means better fitness
def tournamentSelect(population, fitnesses, k=3):
    bestIDX = None
    for _ in range(k):
        idx = random.randrange(len(population))
        if bestIDX is None or fitnesses[idx] < fitnesses[bestIDX]:
            bestIDX = idx
    return population[bestIDX]
#Function for performing an ordered crossover to create a valid child
def orderCrossover(parent1, parent2):
    n = len(parent1)
    a = random.randint(0, n - 2)
    b = random.randint(a + 1, n - 1)

    child = [None] * n
    child[a:b] = parent1[a:b]

    p2Items = [x for x in parent2 if x not in child]
    fillPos = [i for i in range(n) if child[i] is None]

    for i, val in zip(fillPos, p2Items):
        child[i] = val
    
    return child

#Function for swapping two indicies at random to maintain the "genetic" diversity of the population
def swapMutation(order, mutationRate=0.02):
    if random.random() < mutationRate:
        i = random.randint(0, len(order) - 1)
        j = random.randint(0, len(order) - 1)
        order[i], order[j] = order[j], order[i]

#Function for the Genetic Algorithm (APPROXIMATION ALGORITHM)
#To see the genetic algorithm act independently please change seedWithNN to False
def geneticAlgorithm(cities, time_limit=15.0, popSize=80, mutationRate=0.02, tournament_k=3, elitism=True, seedWithNN=True):
    start_time = time.perf_counter()
    n = len(cities)

    base = list(range(1, n))
    population = []
    
    if seedWithNN:        
        #Nearest Neighbour's result is used to seed the genetic algorithm
        nn_tour, _ = nearestNeighbour(cities)
        nn_order = [cities.index(c) for c in nn_tour[1:-1]]
        population.append(nn_order[:])

    #Will fill the remaining population with random permutations as long as they are valid    
    while len(population) < popSize:
        perm = base[:]
        random.shuffle(perm)
        population.append(perm)
    
    #Function evaluates the population's fitness and will return the best individual
    def evaluate(pop):
        fitnesses = [routeDistanceOrder(ind, cities) for ind in pop]
        bestIDX = min(range(len(pop)), key=lambda i: fitnesses[i])
        return fitnesses, pop[bestIDX], fitnesses[bestIDX]
    
    fitnesses, bestOrder, bestDist = evaluate(population)

    generation = 0
    #Will make sure that the algorithms stops when the time limit is reached
    while (time.perf_counter() - start_time) <time_limit:
        generation += 1

        newPopulation = []

        #Elitism will make sure that the best solution will "survive" till the next gen
        if elitism:
            newPopulation.append(bestOrder[:])
        
        #The next generation is built using selection and crossover
        while len(newPopulation) < popSize:
            p1 = tournamentSelect(population, fitnesses, k=tournament_k)
            p2 = tournamentSelect(population, fitnesses, k=tournament_k)

            child = orderCrossover(p1, p2)
            swapMutation(child, mutationRate=mutationRate)

            newPopulation.append(child)

        population = newPopulation
        fitnesses, genBestOrder, genBestDist = evaluate(population)

        #Updates the best solution if a better one is found
        if genBestDist < bestDist:
            bestDist = genBestDist
            bestOrder = genBestOrder[:]
    bestTour = orderToTour(bestOrder, cities)
    return bestTour, bestDist        

#Function for the Ant Colony Optimisation Algorithm
#With:
#Alpha representing pheromone improtance, beta being heuristic importance, rho being the evaporation rate and Q being the pheromone's deposit factor
def antColony(cities, time_limit=15.0, noAnts=30, alpha = 1.0, beta=3.0, rho=0.5, Q=100.0):
    start_time = time.perf_counter()
    n = len(cities)

    #Distance matrix is made to speed up repeated distance lookups
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = euclideanDis(cities[i], cities[j])
            else:
                dist[i][j] = 1e-12
    #Checks for the heuristic desirability of nodes
    eta = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            eta[i][j] = 1.0 / dist[i][j] if i != j else 0.0
    
    #Sets the inital pheromone levels for all edges
    tau0 = 1.0
    tau = [[tau0] * n for _ in range(n)]

    bestTour = None
    bestDist = float("inf")

    starterNode = 0

    #Builds a tour based oon the pheromone and the heuristic probability
    def buildAntTour():
        visited = [False] * n
        visited[starterNode] = True
        tourIDX = [starterNode]

        current = starterNode
        for _ in range (n - 1):
            candidates = [j for j in range(n) if not visited[j]]

            weights = []
            totalWeight = 0.0
            for j in candidates:
                w = (tau[current][j] ** alpha) * (eta[current][j] ** beta)
                weights.append(w)
                totalWeight += w
            
            #Rnadom selection for decision making
            r = random.random() * totalWeight
            cmu = 0.0
            chosen = candidates[-1]
            for j, w in zip(candidates, weights):
                cmu += w
                if cmu >= r:
                    chosen = j
                    break
            
            visited[chosen] = True
            tourIDX.append(chosen)
            current = chosen
        
        tourIDX.append(starterNode)
        return tourIDX

    #Function calculates the length of the tour represented as indicies
    def tourLengthIDX(tourIDX):
        d = 0.0
        for k in range(len(tourIDX) - 1):
            d += dist[tourIDX[k]][tourIDX[k + 1]]
        return d
    
    #Function conerts the tour into a city tour for use in plotting
    def idxCityTour(tourIDX):
        return [cities[i] for i in tourIDX]
    
    iteration = 0
    while(time.perf_counter() - start_time) < time_limit:
        iteration += 1

        antTours = []
        antLengths = []

        #Each "ant" will construct a tour and is then evaluated
        for _ in range(noAnts):
            tIDX = buildAntTour()
            length = tourLengthIDX(tIDX)
            antTours.append(tIDX)
            antLengths.append(length)

            #Tracks the current best solution so far
            if length < bestDist:
                bestDist = length
                bestTour = tIDX
        
        #Accounts for pheromone evaporation
        for i in range(n):
            for j in range(n):
                tau[i][j] *= (1.0 - rho)
        
        #Pheromones are despoited proportionally to the quality of the tour
        for tIDX, length in zip(antTours, antLengths):
            deposit = Q / length
            for k in range(len(tIDX) - 1):
                a = tIDX[k]
                b = tIDX[k + 1]
                tau[a][b] += deposit
                tau[b][a] += deposit
    
    return idxCityTour(bestTour), bestDist







#--------------------- Functions for evaluating each algorithm -------------------------#
#Function to measure the time an algorithm takes to execute
def runtime(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, end - start

#Function to calculate accuracy relative to the reference solution (One of the two Exact Algorithms depending on size of .tsp file)
def computationAccuracy(heuristicDistance, referenceDistance):
    if referenceDistance is None:
        return None
    if heuristicDistance <= referenceDistance:
        return 100.0
    return (referenceDistance / heuristicDistance) * 100


#--------------------- Function for program visualisation -------------------------#

#Function plots a tour using the x and y co-ords of each city
def plotTour(tour, title):
    x = [c[1] for c in tour]
    y = [c[2] for c in tour]

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

#Function generates comparrison plots for runtime, accuracy and distance
def plotComparisons(results, referenceDistance, title="Algorithm Comparisons"):
    names = [r[0] for r in results]
    dists = [r[1] for r in results]
    times = [r[2] for r in results]

    #Creates a bar chart for the distances
    plt.figure()
    plt.bar(names, dists)
    plt.title(title + " - Distance")
    plt.ylabel("Tour Distance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    #Creates a bar chart for the runtimes of the algorithms
    plt.figure()
    plt.bar(names, times)
    plt.title(title + " - Runtime")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    #Creates a bar chart for the accuracy of the algorithms compared to the reference solution
    if referenceDistance is not None:
        accs = []
        for _, dist, _ in results:
            acc = computationAccuracy(dist, referenceDistance)
            accs.append(acc if acc is not None else 0)

        plt.figure()
        plt.bar(names, accs)
        plt.title(title + " - Accuracy vs Reference")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()

#Function to plot the overall results of each algorithm across the datasets
def plotSummary(rows, title='Summary'):
    algs = sorted(set(r["algorithm"] for r in rows))

    #Display for Runtime vs Distance scattergraph
    plt.figure()
    for alg in algs:
        xs = [r["runtime_s"] for r in rows if r ["algorithm"] == alg]
        ys = [r["distance"] for r in rows if r ["algorithm"] == alg]
        plt.scatter(xs, ys, label=alg)
    plt.title(title + " - Runtime vs Distance")
    plt.xlabel("Runtime (s)")
    plt.ylabel("Distance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Display for Runtime vs Accuracy scattergraph
    plt.figure()
    for alg in algs:
        xs, ys = [], []
        for r in rows:
            if r["algorithm"] != alg:
                continue
            if r["accuracy"] is None:
                continue
            xs.append(r["runtime_s"])
            ys.append(r["accuracy"])
        if xs:
            plt.scatter(xs, ys, label=alg)
    plt.title(title + " - Runtime vs Accuracy")
    plt.xlabel("Runtime (s)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()




#Function that runs the dataset returning the results as rows ready to be appended to a .csv file
def runDataset(filepath, bb_time=30, sa_time=15, ga_time=15):
    cities, problem = loadTSPLib(filepath)
    datasetName = problem.name
    n = len(cities)

    print(f"\n=== {datasetName} ({n} cities) ===")

    referenceDistance = None
    referenceType = "None"
    referenceTime = None

    #Selects the suitable exact algorithm to run based on the size of .tsp problem
    if n <= 20:
        referenceDistance, referenceTime = runtime(heldKarp, cities)
        referenceType = "Held-Karp Algorithm"
    elif n <= 40:
        referenceDistance, referenceTime = runtime(branchAndBound, cities, bb_time)
        referenceType = f"Branch&Bound({bb_time}s)"
    else:
        referenceType = "Skipped"

    results, bestTour, bestDist = approximationAlgorithms(cities, referenceDistance, datasetName)

    #Converts data from the algorithms into rows for the CSV file
    rows = []
    for alg, dist, t in results:
        rows.append({
            "dataset": datasetName,
            "file": filepath,
            "n": n,
            "reference_type": referenceType,
            "reference_distance": referenceDistance,
            "algorithm": alg,
            "distance": dist,
            "runtime_s": t,
            "accuracy": computationAccuracy(dist, referenceDistance)
        })

    #Stores the result of the exact algorithm into the CSV if it is available
    if referenceDistance is not None and referenceTime is not None:
        rows.append({
            "dataset": datasetName,
            "file": filepath,
            "n": n,
            "reference_type": referenceType,
            "reference_distance": referenceDistance,
            "algorithm": referenceType,
            "distance": referenceDistance,
            "runtime_s": referenceTime,
            "accuracy": 100.0
        })
    return rows, bestTour, bestDist, datasetName



#----------------------- Main Body -----------------------#

#Executes the algorithms across every dataset available (Every dataset in the file alongside TSP_Program.py)
def main():
    tsp_files = [f for f in os.listdir(".") if f.lower().endswith(".tsp")]
    tsp_files.sort()

    if not tsp_files:
        print("No .tsp files found.")
        return

    all_rows = []

    for tsp in tsp_files:
        rows, bestTour, bestDist, datasetName = runDataset(tsp, bb_time=30)
        all_rows.extend(rows)
        
        #Visualises the best tour for each dataset
        plotTour(bestTour, f"{datasetName} Best Tour (dist={bestDist:.2f})")
    
    #All results are exported into a results csv file and the summary plots are displayed
    saveResults(all_rows, "TSP_Results.csv")
    plotSummary(all_rows, title="TSPLIB Results")
   



def approximationAlgorithms(cities, referenceDistance=None, datasetName=""):
    results = []

    #Nearest Neighbour
    (tour, nnDist), nnTime = runtime(nearestNeighbour, cities)
    results.append(("Nearest Neighbour", nnDist, nnTime))

    #NN with 2-Opt
    (optTour, optDist), optTime = runtime(twoOpt, tour)
    results.append(("NN with 2-Opt", optDist, optTime))

    #Simulated Annealing 
    (saTour, saDist), saTime = runtime(simulatedAnnealing, cities, 15.0)
    results.append(("Simulated Annealing", saDist, saTime))

    #Genetic Algorithm
    (gaTour, gaDist), gaTime = runtime(geneticAlgorithm, cities, 15.0)
    results.append(("Genetic Algorithm", gaDist, gaTime))

    #Ant Colony Optimization
    (acoTour, acoDist), acoTime = runtime(antColony, cities, 15.0)
    results.append(("Ant Colony", acoDist, acoTime))

    # Print results and their accuracy if reference available
    print("\n--- Results ---")
    for name, dist, t in results:
        line = f"{name:25s}  dist={dist:10.2f}  time={t:8.4f}s"
        if referenceDistance is not None:
            acc = computationAccuracy(dist, referenceDistance)
            line += f"  acc={acc:6.2f}%"
        print(line)

    # Plot comparisons 
    plotComparisons(results, referenceDistance, title=f"{datasetName} - Algorithm Comparisons")

    #Selects the best toure based on the shortest distance
    best = min(
        [("NN and 2-Opt", optTour, optDist), ("SA", saTour, saDist), ("GA", gaTour, gaDist), ("ACO", acoTour,acoDist)],
        key=lambda x: x[2]
    )
    return results, best[1], best[2]



if __name__ == "__main__":
    main()
                    
    