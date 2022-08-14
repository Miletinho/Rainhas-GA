import itertools
from pickle import FALSE, TRUE
from statistics import mean
import numpy as np
import pandas as pd
from itertools import permutations 
#import itertools 
import sys
import random
import decimal
import math
import time


# def converte(individuals1,individuals2):
#     for chromo in individuals2:
#         i=0
#         while i<8:
#             print(chromo[i])
#         i=i+1
#     return True

def fitness(individuals):
    """
	returns 28 - <number of conflicts>
	to test for conflicts, we check for 
	 -> row conflicts
	 -> columnar conflicts
	 -> diagonal conflicts
	 
	The ideal case can yield upton 28 arrangements of non attacking pairs.
	for iteration 0 -> there are 7 non attacking queens
	for iteration 1 -> there are 6 no attacking queens ..... and so on 
	Therefore max fitness = 7 + 6+ 5+4 +3 +2 +1 = 28
	hence fitness val returned will be 28 - <number of clashes>
	"""
    totalFitness = []# o fitness de cada chromossomo
    for chromosome in individuals:
        clashes = 0
        columnDifference=0
        rowDifference=0
        chromoFit=0
        i=0
        while i < 8:
            j=0
            j = i+1
            while j<8:
                if int(chromosome[i],2) == int(chromosome[j],2):
                    clashes+=1
                columnDifference = abs(i - j)
                rowDifference = abs(int(chromosome[i],2) - int(chromosome[j],2))
                if (columnDifference == rowDifference):
                    clashes+=1
                j+=1
            i+=1
        chromoFit = 28 - clashes 
        totalFitness.append(chromoFit)
        
        # if chromoFit == 28:
        #     sys.exit("Solução:" + str(chromosome) + " -> Fit: " + str(chromoFit))
            
    return totalFitness # retorna o vetor com as devidas porcentagens de fitness

def toPermutation(chromosome):
    bool = [np.binary_repr(i, width=3) for i in range(0,8)]
    numbers = []
    duplicates = []
    for i in range(0,8):
        if numbers.count(chromosome[i]):
            duplicates.append(i)
        else:
            numbers.append(chromosome[i])

    dif = [x for x in bool if x not in numbers]

    for i in duplicates:
        x = dif.pop()
        chromosome[i] = x

    return chromosome

    
def mutation(chromosome, sigma):
    # randomly pick a gene to mutate from the chromosome
    # globalTal = 1/np.sqrt(8)
    globalTal = 1/np.sqrt(2*8)
    localTal = 1/np.sqrt(2*np.sqrt(8))
    gaussian = np.random.normal()
    newSigma = sigma * np.exp(globalTal*gaussian + localTal*gaussian)
    # newSigma = sigma * np.exp(globalTal*gaussian)

    signal = np.random.choice([-1,1])
    # print(newSigma)
    for i in range(8):
        c = math.ceil(int(chromosome[i], 2) + signal*newSigma*gaussian)
        if c > 7:
            chromosome[i] = '111'
        elif c < 0:
            chromosome[i] = '000'
        else:
            chromosome[i] = np.binary_repr(c, width=3)

    c = toPermutation(chromosome)
    # print(c)

    return chromosome

def populationMutation(newGeneration, pm, sigma):
    for i in range(0, len(newGeneration)):
        rand = random.uniform(0, 1)
        if (rand < pm):
            mutated_chromosome = mutation(newGeneration[i], sigma)
            newGeneration[i] = mutated_chromosome
    newGeneration = np.array(newGeneration)
    return newGeneration

def evaluation(newGeneration, fitnessTotal, individuals, n):
    individuals = np.concatenate((individuals,newGeneration), axis=0)
    dataframe = pd.DataFrame(individuals)
    novo_fitness = fitness(newGeneration)
    fitnessTotal = np.concatenate((fitnessTotal, novo_fitness),axis=None)
    dataframe["Fitness"] = fitnessTotal
    dataframe = dataframe.sort_values(by=["Fitness"], ascending=False, ignore_index=True)
    dataframe = dataframe.drop(columns=["Fitness"])
    newGeneration = dataframe.values.tolist()
    newGeneration = newGeneration[:n]
    newGeneration = np.array(newGeneration)
    fitnessTotal = fitness(newGeneration)
    return [newGeneration, fitnessTotal]


def find_solution(population_size, individuals, pc, pm, sigma):
    fitnessTotal = fitness(individuals)
    # fitness_array = percentFit(fitnessTotal)# adiciona os valores da porcentagem dos fitness a cada chromossomo

    newGeneration = populationMutation(individuals, pm, sigma)
    mutations = len(newGeneration)
    [newGeneration, newfitnessTotal] = evaluation(newGeneration, fitnessTotal, individuals, population_size)

    mutationSuccess = 0
    for i in range(len(newfitnessTotal)):
        if newfitnessTotal[i] > fitnessTotal[i]:
            mutationSuccess += 1

    return [newGeneration, newfitnessTotal, mutationSuccess, mutations]

def initialization(n):
    binary = ["".join(seq) for seq in itertools.product("01", repeat=3)]
    totalPermutations = list(permutations(binary))
    popIndex = np.random.randint(len(totalPermutations), size=n)

    initialPopulation = []
    for i in popIndex:
        initialPopulation.append(totalPermutations[i])

    initialPopulation = np.array(initialPopulation)

    return initialPopulation

def evaluateSigma(sigma, ps):
    c = np.random.uniform(0.8, 1)
    if ps > 1/5:
        return sigma/c  # ampliar a busca -> exploration
    elif ps < 1/5:
        return sigma*c  # concentrar a busca -> explotation

    return sigma

def main(totalConvergence, nExecutions = 10000):
    # number of inidviduals
    n = 100
    # number of genes in each individual
    genes = 8
    # crossover probability
    pc = 0.9
    # mutation probability
    pm = 0.4

    sigma = 1

    allGenerationsFitness = []

    individuals = initialization(n)
    # surround the following function call in a while loop which breaks once a solution is found
    gen = 0
    while(gen < nExecutions):
        #print("Generation: " + str(gen) + "\n")
        numberOfMutations = 0
        numberOfMutationSuccess = 0
        [new_gen, fit, mutationSuccess, mutations] = find_solution(n, individuals, pc, pm, sigma)
        allGenerationsFitness = np.concatenate((allGenerationsFitness, fit), axis=None)
        numberOfMutationSuccess += mutationSuccess
        numberOfMutations += mutations
        # ---- 1/5 da regra de sucesso:
        #   -> se mais de 1/5 das mutações levar a uma melhora, a força da mutação é aumentada (sigma=sigma/c), se == 1/5, mantém (sigma = sigma), se não é diminuída (sigma = sigma*c).
        # ps é a % de mutações com sucesso
        # if gen % 5 == 0:
        #     ps = numberOfMutationSuccess/numberOfMutations
        #     sigma = evaluateSigma(sigma, ps)
            

        if 28 in fit:
            counter = fit.count(28)
            if totalConvergence: 
                if counter == 100:
                    print("Number of Generations to reach total convergence: ", gen+1)
                    return [gen, mean(allGenerationsFitness), counter] 
            else:
                print("Number of Generations to reach individual convergence: ", gen+1)
                return [gen, mean(allGenerationsFitness), counter] 

        individuals = new_gen
        gen = gen + 1
 
    return [gen, mean(allGenerationsFitness), 0] # quantas gerações rodou, a média de fitness de todas as gerações, quantos indivíduos convergiram

def evaluateExecutions(allGen, allFitness, counter, execTime):
    meanGen = np.average(allGen)
    stdGen = np.std(allGen)
    # print(allGen)
    meanFitness = np.average(allFitness)
    stdFitness = np.std(allFitness)
    nConvergence = sum(counter)
    meanConvergence = np.average(counter)
    meanExecTime = np.average(execTime)
    return [meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence, meanExecTime]

def getTime(startTime):
    execTime = round(time.time() - startTime, 3)
    if execTime > 60:
        print("Tempo de execução: ", round(execTime/60, 3), " minutos")
    else:
        print("Tempo de execução: ", execTime, " segundos")
    return execTime

def printEvaluation(meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence, meanExecTime):
    print("Em que iteração o algoritmo convergiu, em média: ", round(meanGen, 3))
    print("Desvio Padrão de em quantas iterações o algoritmo convergiu: ", round(stdGen, 3))
    print("Fitness médio alcançado nas 30 execuções : ", round(meanFitness, 3))
    print("Desvio padrão dos Fitness alcançados nas 30 execuções: ", round(stdFitness, 3))
    print("Em quantas execuções o algoritmo convergiu: ", str(min(nConvergence, 30)) + "/30")
    print("Número de indivíduos que convergiram: ", nConvergence)
    print("Número de indivíduos que convergiram por execução, em média: ", round(meanConvergence, 3))
    print("Tempo médio de execução das 30 execuções: ", round(meanExecTime, 3), " segundos")

if __name__ == '__main__':
    startTime = time.time()
    allFitness = []
    allGen = []
    counter = []
    execTime = []
    for i in range(30):
        print("Execução", i)
        stTime = round(time.time(), 3)
        iteration = main(totalConvergence = False)
        allGen.append(iteration[0])
        allFitness.append(iteration[1])
        counter.append(iteration[2])
        execTime.append(getTime(stTime))

    
    [meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence, meanExecTime] = evaluateExecutions(allGen, allFitness, counter, execTime)


    print("Parte 2-ES:")
    printEvaluation(meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence, meanExecTime)
    getTime(startTime)
    # - Análise adicional: Quantas iterações são necessárias para toda a população convergir?
    print("Quantas iterações são necessárias para toda a população convergir?")
    iteration = main(totalConvergence = True, nExecutions  = 70000)
    print(iteration[1])
    getTime(startTime)