import itertools
from pickle import FALSE, TRUE
from statistics import mean
import numpy as np
import pandas as pd
from itertools import permutations 
import sys
import random
import decimal
import time

def fitness(individuals):
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
                 
    return totalFitness # retorna o vetor com as devidas porcentagens de fitness


def percentFit(fitness_vetor):
    addedValues = sum(fitness_vetor) #pega o valor da soma total do fitness da geração 
    percentFit = []
    for i in fitness_vetor:
        percentFit.append(round(((i / addedValues)), 4))
    return percentFit 

def getFit(parent):
    return parent[1]

def ranking(totalFitness, individuals):
    pair = []
    parentChoice = np.random.choice(range(0, len(individuals)), 5)


    parentFitness = []
    for i in parentChoice:
        parentFitness.append([i, totalFitness[i]])  

    parentFitness.sort(key=getFit)
    
    parentFitness = parentFitness[:2]
    for i in parentFitness:
        pair.append(i[0])
    
    return pair
        
def crossover(parent1, parent2):
    #gerando os index onde os parents precisam trocar os genes 
    crossPoint = random.randint(1,7)
    
    #criando od filhos que terão pedaços de cada pai
    filho1 = np.append(parent1[:crossPoint], parent2[crossPoint:len(parent2)])
    filho2 = np.append(parent2[:crossPoint], parent1[crossPoint:len(parent1)])
    
    filhos = np.array([filho1,filho2])
    return filhos   

def mutacao(chromosome):
    # randomly pick a gene to mutate from the chromosome
    mutated_gene1 = random.randrange(0, 8)
    mutated_gene2 = random.randrange(0, 8)
    while mutated_gene1==mutated_gene2:
        mutated_gene2=random.randrange(0, 8)
    
    # change its value to a random column
    a = chromosome[mutated_gene1] 
    chromosome[mutated_gene1] = chromosome[mutated_gene2]
    chromosome[mutated_gene2] = a
    return chromosome

def selection(population_size, fitness_Total, individuals):
    parents = [] # seleção de parents
    for i in range(0, int(population_size / 2)):
        selected = ranking(fitness_Total, individuals)
        # add the selected values to the new_generation array   
        parents.append(individuals[selected[0]])
        parents.append(individuals[selected[1]])
    return parents
    
def populationCrossover(population_size, pc, parents):
    newGeneration = []
    #randomicamente gerando um número entre 0 e 1, se esse número é menor do que a probabilidade de crossover 
    for i in range(0, population_size, 2):
        rand = random.uniform(0, 1)
        if (rand < pc):
            children = crossover(parents[i], parents[i + 1])
            # troca de de lugar o filho da nova geração com a versão do crossover
            newGeneration.append(children[0])
            newGeneration.append(children[1])
    return newGeneration

def populationMutation(newGeneration, pm):
    for i in range(0, len(newGeneration)):
        rand = random.uniform(0, 1)
        if (rand < pm):
            mutated_chromosome = mutacao(newGeneration[i])
            newGeneration[i] = mutated_chromosome
    newGeneration = np.array(newGeneration)
    return newGeneration

def evaluation(newGeneration, fitness_Total, individuals, n):
    individuals = np.concatenate((individuals,newGeneration), axis=0)
    dataframe = pd.DataFrame(individuals)
    novo_fitness = fitness(newGeneration)
    fitness_Total = np.concatenate((fitness_Total, novo_fitness),axis=None)
    dataframe["Fitness"] = fitness_Total
    dataframe = dataframe.sort_values(by=["Fitness"], ascending=False, ignore_index=True)
    dataframe = dataframe.drop(columns=["Fitness"])
    newGeneration = dataframe.values.tolist()
    newGeneration = newGeneration[:n]
    newGeneration = np.array(newGeneration)
    fitness_Total = fitness(newGeneration)
    return [newGeneration, fitness_Total]


def find_solution(population_size, individuals, pc, pm):
    fitness_Total = fitness(individuals)
    fitness_array = percentFit(fitness_Total)# adiciona os valores da porcentagem dos fitness a cada chromossomo
    #se todas as rainhas em um chromossomo não interfere com a outra _. encontramos a solução
    #ai criamos uma nova geração.
    parents = selection(population_size, fitness_Total, individuals)
    newGeneration = populationCrossover(population_size, pc, parents)
    newGeneration = populationMutation(newGeneration, pm)
    [newGeneration, fitness_Total] = evaluation(newGeneration, fitness_Total, individuals, population_size)
    
    return [newGeneration, fitness_Total]

def initialization(n):
    binary = ["".join(seq) for seq in itertools.product("01", repeat=3)]
    totalPermutations = list(permutations(binary))
    popIndex = np.random.randint(len(totalPermutations), size=n)

    initialPopulation = []
    for i in popIndex:
        initialPopulation.append(totalPermutations[i])

    initialPopulation = np.array(initialPopulation)

    return initialPopulation

def main(totalConvergence, nExecutions = 10000):
    # number of inidviduals
    n = 100
    # number of genes in each individual
    genes = 8
    # crossover probability
    pc = 0.9
    # mutation probability
    pm = 0.4

    allGenerationsFitness = []

    individuals = initialization(n)
    # surround the following function call in a while loop which breaks once a solution is found
    gen = 0
    while(gen < nExecutions):
        #print("Generation: " + str(gen) + "\n")
        [new_gen, fit] = find_solution(n, individuals, pc, pm)
        allGenerationsFitness = np.concatenate((allGenerationsFitness, fit), axis=None)
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
    print("Parte 1:")
    printEvaluation(meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence, meanExecTime)
    getTime(startTime)
#     # - Análise adicional: Quantas iterações são necessárias para toda a população convergir?
#     print("Quantas iterações são necessárias para toda a população convergir?")
#     iteration = main(totalConvergence = True, nExecutions  = 70000)
#     print(iteration[1])
#     getTime(startTime)
