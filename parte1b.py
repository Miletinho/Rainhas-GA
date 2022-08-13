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


def percentFit(fitness_vetor):
    addedValues = sum(fitness_vetor) #pega o valor da soma total do fitness da geração 
    percentFit = []
    for i in fitness_vetor:
        percentFit.append(round(((i / addedValues)), 4))
    return percentFit 

def roulette(chromosomeFit):
    par = []
    whellvalor = []
    fitAcumulativo= 0
    # acha o valor total do fitness
    fitAcumulativo = sum(chromosomeFit);     
    # encontra a porcentagem de cada valor comparado com o total  
    for elemento in chromosomeFit:
        whellvalor.append((elemento/fitAcumulativo))
    
    rodarLoop = True 
    ParentOneIndex =-1 #indica o endereço para um parente(faz com que seja assegurado que o primeiro parente é diferente do segundo)
    
    while(rodarLoop):
        #Gerando um número randomico
        randomInt = random.randint(1,100)
        start = 0 # o início da fatia na roleta por uma percentagem
        end = 0 # o fim da fatia na roleta por uma porcentagem 
        index = 0 # indica o elemento na roleta 
        for elemento in whellvalor:
            end = round((elemento*100+ end),2)
            
            # se o numero random esta entre o start e o and é adicionado ao array de par após passar pela condição if 
            if(start <= randomInt and end >= randomInt):
                if (ParentOneIndex == -1):
                    par.append(index)
                    ParentOneIndex = index
                    break
                elif (index != ParentOneIndex):
                   # print("parentOne: " + str(parentOneIndex))
                    par.append(index)
                    rodarLoop = False
                    break
            index = index + 1
            start = end
        ParentOneIndex = index
    return par

        
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

def selection(population_size, fitness_array, individuals):
    parents = [] # seleção de parents
    for i in range(0, int(population_size / 2)):
        selected = roulette(fitness_array)
        while len(selected)<2:
            selected = roulette(fitness_array)   
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
    parents = selection(population_size, fitness_array, individuals)
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

def main():
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
    while(gen < 10000):
        #print("Generation: " + str(gen) + "\n")
        [new_gen, fit] = find_solution(n, individuals, pc, pm)
        allGenerationsFitness = np.concatenate((allGenerationsFitness, fit), axis=None)
        if 28 in fit:
            print(fit)
            counter = fit.count(28)
            print("ok")
            return [gen, mean(allGenerationsFitness), counter] 
        chromosomes = new_gen
        gen = gen + 1
 
    return [gen, mean(allGenerationsFitness), 0] # quantas gerações rodou, a média de fitness de todas as gerações, quantos indivíduos convergiram

def evaluateExecutions(allGen, allFitness, counter):
    meanGen = np.average(allGen)
    stdGen = np.std(allGen)
    print(allGen)
    meanFitness = np.average(allFitness)
    stdFitness = np.std(allFitness)
    nConvergence = sum(counter)
    meanConvergence = np.average(counter)
    return [meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence]


if __name__ == '__main__':
    # iterations = []
    allFitness = []
    allGen = []
    counter = []
    for i in range(30):
        print("Iteração",i)
        iteration = main()
        allGen.append(iteration[0])
        allFitness.append(iteration[1])
        counter.append(iteration[2])

    
    [meanGen, stdGen, nConvergence, meanFitness, stdFitness, meanConvergence] = evaluateExecutions(allGen, allFitness, counter)

    print("meanGen: ", meanGen)
    print("stdGen: ", stdGen)
    print("meanFitness: ", meanFitness)
    print("stdFitness: ", stdFitness)
    print("nConvergence:", nConvergence)
    print("meanConvergence: ", meanConvergence)
    