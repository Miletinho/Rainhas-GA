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


# def converte(chromes1,chromes2):
#     for chromo in chromes2:
#         i=0
#         while i<8:
#             print(chromo[i])
#         i=i+1
#     return True
def fitness(chromes):
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
    for chromosome in chromes:
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
    oi= True  
    for i in range(1,len(chromes)):
        print(chromes[0],chromes[i])
        if(all(chromes[0]==chromes[i])!=True):
            oi=False
            break
    if (mean(totalFitness) == 28) :
        #print(chromes)
        # print(totalFitness)
        sys.exit("Solution has been found: " + str(chromosome))
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

def crossover(parente1,parente2):
    #genrado os index onde os parentes precisam trocar os genes 
    crossPoint = random.randint(1,7)
    
    #criando od filhos que terão metade de cada pai
    filho1 = np.append(parente1[:crossPoint], parente2[crossPoint:len(parente2)])
    filho2 = np.append(parente2[:crossPoint], parente1[crossPoint:len(parente1)])
    
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

        
def find_solution(population_size, chromes, probabilidade_cross, probabilidade_mut):
    fitness_Total = fitness(chromes)
    fitness_array = percentFit(fitness_Total)# adiciona os valores da porcentagem dos fitness a cada chromossomo
    #se todas as rainhas em um chromossomo não interfere com a outra _. encontramos a solução
    #ai criamos uma nova geração.
    pais = [] # seleção de pais atraves da roleta
    nova_geracao= []
    for i in range(0, int(population_size / 2)):
        selected = roulette(fitness_array)
        while len(selected)<2:
            selected = roulette(fitness_array)
        # add the selected values to the new_generation array   
        pais.append( chromes[selected[0]])
        pais.append( chromes[selected[1]])
    #randomicamente gerando um número entre 0 e 1, se esse número é menor do que a probabilidade de crossover 
    for i in range(0, population_size, 2):
        rand = random.uniform(0, 1)
        if (rand < probabilidade_cross):
            children = crossover(pais[i], pais[i + 1])
            # troca de de lugar o filho da nova geração com a versão do crossover
            nova_geracao.append(children[0])
            nova_geracao.append(children[1])
    #randomicamente gerando um número entre 0 e 1, se esse número é menor do que a probabilidade de crossover        
    for i in range(0, len(nova_geracao)):
        rand = random.uniform(0, 1)
        if (rand < probabilidade_mut):
            mutated_chromosome = mutacao(nova_geracao[i])
            nova_geracao[i] = mutated_chromosome
    
    nova_geracao= np.array(nova_geracao)
    chromes = np.concatenate((chromes,nova_geracao),axis=0)
    dataframe=pd.DataFrame(chromes)
    novo_fitness = fitness(nova_geracao)
    fitness_Total = np.concatenate((fitness_Total,novo_fitness),axis=None)
    dataframe["Fitness"]=fitness_Total
    dataframe = dataframe.sort_values(by=["Fitness"],ascending=False,ignore_index=True)
    dataframe = dataframe.drop(columns=["Fitness"])
    nova_geracao = dataframe.values.tolist()
    nova_geracao= nova_geracao[:n]
    nova_geracao = np.array(nova_geracao)
    return nova_geracao
        
if __name__ == '__main__':
    # each chromosome is comprised of eight genes, each of which corresponding to a column number
    
    # Declaring Variables
    # number of chromosomes
    n = 100
    # number of genes in each chromosome
    genes = 8
    # crossover probability
    pc = 0.9
    # mutation probability
    pm = 0.4

    # randomly generate n chromosomes, organize in an array
    binarios= ["".join(seq) for seq in itertools.product("01", repeat=3)]
    totalPermutations = list(permutations(binarios))
    popIndex = np.random.randint(len(totalPermutations), size=n)
    initialPopulation = []
    for i in popIndex:
        initialPopulation.append(totalPermutations[i])
    chromosomes = np.array(initialPopulation)

    # surround the following function call in a while loop which breaks once a solution is found
    gen = 0
    while(True):
        print("Generation: " + str(gen) + "\n")
        new_gen = find_solution(n,chromosomes, pc, pm)
        chromosomes = new_gen
        gen = gen + 1