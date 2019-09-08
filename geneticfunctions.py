import numpy as np


def fitness(eq, pop):
    genfit = pop.dot(eq)    # Dot product of each instance of a pop by the equation to get vector pf fitness
    square = lambda i: i*i
    vector_square = np.vectorize(square)
    # Function to square fitness values
    # Makes large fitnesses bigger, and smaller ones smaller
    genfit = vector_square(genfit)
    total = np.sum(genfit)
    genfit = genfit/total   # Normalises fitness vector by diving each element by the sum of the whole array
    # print(genfit)
    return genfit


def crossover(parents, parentnum, childnum):
    children = []
    # array to store all the children
    # for each child
    crossover_num = int(parents.shape[1]/2)
    # crosses over at half the parent length
    for c in range(childnum):
        children.append([])
        # use normal array, convert it to a numpy array later because numpy append is weird
        select = np.random.choice(parentnum, 2, replace=False)  # selects a 2 different numbers <= parentnum
        # selects 2 non-equal parents
        parent1 = parents[select[0]]
        parent2 = parents[select[1]]
        # child is half 1 parent, half another
        children[c].extend(parent1[0:crossover_num])
        children[c].extend(parent2[crossover_num:])
    children = np.array(children)
    return children


# function to get fitness of population, and population then work out breeding parents

def nextgen(pop, genfit, parentnum):
    rows = np.random.choice(pop.shape[0], size=parentnum, replace=False, p=genfit)
    # randomly samples a number of individuals from population, based on their fitness
    parents = np.array([pop[i] for i in rows])
    # puts all these individuals in an array together
    return parents


def mutate(pop):
    # modifies a random gene in each individual
    length = pop.shape[1]
    # length of an individual
    for i in range(pop.shape[0]):
        index = np.random.randint(length)           # random weight for individual
        change = np.random.uniform(-1.0, 1.0, 1)    # random change for that weight
        pop[i][index] += change
    return pop
