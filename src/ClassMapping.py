import numpy as np

# The example where the soldier units are grouped into infantry, ranged, and mounted classes.

power_table = np.array([[350,200,800],
                        [500,300,700],
                        [550,350,650],
                        [850,500,200],
                        [700,600,300],
                        [350,700,400],
                        [400,900,550]],dtype=float)

soldier_map = {0:0,
               1:0,
               2:0,
               3:1,
               4:1,
               5:2,
               6:2}

army_combinations = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])

def getArmyRatios(army):
    ratio = np.zeros(3)
    i = np.where(army==1)[0][0]
    ratio[soldier_map[i]] = 1
    return ratio

def sumratios():
    sum = np.zeros(3)
    for r in army_combinations:
        sum += getArmyRatios(r)
    return sum



def getPayoff():
    size = len(soldier_map)
    payoff = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            payoff[i][j]= power_table[i][soldier_map[j]] - power_table[j][soldier_map[i]]
    p = payoff.astype(int)
    print(payoff.astype(int))









lr = 0.01
def update():
    global power_table
    powers = np.zeros(len(army_combinations))
    grad_matrix = np.zeros(power_table.shape)

    for i,atk in enumerate(army_combinations):
        dr = sumratios()
        print("soldier type ratios",dr)
        ap = np.outer(atk,dr)
        dp = np.outer(np.ones(7),getArmyRatios(atk))
        influence = ap-dp
        power = (power_table * influence).sum()
        powers[i] = power
        grad_matrix += influence * power


    print("influence matrix:\n",np.round(grad_matrix, decimals=2))
    print("powers:",np.round(powers, decimals=2))
    err = (powers**2).sum()
    print("err:",np.round(err, decimals=4))
    power_table -= lr*grad_matrix
    power_table[power_table<0]=1
    #print("modified power table:\n", np.round(power_table, decimals=0))
    print("modified power table:\n", power_table.astype(int))
    return err


p = getPayoff()


err = update()
while err>0.01:
    err = update()

getPayoff()

