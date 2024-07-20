import numpy as np

# Probability model
'''
  r p s
r 1 0 2
p 2 1 0
s 0 2 1    
'''
power_table = np.array([  [1, 0, 4],
                                [2, 1, 20],
                                [0, 5, 1]], dtype=float)

dr = [1.0/3, 1.0/3, 1.0/3]
#dr = [1.0/4, 1.0/2, 1.0/4]

army_combinations = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])

lr = 0.01
def update():
    global power_table
    powers = np.zeros(len(army_combinations))
    grad_matrix = np.zeros(power_table.shape)

    for i,atk in enumerate(army_combinations):
        ap = np.outer(atk,dr)
        dp = np.outer(dr,atk)
        influence = ap-dp
        power = (power_table * influence).sum()
        powers[i] = power
        grad_matrix += influence * power


    print("influence matrix:\n",np.round(grad_matrix, decimals=2))
    print("powers:",np.round(powers, decimals=2))
    err = (powers**2).sum()
    print("err:",np.round(err, decimals=4))
    power_table -= lr*grad_matrix
    print("modified power table:\n", np.round(power_table, decimals=2))
    return err


err = update()
while err>0.01:
    err = update()