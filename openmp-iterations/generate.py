from random import random

n = 1000

m = [[0]*n for i in range(n)]
b = [0]*n

def gen(m,b, n, fwithargs):
    for i in range(n):
        m[i][i] = str(fwithargs[0](*fwithargs[1:]) + n)
        for j in range(i + 1, n):
            m[i][j] = str(fwithargs[0](*fwithargs[1:]))
            m[j][i] = m[i][j]

    for i in range(n):
        b[i] = str(fwithargs[0](*fwithargs[1:]))

# gen(m, b, n, (ri, -20, 20))
gen(m, b, n, (random,))

with open("input.txt", "w") as inp:
    inp.write(f"{n}\n")
    inp.writelines([' '.join(m[i])+'\n' for i in range(n)])
    inp.write(' '.join(b))
