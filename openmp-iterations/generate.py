from random import randint as ri

n = 200

m = [[0]*n for i in range(n)]
b = [0]*n

x = 1

def gen(m,b, n, fwithargs):
    for i in range(n):
        m[i][i] = str(fwithargs[0](*fwithargs[1:]))
        for j in range(i + 1, n):
            m[i][j] = str(fwithargs[0](*fwithargs[1:]))
            m[j][i] = m[i][j]

    for i in range(n):
        b[i] = str(fwithargs[0](*fwithargs[1:]))

def oneortwo():
    global x
    if x == 1:
        x = 2
    else:
        x = 1
    return x

gen(m, b, n, (ri, -20, 20))
# gen(m, b, n, (oneortwo,))

with open("input.txt", "w") as inp:
    inp.write(f"{n}\n")
    inp.writelines([' '.join(m[i])+'\n' for i in range(n)])
    inp.write(' '.join(b))