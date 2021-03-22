from os import uname, environ
environ["OPENBLAS_NUM_THREADS"] = "1"
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
environ["CUDA_VISIBLE_DEVICES"] = "-1"
from time import time, sleep
from numpy import mean, std, sum, matmul, transpose, array_split
from numpy.random import seed, random, choice
from numpy.linalg import inv
from pandas import DataFrame
from hashlib import sha512
from sklearn.ensemble import RandomForestClassifier
from psutil import cpu_count, cpu_freq, virtual_memory, swap_memory
from multiprocessing import Pool
from functools import reduce
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


seed(1102)


def pprint(m, l):
    print(
        m + \
        ":\t{t:.10f}\t{m:.10f}\t{s:.10f}"\
        .format(
            t=sum(l), 
            m=mean(l), 
            s=std(l)
        )
    )


def compute_stuff(x): 
    sleep(0.1)
    output = 0.0
    for j in range(100):
        for i in range(100):
            for k in range(10):
                output = output + x**(1.0/(j+1.0))
                output = output + x**(1.0/(-i+0.5))
                output = output + x**(k/1.7)
    return output


def helper_compute_stuff(l):
    return [compute_stuff(x) for x in l]


u = uname()
print("\n\n### SYSTEM INFORMATION ###\n")

print("sysname:\t", u.sysname)
print("nodename:\t", u.nodename)
print("release:\t", u.release)
print("version:\t", u.version)
print("machine:\t", u.machine)
print("CPUs (log):\t", cpu_count(logical=True))
print("CPUs (phy):\t", cpu_count(logical=False))
print("CPUs max freq:\t", cpu_freq(percpu=False).max)
print("RAM:\t\t", virtual_memory().total)
print("SWAP:\t\t", swap_memory().total)

print("\n\n### SPEED TESTS ###\n")

r = random(2000000)
l = []
for j in range(1000):
    start = time()
    x = r.copy()
    del x
    l.append(time()-start)
pprint(m="Memory Read/Write", l=l)

l = []
for j in range(1000000):
    e = str(j).encode()
    start = time()
    h = sha512(e)
    l.append(time()-start)
    del h
pprint(m="Hash Generation", l=l)

l = []
for j in range(1000):
    start = time()
    M = random((500, 500))
    l.append(time()-start)
    del M
pprint(m="Matrix Generation", l=l)

l = []
for j in range(1000):
    M = random((500, 500))
    N = random((500, 500))
    start = time()
    O = matmul(M, N)
    l.append(time()-start)
pprint(m="Matrix Multiplication", l=l)

l = []
for j in range(1000):
    M = random((300, 300))
    M = matmul(M, transpose(M))
    start = time()
    O = inv(M)
    l.append(time()-start)
pprint(m="Matrix Inversion", l=l)

l = []
for j in range(1000):
    M = random((500, 8))
    start = time()
    df = DataFrame(M)
    df.columns = ["A", "B", "C", "D", "E", "F", "G", "H"]
    N = df.copy()
    N["B"] = (N["B"] * 5).astype(int)
    N["C"] = (N["C"] * 5).astype(int)
    O = N.loc[(0.1 < N.A) & (N.A < 0.9), :].groupby(["B", "C"]).agg({"F": "sum", "G": "min", "H": "max"})
    l.append(time()-start)
pprint(m="DataFrame Operations", l=l)

l = []
outputs = []
for j in range(1000):
    x = random()
    start = time()
    outputs.append(compute_stuff(x))
    l.append(time()-start)
del outputs
pprint(m="Parallel Exec w=1", l=l)

l = []
inputs = random(size=1000)
start = time()
pool = Pool(processes=2)
outputs = pool.map(helper_compute_stuff, array_split(inputs, 2))
outputs = reduce(lambda x,y: x+y, outputs)
pool.close()
l.append(time()-start)
l = l + [0] * 999
pprint(m="Parallel Exec w=2", l=l)

l = []
inputs = random(size=1000)
start = time()
pool = Pool(processes=4)
outputs = pool.map(helper_compute_stuff, array_split(inputs, 4))
outputs = reduce(lambda x,y: x+y, outputs)
pool.close()
l.append(time()-start)
l = l + [0] * 999
pprint(m="Parallel Exec w=4", l=l)

l = []
inputs = random(size=1000)
start = time()
pool = Pool(processes=8)
outputs = pool.map(helper_compute_stuff, array_split(inputs, 8))
outputs = reduce(lambda x,y: x+y, outputs)
pool.close()
l.append(time()-start)
l = l + [0] * 999
pprint(m="Parallel Exec w=8", l=l)

l = []
inputs = random(size=1000)
start = time()
pool = Pool(processes=16)
outputs = pool.map(helper_compute_stuff, array_split(inputs, 16))
outputs = reduce(lambda x,y: x+y, outputs)
pool.close()
l.append(time()-start)
l = l + [0] * 999
pprint(m="Parallel Exec w=16", l=l)

l = []
for j in range(100):
    X = random((1000, 30))
    Y = choice([0, 1], size=1000, replace=True)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=1)
    start = time()
    clf = clf.fit(X, Y)
    l.append(time()-start)
pprint(m="Random Forest jobs=1", l=l)

l = []
for j in range(100):
    X = random((1000, 30))
    Y = choice([0, 1], size=1000, replace=True)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    start = time()
    clf = clf.fit(X, Y)
    l.append(time()-start)
pprint(m="Random Forest jobs=2", l=l)

l = []
for j in range(100):
    X = random((1000, 30))
    Y = choice([0, 1], size=1000, replace=True)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=4)
    start = time()
    clf = clf.fit(X, Y)
    l.append(time()-start)
pprint(m="Random Forest jobs=4", l=l)

l = []
for j in range(100):
    X = random((1000, 30))
    Y = choice([0, 1], size=1000, replace=True)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=8)
    start = time()
    clf = clf.fit(X, Y)
    l.append(time()-start)
pprint(m="Random Forest jobs=8", l=l)

l = []
for j in range(100):
    X = random((1000, 30))
    Y = choice([0, 1], size=1000, replace=True)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=16)
    start = time()
    clf = clf.fit(X, Y)
    l.append(time()-start)
pprint(m="Random Forest jobs=16", l=l)

l = []
inputs = Input(shape=(28, 28, 3))
x = Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs)
x = MaxPool2D(pool_size=(2,2))(x)
x= Flatten()(x)
x = Dense(units=16, activation="relu")(x)
x = Dense(units=8, activation="relu")(x)
outputs = Dense(units=2, activation="softmax")(x)
classifier = Model(inputs=inputs, outputs=outputs)
classifier.compile(optimizer=RMSprop(lr=0.05), loss='categorical_crossentropy')
for j in range(5):
    X = random((1000, 28, 28, 3))
    Y = to_categorical(choice([0, 1], size=1000, replace=True), 2)
    start = time()
    classifier.fit(X, Y, batch_size=64, epochs=100, verbose=0)
    l.append(time()-start)
pprint(m="Neural Network CPU", l=l)

environ["CUDA_VISIBLE_DEVICES"] = "0"

l = []
inputs = Input(shape=(28, 28, 3))
x = Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs)
x = MaxPool2D(pool_size=(2,2))(x)
x= Flatten()(x)
x = Dense(units=16, activation="relu")(x)
x = Dense(units=8, activation="relu")(x)
outputs = Dense(units=2, activation="softmax")(x)
classifier = Model(inputs=inputs, outputs=outputs)
classifier.compile(optimizer=RMSprop(lr=0.05), loss='categorical_crossentropy')
for j in range(5):
    X = random((1000, 28, 28, 3))
    Y = to_categorical(choice([0, 1], size=1000, replace=True), 2)
    start = time()
    classifier.fit(X, Y, batch_size=64, epochs=100, verbose=0)
    l.append(time()-start)
pprint(m="Neural Network GPU", l=l)