import pandas as pd

##En vista de que sigue sin crearme un csv el preproceso.py, cree este solo para crear el csv del test y el train.

df = pd.read_csv(r"C:\Users\marce\Desktop\ANN-PSO-master\KDDTest+.txt", header=None)
df.head()

df[41] = df[41].apply(lambda y: 1 if y == "normal" else -1)
Y = df[41]

for i in [1, 2, 3]:
    df[i] = df[i].astype("category")
    df[i] = df[i].cat.codes

#for i in (list(range(4, 40)) + [0, 42]):
#    df[i]=((df[i]-df[i].min())/(df[i].max()-df[i].min()))*1

# Dejamos la categoría al último
df[[41, 42]] = df[[42, 41]]
df.to_csv(r"C:\Users\marce\Desktop\ANN-PSO-master\test.csv", header=None, index=False)
df = df.dropna(axis=0)
df.drop(42, axis=1)
df.head()

