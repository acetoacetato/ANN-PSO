import pandas as pd




def main(): 
    df = pd.read_csv("./KDDTrain+_20Percent.txt", header=None)

    # Transformamos la clase a -1 y 1
    df[41] = df[41].apply(lambda y: 1 if y == "normal" else -1)
    Y = df[41]

    # Transformamos las variables categóricas a numéricas
    for i in [1, 2, 3]:
        df[i] = df[i].astype("category")
        df[i] = df[i].cat.codes

    # Normalizamos las variables numéricas
    for i in (list(range(6, 40)) + [0, 42]):
        df[i]=((df[i]-df[i].min())/(df[i].max()-df[i].min()))*1

    # Dejamos la categoría al último
    df[[41, 42]] = df[[42, 41]]
    df.to_csv("train.csv", header=None)
    df = df.dropna(index=0)
    df.head()


if "__name__" == "__main__":
    main()