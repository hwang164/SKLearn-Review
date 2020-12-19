import numpy as np
from sklearn.datasets import load_boston,load_wine,load_iris
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def Bostonhousing():

    # Linear model
    data=load_boston()
    model=LinearRegression()
    model.fit(data.data,data.target)

    # Fit result
    pre=model.predict(data.data)
    print("-"*80)
    print("Training MSE:",mean_squared_error(data.target,pre))
    print("Coefficients:",r2_score(data.target,pre))
    print("-"*80)

    # Display from greatest to least
    abscoef=np.abs(model.coef_)
    a=sorted(abscoef,reverse=True)
    index=[np.where(abscoef==i)[0][0] for i in a]
    name=data.feature_names[index]
    coef=np.round(model.coef_[index],4)
    coefabs=np.abs(coef)

    print("Name, Abscoef, Coefficients")
    print(np.array([name,coefabs,coef]).T)
    print("-"*80)


def elbow():
    wine,_= load_wine(return_X_y=True)
    iris,_= load_iris(return_X_y=True)
    
    seed=123
    wines=[]
    iriss=[]
    n=range(1,11)
    
    for i in n:
        cluster=KMeans(n_clusters=i,init="random",random_state=seed).fit(wine)
        wines.append(cluster.inertia_)
    for i in n:
        cluster=KMeans(n_clusters=i,init="random",random_state=seed).fit(iris)
        iriss.append(cluster.inertia_)

    plt.plot(n,wines,marker="o")
    plt.title("Wine")
    plt.xlabel("Populations")
    plt.ylabel("Sum of squared distance")
    plt.show()
    
    plt.plot(n,iriss,marker="o")
    plt.title("Iris")
    plt.xlabel("Populations")
    plt.ylabel("Sum of squared distance")
    plt.show()

    print('The decrease rate of sum of squared distances is moderate after the 3 populations, so 3 is the correct number.')


if __name__ == "__main__":
    Bostonhousing()
    elbow()
