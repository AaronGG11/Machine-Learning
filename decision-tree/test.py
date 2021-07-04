# decision trees -> https://www.python-course.eu/Decision_Trees.php
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

arbol = DecisionTreeClassifier()
arbol.fit(x_train, y_train)
arbol.score(x_test, y_test)
arbol.score(x_train, y_train)

export_graphviz(arbol, out_file='arbol.dot', class_names=iris.target_names, feature_names=iris.feature_names, impurity=False, filled=True)

with open('arbol.dot') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)