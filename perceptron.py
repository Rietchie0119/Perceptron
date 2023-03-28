import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    ln = plt.plot(x1, x2)
    plt.pause(0.001)
    ln[0].remove()
    
# Sigmoid function to evaluate score for each points 
def sigmoid(score):
    return 1 / (1 + np.exp(-score))

def calculate_error(line_parameters, points, y):
    m = points.shape[0]
    p = sigmoid (points*line_parameters)# 행렬 곱 구하기 (각 점과 weights 곱하기) 후 sigmoid 구하기
    cross_entropy = -(np.log(p).T * y + np.log(1-p).T * (1-y))/m # 에러를 구하기 위한 식. 파란색이면 왼쪽, 빨간색이면 오른쪽 곱이 더해짐.
    return cross_entropy

# 최적의 선을 찾는 과정. 2000 번 반복함
def gradient_descent(line_parameters, points, y, alpha):
    m = points.shape[0]
    for i in range(2000):
        p = sigmoid(points*line_parameters)
        gradient = points.T * (p - y) * alpha / m # 알파값은 작은 값. 천천히 adjusting 해간다
        line_parameters -= gradient
        
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        # 그래프에 선을 그리기 위한 최솟값을 가지는 점 찾기
        x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
        x2 = -b / w2 + x1 * (-w1/w2)
        draw(x1, x2)
        print(calculate_error(line_parameters, points, y))

n_pts = 100 # 점 100개
np.random.seed(0) # 에러를 찾기 위한 시드값 고정
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12,2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6,2, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))

# x1 과 x2의 weight, 또는 계수를 정한다.

line_parameters = np.matrix(np.zeros(3)).T

y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1) # error을 계산하기 위한 y 값

_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color ='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color ='b')
gradient_descent(line_parameters, all_points, y , 0.06)
plt.show()
