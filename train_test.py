import numpy
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3,1,100)    #### x shows the number of mins to make a purchase
y = numpy.random.normal(150,40,100)/x   #### y shows the amount of money spent

###### Training 80%

train_x = x[:80]
train_y = y[:80]

###### Testing 20%

test_x = x[80:]
test_y = y[80:]

###### Polynomial regression

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
myline = numpy.linspace(0,6,100)

###### Check R2 or know how it is related x, y axis, 0 is no, 1 is good enough relation(predictible)

r2_train = r2_score(train_y, mymodel(train_x))
r2_test = r2_score(test_y, mymodel(test_x))


# print(r2_train)
# print(r2_test)
# plt.scatter(train_x, train_y)
# plt.plot(myline,mymodel(myline))
# plt.show()

# plt.scatter(test_x, test_y)
# plt.show()

##### Prediction test
print(mymodel(5))