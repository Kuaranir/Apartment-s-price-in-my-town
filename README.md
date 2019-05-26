# Apartment-s-price-in-my-town
Predicting apt. price in my town
import numpy as np
import pandas as pd
data = pd.read_csv('AD.csv')
data.head()
X, y = data.drop('Стоимость', axis=1), data['Стоимость']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=0, max_depth=2, learning_rate=0.01).fit(X_train, y_train)
print('Точность на обучающем наборе: {:.3f}'.format(gbr.score(X_train, y_train)))
print('Точность на тестовом наборе: {:.3f}'.format(gbr.score(X_test, y_test)))
flat_new = np.array([[
    int(input('Введите площадь квартиры в м^2: ')),
    int(input('Количество комнат? ')),
    int(input('Этаж: ')),
    int(input('Расстояние до центра города в метрах: ')),
    int(input('Расстояние до ближайшей остановки в метрах: ')),
    int(input('Расстояние до ближайшего магазина в метрах: ')),
    int(input('Новый дом? да - 1, нет - 0: ')),
    int(input('Есть ли ремонт? да - 1, нет - 0: '))
]])
prediction = gbr.predict(flat_new)
print('Спрогнозированная цена: {} рублей'.format(int(prediction[0])))
