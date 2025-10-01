import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

class MGUA:
    def __init__(self, max_models_per_level=5):
        self.max_models_per_level = max_models_per_level
        self.best_model = None
        self.all_selections = []
    
    def least_squares(self, X, y):
        # Добавляем столбец единиц для свободного члена a0
        X_extended = np.c_[np.ones(X.shape[0]), X]
        
        # Вычисляем коэффициенты по формуле: a = (X^T * X)^(-1) * X^T * y
        coeffs, _, _, _ = np.linalg.lstsq(X_extended, y, rcond=None)
        
        return coeffs
    
    def predict(self, X, coeffs):
        # y = a0 + a1*x1 + a2*x2
        X_extended = np.c_[np.ones(X.shape[0]), X]
        y_pred = X_extended @ coeffs  # Матричное умножение
        return y_pred
    
    def calculate_mse(self, y_true, y_pred):
        # MSE = (1/n) * Σ(y_true - y_pred)²
        return np.mean((y_true - y_pred) ** 2)
    
    def calculate_mape(self, y_true, y_pred):
        # MAPE = (100/n) * Σ||(y_true - y_pred) / y_true||
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def fit(self, X_train, y_train, X_valid, y_valid):
        n_features = X_train.shape[1]
        models = []
        
        print("ПЕРВЫЙ РЯД СЕЛЕКЦИИ")
        print()
        print(f"Количество признаков: {n_features}")
        print(f"Обучающая выборка: {X_train.shape[0]} наблюдений")
        print(f"Проверочная выборка: {X_valid.shape[0]} наблюдений")
        
        feature_combinations = list(combinations(range(n_features), 2))
        print(f"Количество попарных комбинаций: {len(feature_combinations)}")
        print()
        
        # Для каждой комбинации строим модель
        for idx, (i, j) in enumerate(feature_combinations, 1):
            X_train_pair = X_train[:, [i, j]]
          
            coeffs = self.least_squares(X_train_pair, y_train)
            
            # Оцениваем модель на проверочной выборке
            X_valid_pair = X_valid[:, [i, j]]
            y_pred = self.predict(X_valid_pair, coeffs)
            mse = self.calculate_mse(y_valid, y_pred)
            
            models.append({
                'features': (i, j), 
                'feature_names': (f'x{i+1}', f'x{j+1}'),
                'coeffs': coeffs, 
                'mse': mse
            })
            
            print(f"Модель {idx}: x{i+1}, x{j+1} -> MSE = {mse:.6f}")
        
        models.sort(key=lambda x: x['mse'])
        best_models = models[:self.max_models_per_level]
        
        print()
        print(f"Отобрано {len(best_models)} лучших моделей:")
        for idx, model in enumerate(best_models, 1):
            print(f"  {idx}. {model['feature_names']} - MSE: {model['mse']:.6f}")
        
        self.all_selections.append({
            'iteration': 1,
            'models': best_models,
            'best_mse': best_models[0]['mse']
        })
        
        self.best_model = best_models[0]
        
        print()
        print("ЛУЧШАЯ МОДЕЛЬ:")
        print(f"  Признаки: {self.best_model['feature_names']}")
        print(f"  Уравнение: y = {self.best_model['coeffs'][0]:.4f} + "
              f"{self.best_model['coeffs'][1]:.4f}*{self.best_model['feature_names'][0]} + "
              f"{self.best_model['coeffs'][2]:.4f}*{self.best_model['feature_names'][1]}")
        print(f"  MSE: {self.best_model['mse']:.6f}")
        print()
    
    def predict_best(self, X):
        if self.best_model is None:
            raise ValueError("Модель не обучена! Вызовите метод fit().")
        
        i, j = self.best_model['features']
        X_pair = X[:, [i, j]]

        return self.predict(X_pair, self.best_model['coeffs'])

def main():
    # Исходные данные из таблицы 1
    # Столбцы: y, x1, x2, x3, x4
    data = np.array([
        [0.904, 75.5, 25.2, 3343, 77], [0.922, 78.5, 21.8, 3001, 78.2],
        [0.763, 78.4, 25.7, 3101, 68], [0.923, 77.7, 17.8, 3543, 77.2],
        [0.918, 84.4, 15.9, 3237, 77.2], [0.906, 75.9, 22.4, 3330, 77.2],
        [0.905, 76, 20.6, 3808, 75.7], [0.545, 67.5, 25.2, 2415, 62.6],
        [0.894, 78.2, 20.7, 3295, 78], [0.9, 78.1, 17.5, 3504, 78.2],
        [0.932, 78.6, 19.7, 30565, 79], [0.74, 84, 18.5, 3007, 67.6],
        [0.701, 59.2, 54.4, 2844, 69.8], [0.744, 90.2, 23, 2861, 68.4],
        [0.921, 72.8, 20.2, 3259, 77.9], [0.927, 67.7, 25.2, 3350, 78.1],
        [0.802, 82.6, 22.4, 3344, 72.5], [0.747, 74.4, 22.7, 2704, 66.6],
        [0.927, 83.3, 18.1, 3642, 76.7], [0.721, 83.7, 20.1, 2753, 68.8],
        [0.913, 73.8, 17.3, 2916, 76.8], [0.918, 79.2, 16.8, 3551, 78.1],
        [0.833, 71.5, 29.9, 3177, 73.9], [0.914, 75.3, 20.3, 3280, 78.6],
        [0.923, 79, 14.1, 3160, 78.5]
    ])
    
    # Разделяем данные на признаки (X) и целевую переменную (y)
    y_all = data[:, 0]  # Первый столбец - выходная величина y
    X_all = data[:, 1:]  # Остальные столбцы - признаки x1, x2, x3, x4
    
    print("ИСХОДНЫЕ ДАННЫЕ")
    print(f"Всего наблюдений: {len(data)}")
    print(f"Количество признаков: {X_all.shape[1]}")
    print()

    X_train_full = X_all[:20]
    y_train_full = y_all[:20]
    
    X_test = X_all[20:]
    y_test = y_all[20:]
    
    split_idx = int(len(X_train_full) * 0.6) 
    
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    
    X_valid = X_train_full[split_idx:]
    y_valid = y_train_full[split_idx:]
    
    print("РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"  Обучающая выборка: {len(X_train)} наблюдений (60% от 20)")
    print(f"  Проверочная выборка: {len(X_valid)} наблюдений (40% от 20)")
    print(f"  Тестовая выборка: {len(X_test)} наблюдений")
    print()
    
    mgua = MGUA(max_models_per_level=5)
    mgua.fit(X_train, y_train, X_valid, y_valid)

    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    
    y_pred_all = mgua.predict_best(X_all)
    
    y_pred_test = mgua.predict_best(X_test)
    
    test_mse = mgua.calculate_mse(y_test, y_pred_test)
    test_mape = mgua.calculate_mape(y_test, y_pred_test)
    
    print(f"MSE на тестовой выборке: {test_mse:.6f}")
    print(f"MAPE на тестовой выборке: {test_mape:.2f}%")
    print()
    
    print("ТАБЛИЦА СРАВНЕНИЯ ЗНАЧЕНИЙ")
    
    results_df = pd.DataFrame({
        '№': range(1, len(y_all) + 1),
        'Исходное Y': y_all,
        'Предсказанное Y': y_pred_all,
        'Ошибка': y_all - y_pred_all,
        'Ошибка, %': np.abs((y_all - y_pred_all) / y_all) * 100
    })
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(results_df.to_string(index=False))
    print()
    print("Примечание: строки 21-25 - тестовая выборка")
    print()
    
    print("ВЫВОДЫ О КАЧЕСТВЕ МОДЕЛИ")
    
    if test_mape < 5:
        quality = "ОТЛИЧНОЕ"
        comment = "Модель имеет высокую точность предсказаний."
    elif test_mape < 10:
        quality = "ХОРОШЕЕ"
        comment = "Модель показывает хорошие результаты."
    elif test_mape < 20:
        quality = "УДОВЛЕТВОРИТЕЛЬНОЕ"
        comment = "Модель приемлема для использования, но есть резерв для улучшения."
    else:
        quality = "НИЗКОЕ"
        comment = "Модель требует доработки или использования других методов."
    
    print(f"Качество модели: {quality} (MAPE = {test_mape:.2f}%)")
    print(f"Комментарий: {comment}")
    print()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(y_all) + 1), y_all, 'o-', 
             label='Исходные данные', linewidth=2, markersize=6)
    plt.plot(range(1, len(y_all) + 1), y_pred_all, 's--', 
             label='Модель МГУА', linewidth=2, markersize=6)
    plt.axvline(x=20.5, color='red', linestyle=':', linewidth=2, 
                label='Граница тестовой выборки')
    plt.xlabel('№ наблюдения', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Сравнение исходных и предсказанных значений', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    test_indices = range(21, 26)
    plt.plot(test_indices, y_test, 'o-', 
             label='Исходные данные', linewidth=2, markersize=8)
    plt.plot(test_indices, y_pred_test, 's--', 
             label='Модель МГУА', linewidth=2, markersize=8)
    plt.xlabel('№ наблюдения', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Тестовая выборка (детально)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()