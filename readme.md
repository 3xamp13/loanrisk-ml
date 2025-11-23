# Credit Risk Analysis

Проект предназначен для анализа кредитного риска и предсказания вероятности дефолта по кредитам с использованием методов машинного обучения.

---

## Описание

Пайплайн выполняет полный цикл анализа кредитных данных:

1. **Загрузка данных**: чтение CSV-файла `credit_risk_dataset.csv`.
2. **Предварительный анализ данных**:
   - Первичный просмотр таблицы.
   - Проверка информации о признаках.
   - Вычисление доли дефолтов.
3. **Очистка и подготовка данных**:
   - Удаление дубликатов.
   - Заполнение пропусков с помощью KNNImputer.
   - Удаление явных выбросов (возраст >120 лет, стаж работы >50 лет).
4. **Визуализация**:
   - Boxplots для выявления выбросов.
   - Гистограммы распределения числовых признаков.
5. **Кодирование признаков**:
   - Label Encoding для бинарных признаков.
   - One-Hot Encoding для категориальных признаков с несколькими классами.
6. **Масштабирование числовых признаков** с помощью `StandardScaler`.
7. **Обучение моделей**:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
8. **Оценка моделей**:
   - Метрики: Accuracy, Precision, Recall, F1-score, ROC-AUC.
   - Confusion Matrix.
   - Кривая обучения.
9. **Тюнинг XGBoost**:
   - RandomizedSearchCV для подбора гиперпараметров.
10. **Сохранение и загрузка модели** для предсказания новых клиентов.

---

## Визуализации

- Boxplots для проверки выбросов
- Гистограммы распределений признаков
- Confusion Matrix
- Learning Curve

---

## ⚙️ Установка зависимостей

```bash
git clone https://github.com/3xamp13/loanrisk-ml.git
cd loanrisk-ml
pip install -r requirements.txt

## Использование

Открыть **Jupyter Notebook** (`Credit_Risk_Analysis.ipynb`) и следовать шагам:

1. **Запуск всех ячеек:**
   - Данные загружаются из `data/credit_risk_dataset.csv`.
   - Производится очистка, кодирование и масштабирование.
   - Обучаются модели (Logistic Regression, Decision Tree, Random Forest, XGBoost).
   - Выводятся метрики и визуализации.

2. **Предсказание для новых клиентов:**

```python
import pandas as pd
import pickle

# Загружаем обученную модель
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Пример новых клиентов
new_data = pd.DataFrame({
    "person_age": [30, 45, 19],
    "person_income": [60000, 85000, 9000],
    "person_emp_length": [10, 20, 0],
    "loan_amnt": [15000, 25000, 20000],
    "loan_int_rate": [12.5, 8.7, 26.1],
    "cb_person_cred_hist_length": [5, 12, 1],
    "person_home_ownership": ["MORTGAGE", "OTHER", "RENT"],
    "loan_intent": ["EDUCATION", "HOMEIMPROVEMENT", "PERSONAL"],
    "loan_grade": ["B", "C", "G"]
})

# Подготовка данных (кодирование категориальных признаков и масштабирование)
new_X = pd.get_dummies(new_data)
new_X = new_X.reindex(columns=X_train.columns, fill_value=0)
new_X[num_cols] = scaler.transform(new_X[num_cols])

# Предсказание
predictions = model.predict(new_X)

# Преобразуем бинарный LabelEncoder обратно в оригинальные метки
final_pred = le.inverse_transform(predictions)

for i, p in enumerate(final_pred):
    print(f"Client {i+1} → Prediction: {p}")