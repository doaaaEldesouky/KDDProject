import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('students_performance_data/Cleaned_Students_Performance.csv')

df.fillna(method='ffill', inplace=True)

df['gender'] = df['gender'].map({'female': 0, 'male': 1})


numeric_columns = df.select_dtypes(include=['number']).columns

correlation_matrix = df[numeric_columns].corr()

# Draw (Heatmap)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# رسم Scatter Plot بين math_score و reading_score
sns.scatterplot(x='math_score', y='reading_score', data=df)
plt.title('Math Score vs Reading Score')
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.show()

# إعداد البيانات لـ Linear Regression
X = df[['math_score']]  # المتغير المستقل (الخاص بالرياضيات)
y = df['reading_score']  # المتغير التابع (الخاص بالقراءة)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error: {mse}")
print(f"R²: {r2}")

# draw Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='math_score', y='reading_score', data=df, label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Math Score vs Reading Score with Linear Regression Line')
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.legend()
plt.show()
