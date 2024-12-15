import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.tree import export_graphviz
from graphviz import Source

# قراءة البيانات
df = pd.read_csv('students_performance_data/Cleaned_Students_Performance.csv')

# تعديل قيم 'gender' من 0 و 1 إلى 'female' و 'male'
df['gender'] = df['gender'].replace({0: 'female', 1: 'male'})

# معالجة القيم المفقودة باستخدام ffill
df.ffill(inplace=True)

# معالجة البيانات المكررة
df.drop_duplicates(inplace=True)

# عرض بعض المعلومات الأساسية عن البيانات
print(df.info())  # معلومات عامة عن البيانات
print(df.describe())  # الإحصائيات الأساسية

# رسم التوزيعات باستخدام seaborn
plt.figure(figsize=(10, 6))
sns.histplot(df['math_score'], kde=True, color='blue')
plt.title('Distribution of Math Scores')
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.show()

# رسم Countplot للتوزيع حسب 'gender'
plt.figure(figsize=(8, 5))
sns.countplot(x='gender', data=df, palette='viridis')
plt.title('Distribution of Gender')
plt.show()

# رسم KDE Plot لعرض توزيع الدرجات بين 'math_score' و 'reading_score'
plt.figure(figsize=(10, 6))
sns.kdeplot(df['math_score'], fill=True, color='purple', label='Math Scores')
sns.kdeplot(df['reading_score'], fill=True, color='green', label='Reading Scores')
plt.legend()
plt.title('KDE Plot for Math and Reading Scores')
plt.show()

# رسم Boxplot لعرض العلاقة بين 'gender' و 'math_score'
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='math_score', data=df, palette='viridis')
plt.title('Boxplot of Math Score by Gender')
plt.show()

# تحويل الأعمدة غير العددية إلى متغيرات عددية باستخدام one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)
print(df_encoded.head())
df_cleaned = df_encoded[(np.abs(stats.zscore(df_encoded.select_dtypes(include=['number']))) < 3).all(axis=1)]

# تحديد المتغيرات المستقلة (features) والمتغير التابع (target)
X = df_encoded.drop('reading_score', axis=1)  # المتغيرات المستقلة
y = df_encoded['reading_score']  # المتغير التابع

# استبعاد القيم الشاذة باستخدام Z-Score
df_cleaned = df[(np.abs(stats.zscore(df.select_dtypes(include=['number']))) < 3).all(axis=1)]

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تطبيع البيانات باستخدام StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تعريف النماذج المناسبة
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Ridge': Ridge(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Linear Regression': LinearRegression()
}

# تدريب النماذج وتقييمها
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # تأكد من إضافة النتائج بشكل صحيح
    results[model_name] = {'MSE': mse, 'R²': r2, 'MAE': mae}
    print(f'{model_name} - MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}')

# مقارنة الدقة بين النماذج باستخدام R²
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=[res['R²'] for res in results.values()], hue=list(results.keys()), palette='viridis')

plt.title('Model R² Score Comparison')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.show()

# رسم Heatmap للمصفوفة الارتباطية
df['gender'] = df['gender'].map({'female': 0, 'male': 1})  # إعادة تحويل 'gender' إلى قيم رقمية
numeric_columns = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap')
plt.show()

# رسم Scatter Plot بين math_score و reading_score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='math_score', y='reading_score', data=df, color='purple')
plt.title('Math Score vs Reading Score')
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.show()

# إعداد البيانات لـ Linear Regression
X = df[['math_score']]  # المتغير المستقل (الخاص بالرياضيات)
y = df['reading_score']  # المتغير التابع (الخاص بالقراءة)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء نموذج Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# التنبؤ بالقيم باستخدام النموذج المدرب
y_pred = model.predict(X_test)

# حساب الأخطاء باستخدام MSE و R² و MAE
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# طباعة نتائج الأخطاء
print(f"Linear Regression - MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

# رسم Scatter Plot مع خط الانحدار (Regression Line)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='math_score', y='reading_score', data=df, label='Actual Data', color='green')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Math Score vs Reading Score with Linear Regression Line')
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.legend()
plt.show()

# رسم Residuals vs Predicted Values
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Reading Score')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# تدريب نموذج شجرة الانحدار مع تقليل عمق الشجرة
dt_model = DecisionTreeRegressor(random_state=42, max_depth=4)  # تقليل عمق الشجرة
dt_model.fit(X_train, y_train)

# تصدير الشجرة إلى صيغة Graphviz
dot_data = export_graphviz(dt_model, out_file=None, 
                           feature_names=X.columns,  
                           filled=True, rounded=True,  
                           special_characters=True, 
                           proportion=True)  # تصغير الصورة وجعل العقد أكثر ترتيبًا

graph = Source(dot_data)

# توليد صورة بصيغة PNG للشجرة مع تصغير الحجم
graph.render("regression_tree", format="png", cleanup=True, engine='dot')  # تحديد محرك الرسم لتقليل الحجم
graph.view()  # عرض الصورة
