#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


# In[2]:


from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


# # 1. Opis danych

# Zestaw danych, który służy mi do wyznaczenia mocy umownej dotyczy elektrowni gazowo-parowej o cyklu kombinowanym. Dane zawierają 9568 rekordów, z czego każdy z rekordów stanowi średnią godzinową z pomiarów. Dane zostały zebrane na przestrzeni ponad 6 lat (w latach 2006-2011), gdy elektrownia pracowała pod pełnym obciążeniem. 

# Dane składają się z pięciu cech:
# * Temperatura otoczenia (AT) w °C,
# * Ciśnienie Atmosferyczne (AP) w mbar,
# * Względna Wilgotność (RH) w %,
# * Podciśnienie (V) w cmHg,
# * najwyższa wartość mocy pobranej w ciągu godziny (EP) w MW.

# Zmienne niezależne X:
# * AT,
# * AP,
# * RH,
# * V.
# 
# Zmienna zależna y, zmienna celu:
# * EP.

# Źródło: Zbiór danych pochodzi z the Machine Learning Repository of Center for Machine Learning and Intelligent Systems at the University of California, Irvine.
# 
# Link do zestawu: https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant

# # 2. Przejrzenie danych

# In[3]:


#wczytanie zestawu danych
dataset = pd.read_excel('Dane/Folds5x2_pp.xlsx')


# In[4]:


dataset = dataset.rename(columns={'PE': 'EP'})


# In[5]:


dataset.head(5)


# In[6]:


dataset.shape


# In[7]:


print(dataset.info())


# In[8]:


print(dataset.describe())


# In[9]:


len(dataset[dataset['RH'] == dataset['RH'].mode()[0]])


# In[10]:


dataset.isna().sum()


# # 3.Wizualizacja danych 

# ## 3.1 Podstawowy wykres pairplot obrazujący zależności miedzy zmiennymi

# In[11]:


sns.pairplot(dataset)


# ## 3.2 Przejrzenie rozkładu zmiennych

# In[12]:


sns.set(style="white", palette="muted", color_codes=True)
fig = plt.figure(figsize=(10,8))
spec = GridSpec(ncols=6, nrows=2, hspace=0.3) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0:2]) # row 0 with axes spanning 2 cols on evens
sns.histplot(data=dataset, x="AT", kde=True, color="b", ax=ax1)
ax1.set_xlabel('Temperatura (AT)')
ax1.set_ylabel('Liczebność')

ax2 = fig.add_subplot(spec[0,2:4], sharey=ax1)
sns.histplot(data=dataset, x="AP", kde=True, color="g", ax=ax2)
ax2.set_xlabel('Ciśnienie otoczenia (AP)')
ax2.yaxis.set_tick_params(labelleft=False)
ax2.axes.get_yaxis().set_visible(False)

ax3 = fig.add_subplot(spec[0,4:],sharey=ax1)
sns.histplot(data=dataset, x="RH", kde=True, color="r", ax=ax3)
ax3.set_xlabel('Wilgotność względna (RH)')
ax3.yaxis.set_tick_params(labelleft=False)
ax3.axes.get_yaxis().set_visible(False)

ax4 = fig.add_subplot(spec[1,:3])
sns.histplot(data=dataset, x="V", kde=True, color="m", ax=ax4)
ax4.set_xlabel('Podciśnienie w układzie wydechowym (V)')
ax4.set_ylabel('Liczebność')

ax5 = fig.add_subplot(spec[1,3:], sharey=ax4)
sns.histplot(data=dataset, x="EP", kde=True, color="orange",ax=ax5)
ax5.set_xlabel('Moc pobrana (EP)')
ax5.set_ylabel('Liczebność')
ax5.yaxis.set_tick_params(labelleft=False)
ax5.axes.get_yaxis().set_visible(False)
plt.savefig('rozklad.png')
plt.show()


# ## 3.3 Analiza  brakujących wartości, powielonych obserwacji oraz obserwacji odstających

# ### 3.3.1 Wartości brakujące

# In[13]:


dataset.isna().sum()


# **Wniosek**: W danych nie występują wartości brakujące

# ### 3.3.2 Powielone obserwacje

# In[14]:


dataset.duplicated().sum()


# **Wniosek**: w zestawie danych wystepuje 41 obserwacji, które posiadają taką samą wartość. Mimo to w przypadku tego typu danych, nie wydaje się, żeby wystąpienie identycznych obserwacji świadczyło o błędzie. Jest wielce prawdopodobne, że podobne obserwacje mogły wystapić. Dlatego też postanowiłem pozostawić te obserwacje w danych. 

# ### 3.3.3 Sprawdzenie występowania obserwacji odstających przy użyciu metody zakresu międzykwartylowego

# **Metoda**: zakres międzykwartylowy (IQR)
# 
# **Uzasadnienie**: zmienne posiadają różnorodne rozkłady

# In[15]:


# Listy pomocnicze służące do stworzenia tablicy nizbędnych miar wartości odstających dal metody IQR
lower_limits = list()
upper_limits = list()
occurances = list()
outliers_counts = list()
ranges_below = list()
ranges_above = list()

# wyznaczenie obserwacji odstających metodą 1.5 * rozstęp międzykwartylowy
for i, col in enumerate(['AT','AP','RH','V','EP']):
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    outliers = dataset[(dataset[col] < lower_bound) | (dataset[col] > upper_bound)]
    
    # uzupełnienie rekordów do list pomocniczych
    
    #dolna i górna granica metody 
    lower_limits.append(lower_bound)
    upper_limits.append(upper_bound)
    #ustalenie czy dana zmienna posiada obserwacje odstające
    occurances.append("Tak" if len(outliers) !=0 else "Nie")
    #ilość obserwacji odstającej dla zmiennej
    outliers_counts.append(len(outliers))
    #ustalenie zakresu danych odstających poniżej i powyżej granic, jesli takowe występują
    outliers_below = dataset[dataset[col] < lower_bound][col]
    outliers_above = dataset[dataset[col] > upper_bound][col]
    
    if not outliers_below.empty:
        range_below = f"{round(outliers_below.max(),2)} - {round(lower_bound,2)}"
    else:
        range_below = "Brak"
    ranges_below.append(range_below)
    # wyznaczenie zakresu wartości dla wartości powyżej górnej granicy IQR
    if not outliers_above.empty:
        range_above = f"{round(upper_bound,2)} - {round(outliers_above.min(),2)}"
    else:
        range_above = "Brak"
    ranges_above.append(range_above)
    # wyświetlenie wyników
    
index = ['Temperatura', 'Ciśnienie otoczenia', 'Wilgotność względna', 'Podciśnienie w układzie wydechowym', 'Zapotrzebowanie na energię']

# utworzenie dataframe
df_outliers = pd.DataFrame({
    'Dolna granica': lower_limits,
    'Górna granica': upper_limits,
    'Czy obserwacje odstające występują?': occurances,
    'Liczba obserwacji odstających': outliers_counts,
    'Zakres poniżej dolnej granicy': ranges_below,
    'Zakres powyżej górnej granicy': ranges_above
}, index=index)


# In[16]:


df_outliers


# Wizualizacja wartości odstających przy użyciu wykresów pudełkowych dla zmiennych Ciśnienie otoczenia (AP) Wilgotności względenj(RH)

# In[17]:


fig, axes = plt.subplots(1, 2, figsize=(10,5))
sns.boxplot(data=dataset,ax=axes[0], x ='AP',color= 'g')
axes[0].set_xlabel('Ciśnienie otoczenia (AP)')
sns.boxplot(data=dataset,ax=axes[1], x ='RH',color= 'b')
axes[1].set_xlabel('Wilgotność względna (RH)')
fig.suptitle('Wizualizacja obserwacji odstających przy użyciu wykresów pudełkowych')
plt.show()


# **Wniosek**: obserwacje odstające wystepujące zarówno w zmiennej AP, jak i RH nie wydają się rekordami błędnymi. Prezentują rzeczywiste wartości, które mogą wystapić. Dltego też obserwacje te z danych nie zostaną usunięte.

# ## 3.4 Analiza zależności między zmiennymi niezależnymi X, a zmienną zależną y

# ### 3.4.1 Wizualizacja zależności przy użyciu wykresów punktowych

# In[18]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6),sharey=True)
fig.tight_layout(pad=2.5)
plt.suptitle("Wykresy relacji między zmiennymi", fontsize=14)
sns.scatterplot(data=dataset, x='AT', y='EP', ax=axes[0,0])
axes[0,0].set_xlabel('Temperatura (AT)');
axes[0,0].set_ylabel('Moc pobrana (EP)');
sns.scatterplot(data=dataset,x='AP', y='EP', ax=axes[0,1])
axes[0,1].set_xlabel('Ciśnienie otoczenia (AP)');
axes[0,1].set_ylabel('Moc pobrana (EP)');
sns.scatterplot(data=dataset,x='RH', y='EP', ax=axes[1,0])
axes[1,0].set_xlabel('Wilgotność względna (RH)');
axes[1,0].set_ylabel('Moc pobrana (EP)');
sns.scatterplot(data=dataset,x='V', y='EP', ax=axes[1,1])
axes[1,1].set_xlabel('Podciśnienie w układzie wydechowym (V)');
axes[1,1].set_ylabel('Moc pobrana (EP)');
plt.show()


# Na podstawie powyższych wykresów widzimy, że zmienne AT i V przypominają zależność liniową ze zmienną PE.

# ### 3.4.2 Badanie zależności liniowej przy użyciu korelacji Pearsona

# In[19]:


mask = np.triu(np.ones_like(dataset.corr(method='pearson'), dtype=bool))
sns.heatmap(dataset.corr(method='pearson'), annot=True, fmt='.1g', mask=mask).set_title('Korelacja Pearsona dla atrybutów', fontdict={'fontsize':12}, pad=13)
plt.show()


# ## Zbadanie korelacji między zmiennymi niezależnymi, a zmieną zależną

# In[20]:


sns.heatmap(dataset.corr(method='pearson')[['EP']][:-1],annot=True).set_title('Korelacja mocy poranej [EP] ze zmiennymi niezależnymi ', fontdict={'fontsize':12}, pad=13);
plt.show()


# **Wnioski**:Na podstawie wyznaczonej korelacji Pearsona (zależności liniowej zmiennych) 
# między zmiennmi AT, V, AP, RH, a zmienną PE wyraźnie widać, że zmienne 
# AT i V oraz AP posiadają silną korelację ze zmienną PE zakłądając, że silna koralacja wystepuje w przedziale \[-1, -0.5\] U \[0.5, 1\]. 
# Zmienna RH posiadaja umiarkowaną korelację dodatnią ze zmienną PE.

# # 4. Wybór modelu

# ## 4.1 Podział danych

# **Strategia**: Na początkowym etapie dane zostały podzielone na grupę uczącą i testującą w proporcjach 80:20. Grupa uczące będzie służyła do zastosowania przy wyborze modelu. W toku wyboru modelu zastosuję technikę 5-krotnej walidacji krzyżowej, co będzie stanowiło podział dla Kolejnej grupy uczącej i walidacyjnej w proporcji ok. 80:20.

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


# Zeminne niezależne
input_cols = dataset.columns[dataset.columns != 'EP']
# Zmienna zależna
target_col = dataset.columns[dataset.columns == 'EP']


# In[23]:


# Podział danych na grupę uczącą oraz testującą w proporcjach 80:20
X_train, X_test, y_train, y_test = train_test_split(dataset[input_cols], dataset[target_col], test_size=0.2, random_state=42)


# In[24]:


# Zestawienie podziału danych 
Dataset_division = pd.DataFrame({"Dane uczące": [int(X_train.shape[0]),round(X_train.shape[0]/len(dataset) * 100)],
                                 "Dane testujące": [int(X_test.shape[0]),round(X_test.shape[0]/len(dataset) * 100)]
                                })
Dataset_division.index = ['Liczba', '%']
print("Podział danych na grupę uczącą i testującą")
Dataset_division


# ### Przydatne metody

# In[25]:


# Zaimportowanie implementacji algorytmu przeszukiwania siatki z biblioteki Scikit-learn
from sklearn.model_selection import GridSearchCV

# Funkcja do przeszukiwania hiperparametrów danego estymatora
def hyperparam_checking(estimator, params, X_train, y_train):
    """
    Metoda do przeszukiwania hiperparametrów estymatora przy użyciu metody przeszukiwania siatki.
    
    Parametry:
    ----------
    estimator: wybrany model uczenia maszynowego z biblioteki Scikit-learn
    params: słownik z rozważnymi parametrami
    X_train: wektor zmiennych niezależnych
    y_train: wektor zmiennej zależnej
    
    Zwraca: 
    ----------
    słownik ze szczegółym zestawieniem wyników
    """
    scoring = 'neg_mean_squared_error'
    grid_search = GridSearchCV(estimator, param_grid=params, cv=5, scoring=scoring, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train.values.ravel())
    results = grid_search.cv_results_

    return results


# In[26]:


def make_df(results):
    """ Metoda tworzy Tabelę służącą do porównania hiperparametrów.
        Tabela składa się z kolumn: Nazwa każdego hiperparametru, Średni wynik grupy uczącej, Średni wynik grupy walidacyjnej,
        Różnica między grupą uczącą, a walidacyjną
        
         Parametry:
        ----------
        results: słownik ze szczegółym zestawieniem wyników z metody GridSearch
        params: słownik z rozważnymi parametrami
        X_train: wektor zmiennych niezależnych
        y_train: wektor zmiennej zależnej

        Zwraca: 
        ----------
        DataFrame ze średnim wynikiem dla grupy uczącej, walidacyjnej oraz różnicę między grupami
    """
    df_results = pd.DataFrame()
    train_scores = results['mean_train_score']
    test_scores = results['mean_test_score']
    for key in list(results['params'][0].keys()):
        df_results[f'{key}'] = [params[f'{key}'] for params in results['params']]
    df_results['Średni wynik grupy uczącej'] = abs(train_scores)
    df_results['Średni wynik grupy walidacyjnej'] = abs(test_scores)
    df_results['Różnica między grupą uczącą, a walidacyjną'] = abs(train_scores - test_scores)
    
    return df_results


# In[27]:


def make_df_final_model(results, params, model_name):
    """ 
    Metoda wyświetlająca wybrany model w formie tabeli, która nazwiera nazwę algorytmu, dobrane hiperparametry,
    średni wynik dla grupy walidacyjnej oraz odchylenie dal grupy walidacyjnej
    
    Parametry:
    ----------
    results: słownik ze szczegółym zestawieniem wyników z metody GridSearch
    params: słownik z przyjętymi wartościami parametrów dla danej metody
    model_name: nazwa modelu
   
    Zwraca: 
    ----------
    DataFrame ze średnim wynikiem dla grupy walidacyjnej oraz odchylenie standardowe dla grupy walidacyjnej
    """
    results = pd.DataFrame(results)
    df = pd.DataFrame({
                       'Średni wynik dla grupy walidacyjnej': abs(results[results['params'] == params]['mean_test_score'].values[0]),
                       'Odchylenie std dla grupy walidacyjnej' : results[results['params'] == params]['std_test_score'].values[0]
                      }, index= [model_name])
    return df


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[29]:


from sklearn.model_selection import cross_validate


# ## 4.1 Regresja Liniowa

# In[30]:


linear = LinearRegression()
scoring = 'neg_mean_squared_error'
scores_lin = cross_validate(linear, X_train, y_train.values.ravel(), cv=5, scoring=scoring, return_train_score=True)


# In[31]:


df_lin_model = pd.DataFrame( {"Średni wynik grupy uczącej": abs(scores_lin['train_score']).mean(),
                                    "Średni wynik grupy walidacyjnej": abs(scores_lin['test_score']).mean(),
                                     "Różnica między grupą uczącą, a walidacyjną":abs((abs(scores_lin['train_score']).mean() - abs(scores_lin['test_score']).mean()))}, index =[0])
                            


# Wybrany model regresji liniowej jest domyślną formą zapewnioną przez bibliotekę scikit-learn

# ### Wyniki dla wybranego modelu

# In[32]:


df_lin_model


# In[33]:


df_line = pd.DataFrame({
    'Średni wynik MSE dla grupy walidacyjnej': abs(scores_lin['test_score']).mean(),
    'Odchylenie std MSE dla grupy walidacyjnej': abs(scores_lin['test_score']).std()
}, index=['Regresja Liniowa'])


# In[34]:


df_line


# ## 4.2 KNN

# In[35]:


estymator_knn = KNeighborsRegressor()
param_knn  = {'n_neighbors': list(range(1,41))}


# In[36]:


knn_results = hyperparam_checking(estymator_knn, param_knn, X_train, y_train)


# ### Wybór hiperparametrów dla kNN

# In[37]:


df_knn = make_df(knn_results)


# In[38]:


df_knn[df_knn['Różnica między grupą uczącą, a walidacyjną'] <= 2].sort_values('Średni wynik grupy walidacyjnej', ascending=True).head(5).reset_index(drop=True)


# Wybrany parametry dla knn to: 
# * **liczba sąsiadów**: 20

# ### Wyniki dla wybranego modelu

# In[39]:


final_knn =  make_df_final_model(knn_results, {'n_neighbors': 20}, 'K-najbliższych sąsiadów')


# In[40]:


final_knn


# ### Ocena modelu k-najbliższych sąsiadów poprzez 5-krotną walidację krzyżową

# ## 4.3 Decision Tree

# In[41]:


estymator_tree = DecisionTreeRegressor()
param1_tree = {'max_depth': list(range(1,31))}


# In[42]:


tree_results = hyperparam_checking(estymator_tree,param1_tree, X_train, y_train)


# ### Wybór hiperparametrów dla drzewa dezycyjnego

# In[43]:


df_tree = make_df(tree_results)


# In[44]:


df_tree[df_tree['Różnica między grupą uczącą, a walidacyjną'] <= 2.0].sort_values('Średni wynik grupy walidacyjnej', ascending=True).head(5).reset_index(drop=True)


# Wybrany parametry dla drzewa decyzyjnego to: 
# * **maksymalna głębokość drzewa**: 5

# ### Wyniki dla wybranego modelu

# In[45]:


final_tree = make_df_final_model(tree_results, {'max_depth': 5}, 'Drzewo decyzyjne')


# In[46]:


final_tree


# ## 4.4 Random Forest

# In[47]:


estymator_forest = RandomForestRegressor()
param1_forest = {'max_depth': [3,4,5,6,7,10,15,20,30], 'n_estimators' : [50,70,90,100,200,300,400,500]}


# ### Wybór hiperparametrów dla lasu losowego

# In[48]:


get_ipython().run_cell_magic('time', '', 'forest_results = hyperparam_checking(estymator_forest,param1_forest, X_train, y_train)\n')


# In[49]:


df_forest = make_df(forest_results)


# In[50]:


df_forest [df_forest ['Różnica między grupą uczącą, a walidacyjną'] <= 2.0].sort_values('Średni wynik grupy walidacyjnej', ascending=True).head(5).reset_index(drop=True)


# Wybrany parametry dla drzewa decyzyjnego to: 
# * **maksymalna głębokość drzewa**: 5
# * **liczba estymatorów**: 200

# ### Wyniki dla wybranego modelu

# In[51]:


final_forest = make_df_final_model(forest_results, {'max_depth': 5, 'n_estimators':200 }, 'Las losowy')


# In[52]:


final_forest


# ## 4.5 Gradient Boosting

# In[53]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[54]:


estymator_gradient = GradientBoostingRegressor()
param1_gradient = { 'max_depth': [2,4,5,6,10,15,30],
                    'n_estimators' : [10, 50, 75, 100, 200, 300],
                    'learning_rate' : [0.1,0.2, 0.3,0.5, 1]}


# ### Wybór hiperparametrów dla Gradient boostingu

# In[55]:


get_ipython().run_cell_magic('time', '', 'gradient_results = hyperparam_checking(estymator_gradient, param1_gradient, X_train, y_train)\n')


# In[56]:


df_gradient = make_df(gradient_results)


# In[57]:


df_gradient[df_gradient['Różnica między grupą uczącą, a walidacyjną'] <= 2.0].sort_values('Średni wynik grupy walidacyjnej', ascending=True).head(5).reset_index(drop=True)


# Wybrany parametry dla drzewa decyzyjnego to: 
# * **maksymalna głębokość drzewa**: 4
# * **liczba estymatorów**: 50
# * **współczynnik uczenia**: 0.1

# ### Wyniki dla wybranego modelu

# In[58]:


final_gradient = make_df_final_model(gradient_results, {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 50}, f'Wzmocnienie gradientu')


# In[59]:


final_gradient


# ## 4.6 ANN

# In[60]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


# ### 4.6.1 Konstrukcja sieci

# Sieć składa się z:
# * **liczba warstw ukrytych**: 2
# * **liczba neuronów w warstwach ukrytych**: 6
# * **rozmiar partii danych**: 32 
# * **liczba epok**: 100
# * **optymajzer**: Adam
# * **funkcja aktywacyjna dla warstwy wejściowej oraz warstw ukrytych**: ReLu
# * **funkcja aktywacyjna dla warstwy wyjściowej**: Linear

# ### Wizualne przedstawienie sieci

# <img src="images\final_model.png">

# ### 4.6.2 Sprawdzenie sieci przy pomocy pięcio krotnej walidacji krzyżowej

# In[61]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def eval_ann(hidden_layers, units, X_train, y_train, epochs, batch_size):
    """
    Metoda służąca do zbudowania sieci neuronowej
    
    Parametry:
    ----------
    hidden_layers: liczba warstw ukrytych
    units: liczba neuronów w warstwie ukrytej
    X_train: Wektor zmiennych niezaleznych (X) zestawu uczącego
    y_train: Wektor zmiennej celu (u) zestawu uczącego
    epochs: liczba epok
    batch_size: liczba danych w partii
    
    Zwraca: 
    ----------
    DataFrame ze średnim wynikiem dla grupy uczącej i walidacyjnej oraz ich odchylenia
    """
    ann = tf.keras.models.Sequential()
    for i in range(hidden_layers):
        ann.add(tf.keras.layers.Dense(units=units, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1))
    ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    train_mse = []
    valid_mse = []
    
    for train_idx, test_idx in kfold.split(X_train):
        # podział danych na zbiory treningowe i testowe
        X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

        # trenowanie modelu i obliczenie metryk
        ann.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0)
        train_pred = ann.predict(X_train_fold, verbose=0)
        test_pred = ann.predict(X_test_fold, verbose=0)
        train_mse.append(mean_squared_error(y_train_fold, train_pred))
        valid_mse.append(mean_squared_error(y_test_fold, test_pred))
        
    results_df = pd.DataFrame({
        'Warstwy ukryte' : hidden_layers,
        'Neurony w warstwach ukrytych' : units,
        'Średnia z wyników dla grupy uczącej': np.mean(train_mse),
        'Odchylenie std dla grupy uczącej': np.std(train_mse),
        'Średnia z wyników dla grupy walidacyjnej': np.mean(valid_mse),
        'Odchylenie std dla grupy walidacyjnej': np.std(valid_mse),
    }, index=[0])
    return results_df


# In[62]:


get_ipython().run_cell_magic('time', '', 'ann_score_df = eval_ann(2, 6, X_train, y_train, 100, 32)\n')


# In[63]:


ann_score_df


# In[64]:


final_ann = ann_score_df[['Średnia z wyników dla grupy walidacyjnej', 'Odchylenie std dla grupy walidacyjnej']]


# In[65]:


final_ann


# Zapisanie finalnych wyników z dania 12.04.2023 w formacie CSV jako wyniki wykorzystane przy porównaniu modeli

# In[66]:


#final_ann.to_csv('wyniki_dla_ann.csv', index=False)


# ## 4.7 Analiza porównawcza wyników uzyskanych przez modele 

# Stoworzenie tabeli do porówniania modeli

# In[67]:


final_df = pd.DataFrame(columns = ['Średni wynik MSE dla grupy walidacyjnej', 'Odchylenie std MSE dla grupy walidacyjnej'])
# Regresja liniowa
lin_row = pd.DataFrame({
    'Średni wynik MSE dla grupy walidacyjnej': abs(scores_lin['test_score']).mean(),
    'Odchylenie std MSE dla grupy walidacyjnej': abs(scores_lin['test_score']).std()
}, index=['Regresja Liniowa'])
final_df = pd.concat([final_df, lin_row])
# kNN
final_knn =  make_df_final_model(knn_results, {'n_neighbors': 20}, 'K-najbliższych sąsiadów')
knn_row = pd.DataFrame({
   
    'Średni wynik MSE dla grupy walidacyjnej': final_knn.iloc[0].values[0] ,
    'Odchylenie std MSE dla grupy walidacyjnej': final_knn.iloc[0].values[1] 
}, index=['k-najbliższych sąsiadów'])
final_df = pd.concat([final_df, knn_row])
# Drzewo decyzyjne
final_tree = make_df_final_model(tree_results, {'max_depth': 5}, 'Drzewo decyzyjne')
tree_row = pd.DataFrame({
   
    'Średni wynik MSE dla grupy walidacyjnej': final_tree.iloc[0].values[0] ,
    'Odchylenie std MSE dla grupy walidacyjnej': final_tree.iloc[0].values[1] 
}, index=['Drzewo decyzyjne'])
final_df = pd.concat([final_df, tree_row])
# Las losowy
final_forest = make_df_final_model(forest_results, {'max_depth': 5, 'n_estimators':200 }, 'Las losowy')
random_row = pd.DataFrame({
    
    'Średni wynik MSE dla grupy walidacyjnej': final_forest.iloc[0].values[0] ,
    'Odchylenie std MSE dla grupy walidacyjnej': final_forest.iloc[0].values[1] 
}, index=['Las losowy'])
final_df = pd.concat([final_df, random_row])
# Zejście gradientu
final_gradient = make_df_final_model(gradient_results, {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 50}, f'Wzmocnienie gradientu')
gradient_row = pd.DataFrame({
    
    'Średni wynik MSE dla grupy walidacyjnej': final_gradient.iloc[0].values[0] ,
    'Odchylenie std MSE dla grupy walidacyjnej': final_gradient.iloc[0].values[1] 
}, index=['Wzmocnienie gradientu'])
final_df = pd.concat([final_df, gradient_row])
# ANN
final_ann = pd.read_csv("wyniki_dla_ann.csv")

ann_row = pd.DataFrame({
    
    'Średni wynik MSE dla grupy walidacyjnej': final_ann.iloc[0].values[0],
    'Odchylenie std MSE dla grupy walidacyjnej': final_ann.iloc[0].values[1]
}, index=['Sztuczna sieć neuronowa'])
final_df = pd.concat([final_df, ann_row])
final_df


# Wyraźnie widać że najlepszą model jest Wzmocnienie gradientu o hiperparametrach 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 50`

# ## 5. Ewaluacja wyselekcjonowanego modelu modelu

# In[68]:


from sklearn.ensemble import GradientBoostingRegressor


# In[69]:


tuned_model = GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 50, max_depth = 4);
tuned_model.fit(X_train, y_train.values.ravel());


# In[70]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[71]:


y_predicted = tuned_model.predict(X_test)
y_actual = y_test.values.ravel()


# In[72]:


from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error

def evaluate_regression_model(model_name: str, y: list[int], y_pred: list[int], X : list[int])-> pd.DataFrame:
    """
    Funkcja ocenia model regresji na podstawie wybranych miar.
    
    model_name: nazwa modelu
    y: wektor wartości rzeczywistych
    y_pred: wektor wartości przewidzianych przez model
    X: macierz cech (atrybutów)
    
    Zwraca: 
    dataframe zawierający miary oceny modelu
    """
    # Miary oceny modelu
    metrics = {
        
        'R^2': round(r2_score(y, y_pred),3),
        'MAE': round(mean_absolute_error(y, y_pred),3),
        'RMSE': round(mean_squared_error(y, y_pred, squared=False),3),
        
    }
    
    return pd.DataFrame(metrics, index=[f"{model_name}"])


# In[73]:


evaluate_regression_model("GradientBoosting",y_actual, y_predicted, X_test)


# In[74]:


import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_regression_results(y_predicted: list[int], y_actual: list[int]) -> None:
    """
    Funkcja wizualizuje analizę błędów (residuals) oraz różnicę między wartościami rzeczywistymi
    a przewidzianymi przez model regresji.
    
    model: obiekt reprezentujący model regresji
    X: macierz cech (atrybutów)
    y: wektor wartości docelowych
    title: tytuł wykresu (opcjonalny, domyślnie brak tytułu)
    """
    
    # Analiza pozostałości
    residuals = y_actual - y_predicted
    
    # Tworzenie trzech wykresów
    gs = gridspec.GridSpec(2, 2)
    plt.figure(figsize=(10,10))
    
    # Wykres analizy pozostałości (residuals)
    
    ax1 = plt.subplot(gs[0, 0]) # row 0, col 0
    sns.scatterplot(x=y_predicted, y=residuals, ax=ax1)
    ax1.axhline(y=0,color='r', linestyle='--')
    ax1.set_xlabel('Przewidywane wartości')
    ax1.set_ylabel('Pozostałości')
    ax1.set_title("Zestawienie pozostałości z wartościami przewidywanymi",fontsize = 10)
   

    # Wykres różnicy między wartościami rzeczywistymi a przewidzianymi
    sns.set(style="white", palette="muted", color_codes=True)
    ax2 = plt.subplot(gs[0, 1]) 
    sns.scatterplot(x=y_predicted, y=y_actual, ax=ax2)
    sns.lineplot(x=[420, 490], y=[420, 490], ax=ax2, color='r', linestyle='--')
    ax2.set_xlabel('Przewidywane wartości')
    ax2.set_ylabel('Wartości rzeczywiste')
    ax2.set_title("Zestawienie wartości przewidywanych z rzeczywistymi",fontsize = 10)
    
    # Rozkład pozostałości (residuals)
    ax3 = plt.subplot(gs[1, :]) 
    sns.histplot(residuals, kde=True, ax=ax3, color="purple")
    ax3.set_xlabel('Pozostałości')
    ax3.set_ylabel('Liczebność')
    ax3.set_title("Rozkład pozostałości",fontsize = 10)
    plt.show()


# In[75]:


plot_regression_results(y_predicted, y_actual)


# Wnioski: Model Wzmocnienia gradientu wydaje się być stabilny i ogólny, zatem jest odpowiednim modelem do wdrożenia
