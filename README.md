# JobBarcelona2023

## Descripción del reto  
Somos un banco que dispone de una base de datos con una gran cantidad de información sobre nuestros clientes. Nuestro objetivo es ayudar a los analistas a predecir la tasa de abandono de estos clientes para así poder reducirla. La base de datos incluye información demográfica como la edad, el sexo, el estado civil y la categoría de ingresos. También contiene información sobre el tipo de tarjeta, el número de meses en cartera y los periodos inactivos. Además, dispone de datos clave sobre el comportamiento de gasto de los clientes que se acercan a su decisión de cancelación. Entre esta última información hay el saldo total renovable, el límite de crédito, la tasa media de apertura a la compra y métricas analizables como el importe total del cambio del cuarto trimestre al primero o el índice medio de utilización.

Frente a este conjunto de datos podemos capturar información actualizada que puede determinar la estabilidad de la cuenta a largo plazo o su salida inminente.  

## Objetivo  
Crea un modelo predictivo de clasificación para poder clasificar los datos del archivo de testing. Primero entrena tu modelo con el conjunto de datos de training y una vez que tengas el modelo que maximice la puntuación f1 (macro.) utiliza los datos de testing como entrada para tu modelo.  


## Solución adoptada  

Tras comprobar que los datos estaban desbalanceados, primero se han realizado los distintos modelos sin realizar ninguna acción sobre los mismos. Los modelos escogidos para este reto han sido:  
- Adaboost  
- RandomForest  
- LGBM  
- Gradient Boosting Classifier (sklearn)  
- XGBoost  

Debido la existencia de outliers en algunas columnas, se ha decidido aplicar distintos métodos para el escalado de los datos, para no eliminar dichos datos ni que ocasionaran distorsiones importantes en los modelos. En el caso de las variables con outliers se aplica RobustScaler, mientras que en el resto, StarndarScaler. Esto se realiza mediante la siguiente función, una vez separados las columnas correspondientemente.

```
def scaler_columns(X, outliers_cols, no_outliers_cols):
    pipe = ColumnTransformer(transformers=[('robust', RobustScaler(), outliers_cols),
                                            ('standard', StandardScaler(), no_outliers_cols)],
                            remainder='passthrough')
    X_scaled = pipe.fit_transform(X)
    return pd.DataFrame(X_scaled, columns = X.columns)
```

Para cada uno de los modelos, se ha recurrido a la técnica de seleccion de características Recursive Feature Elimination. De esta forma se obtienen las característica más importante para cada uno de los modelos y, de esta forma, se evita entrenar con todo el conjunto de datos completo. Algunos enlaces para ver cómo funciona este método se encuentran en:  
- https://machinelearningmastery.com/rfe-feature-selection-in-python/   
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV  

y la función que lo realiza:

```
def recursiveFeatureElimination(X, y, model, X_test):
    '''
        This function uses Recursive Feature Elimination method
        to select the most importance features. This is a features selection based
        in wrapped
        
        Params: model: The basic model to find the most important features
        
        Return: Selector object
    '''
    estimator = eval(model)
    rfecv = RFECV(estimator, step=1, cv=5, n_jobs = -1, scoring='f1_macro')
    rfecv = rfecv.fit(X,y)
    X_transformed = rfecv.transform(X)
    X_test_transform = rfecv.transform(X_test)    
    return X_transformed, X_test_transform, rfecv
```

Para cada modelo se realiza un GridSearch pasándo un diccionario con distintos parametros, para encontrar el mejor modelo. Además de esto, se comprueba que el modelo no genere overfitting devolviendo un string 'no'. En caso de hacerlo la función devolverá un string con 'yes' que más tarde se utilizará para el filtrado de los modelos.

```
def gridSearch(model, params, X_train, y_train, X_test, y_test, nameModel):
    '''
       This function do cross validation and check if exists overffiting
       and call to confusionMatrix function.
       
       Params : model, params to set, train and test data, name model
       
       Return: f1score value, 
               overfitting : string 'yes' or 'no'
               rnd.best_params_ : dict with best params of the model
    '''
    # Search for best parameters
    rnd = RandomizedSearchCV(model, 
                             params, 
                             n_iter = 30, 
                             cv=5, 
                             random_state = 42,
                             scoring = "f1_macro",
                             n_jobs = -1)
    
    rnd.fit(X_train, y_train)
    print("--------\nResults\n--------\n")
    print(f"Best hyperparameters found: {rnd.best_params_}")
    #print(rnd.cv_results_)
        
    # Cross Validation
    model.set_params(**rnd.best_params_)
    train_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    test_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1_macro')
    print("Cross validation train:", train_scores)
    print("Cross validation test", test_scores)
    
    # Overfitting
    if train_scores.mean() > test_scores.mean():
        overfitting = 'yes'
        print("There may be overfitting in the model")
    else:
        overfitting = 'no'
        print("The model generalizes well")
        
        
    # Training and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1Score = f1_score(y_test, y_pred, average='macro')
    print("\n-------\nMetrics\n-------")
    print(f"f1_score: {f1Score}")
    
    # Show confusion matrix
    confusionMatrix(y_test, y_pred, nameModel)
    
    return f1Score, overfitting, rnd.best_params_
```

Finalmente, debido a la existencia de un desbalanceo importante en los datos, se aplica SMOTE sobre los mismos para generar un sobremuestreo de los datos con el fin de obtener mejores resultados. 

## Resultados y análisis  
El reto pedía una métrica f1-macro como evaluación de los resultados. Tras aplicar distintos modelos, se ha obtenido los siguientes resultados:  

|Model|f1_score(macro)|oversample|overfitting|
|------|----------|------------|-----------|
|AdaBoostClassifier()|0.797047|no|yes|
|RandomForestClassifier()|0.850013|no|no|
|lgb.LGBMClassifier()|0.948974|no|yes|
|GradientBoostingClassifier()|0.884724|no|yes|
|XGBClassifier()|0.928229|no|yes|
|AdaBoostClassifier()|0.915049|yes|no|
|RandomForestClassifier()|0.930880|yes|no|
|lgb.LGBMClassifier()|0.986766|yes|yes|
|GradientBoostingClassifier()|0.982357|yes|yes|
|XGBClassifier()|0.982354|yes|yes|

Dados los resultados, el modelo que mejor f1Score obtiene y sin que aparezca overfitting ha sido Random Forest tras aplicar SMOTE  

## Complicaciones y mejoras futuras  
Durante el proceso de creación de los modelos, destacaría la dificultad para parametrizar los modelos y no obtener sobreajuste. Tal y como se ha podido ver en la tabla de resultados, la mayoría de ellos presentan este problema. Como mejoras futuras se pueden estudiar más a fondo la parametrización de cada modelo y trabajarlo más a fondo, además de realizar otras transformaciones a los datos que puedan, junto con lo anterior, aportar mejores resultados evitando el sobreajuste de los datos.  

También sería importante agregar un EDA bien completo para que quienes visitan este proyecto entienda cada una de las variables que componen el dataset y las decisiones que puedan tomarse sobre los mismos datos. 

## Sobre el autor
David Molina, estudiante de Ciencia de Datos  
Se aceptan todo tipo de sugerencias que me permitan mejorar en futuros proyectos.  
Comparto mi linkedin: https://www.linkedin.com/in/david-molina-pulgarin-298253101
