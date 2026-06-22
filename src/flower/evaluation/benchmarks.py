from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def resiualize_categorical(X_train, y_train, X_test, y_test, model):
    # 2. Regress out 'factor_to_remove' using ONLY the fold's training data
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    y_train = ohe.fit_transform(y_train)
    y_test = ohe.transform(y_test)

    pipeline = Pipeline([
        ('StandardScaler', StandardScaler()),
        ('lr', model)
    ])
    pipeline.fit(y_train, X_train)

    # 3. Calculate Residuals for this fold
    X_train_res = X_train - pipeline.predict(y_train)
    X_test_res = X_test - pipeline.predict(y_test)
    return X_train_res, X_test_res

def resiualize_continuous(X_train, y_train, X_test, y_test, model):
    pipeline = Pipeline([
        ('StandardScaler', StandardScaler()), # added
        ('lr', model)
    ])
    pipeline.fit(y_train, X_train)

    X_train_res = X_train - pipeline.predict(y_train)
    X_test_res = X_test - pipeline.predict(y_test)
    return X_train_res, X_test_res