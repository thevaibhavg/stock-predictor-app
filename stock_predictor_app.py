def train_models(data):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'RSI_14', 'Daily_Return']
    
    # Target for regression (next day's price)
    data['Target_Close'] = data['Close'].shift(-1)
    
    # Target for classification (1 = UP, 0 = DOWN)
    data['Target_Direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    data = data.dropna()  # Drop rows with NaNs after shifting
    
    X = data[features]
    y_reg = data['Target_Close']
    y_clf = data['Target_Direction']
    
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    reg_model.fit(X, y_reg)
    clf_model.fit(X, y_clf)
    
    # Return models + last row to predict next value
    last_row = X.tail(1)
    return reg_model, clf_model, last_row

