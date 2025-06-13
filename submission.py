import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

def preprocess_ndvi(df):
    # Extract and sort NDVI columns chronologically
    ndvi_cols = [c for c in df.columns if '_N' in c]
    dates = [datetime.strptime(c.split('_')[0], '%Y%m%d') for c in ndvi_cols]
    sorted_cols = [c for _, c in sorted(zip(dates, ndvi_cols))]
    return df[sorted_cols]

def extract_features(df):
    features = []
    for _, row in df.iterrows():
        # Interpolation
        interp_vals = row.interpolate(method='linear', limit_direction='both')
        
        # Denoising
        if len(interp_vals) >= 5:
            smoothed = savgol_filter(interp_vals, 5, 2)
        else: 
            smoothed = interp_vals
            
        # Basic statistics
        stats = {
            'mean': np.mean(smoothed),
            'std': np.std(smoothed),
            'max_min_diff': np.max(smoothed) - np.min(smoothed),
            'trend': np.polyfit(range(len(smoothed)), smoothed, 1)[0]
        }
        
        # Seasonal features
        seasonal = {'spring': [], 'summer': [], 'fall': [], 'winter': []}
        for col, val in zip(df.columns, smoothed):
            month = datetime.strptime(col.split('_')[0], '%Y%m%d').month
            if 3 <= month <= 5:
                seasonal['spring'].append(val)
            elif 6 <= month <= 8:
                seasonal['summer'].append(val)
            elif 9 <= month <= 11:
                seasonal['fall'].append(val)
            else:
                seasonal['winter'].append(val)
        
        for k, v in seasonal.items():
            stats[f'{k}_mean'] = np.mean(v) if v else np.nan
            stats[f'{k}_std'] = np.std(v) if v else np.nan
            
        # Phenology features
        peaks, _ = find_peaks(smoothed, prominence=0.1)
        stats['peak_count'] = len(peaks)
        
        features.append(stats)
        
    return pd.DataFrame(features)

# Load data
train = pd.read_csv('hacktrain.csv')
test = pd.read_csv('hacktest.csv')

# Preprocess
X_train = preprocess_ndvi(train)
y_train = train['class']
X_test = preprocess_ndvi(test)

# Feature engineering
train_features = extract_features(X_train)
test_features = extract_features(X_test)

# Imputation
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(train_features)
X_test_imp = imputer.transform(test_features)

# Updated model pipeline - removed multi_class parameter
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='saga', max_iter=1000)
)

param_grid = {
    'logisticregression__C': [0.1, 1, 10],
    'logisticregression__class_weight': [None, 'balanced']
}

# Grid search
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_imp, y_train)

# Predictions
test_pred = grid.best_estimator_.predict(X_test_imp)

# Submission
submission = pd.DataFrame({
    'ID': test['ID'],
    'class': test_pred
})
submission.to_csv('submission.csv', index=False)
