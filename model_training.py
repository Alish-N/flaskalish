# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

def analyze_data(df):
    """Analyze data to understand its characteristics"""
    print("\nData Analysis:")
    print("Shape:", df.shape)
    print("\nClass Distribution:")
    print(df['churn_risk'].value_counts(normalize=True))
    print("\nMissing Values:")
    print(df.isnull().sum())
    
def preprocess_data(df):
    try:
        print("Starting preprocessing...")
        df_processed = df.copy()
        
        # 1. Handle missing values
        numeric_cols = ['monthly_charges', 'bandwidth_mb', 'avg_monthly_gb_usage', 
                       'customer_rating', 'support_tickets_opened']
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # 2. Feature Engineering
        # Customer Value Metrics
        df_processed['price_per_mb'] = df_processed['monthly_charges'] / df_processed['bandwidth_mb']
        df_processed['usage_efficiency'] = df_processed['avg_monthly_gb_usage'] / df_processed['bandwidth_mb']
        
        # Customer Satisfaction Metrics
        df_processed['satisfaction_index'] = (df_processed['customer_rating'] * 10) / (1 + df_processed['support_tickets_opened'])
        df_processed['support_ticket_ratio'] = df_processed['support_tickets_opened'] / df_processed['customer_rating']
        
        # Value for Money Metrics
        df_processed['value_score'] = (df_processed['bandwidth_mb'] * df_processed['customer_rating']) / df_processed['monthly_charges']
        
        # Usage Patterns
        df_processed['usage_intensity'] = df_processed['avg_monthly_gb_usage'] / df_processed['bandwidth_mb']
        
        # Additional Features
        df_processed['efficiency_rating'] = df_processed['customer_rating'] * df_processed['usage_efficiency']
        df_processed['cost_per_rating'] = df_processed['monthly_charges'] / df_processed['customer_rating']
        
        # 3. Categorical Encoding
        le_dict = {}
        categorical_cols = ['service_plan', 'connection_type', 'churn_risk']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            le_dict[col] = le
        
        # Save encoders
        joblib.dump(le_dict, 'models/label_encoders.pkl')
        
        # 4. Feature Selection
        features = [
            'service_plan', 'connection_type', 
            'monthly_charges', 'bandwidth_mb',
            'avg_monthly_gb_usage', 'customer_rating',
            'support_tickets_opened', 'price_per_mb',
            'usage_efficiency', 'satisfaction_index',
            'support_ticket_ratio', 'value_score',
            'usage_intensity', 'efficiency_rating',
            'cost_per_rating'
        ]
        
        X = df_processed[features]
        y = df_processed['churn_risk']
        
        # 5. Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # After preprocessing but before saving the scaler
        print("Feature order during training:", features)
        
        # Save scaler
        joblib.dump(scaler, 'models/scaler.pkl')
        
        return X_scaled, y, features
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

def train_model():
    try:
        # Create models directory if it doesn't exist
        import os
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # 1. Load and analyze data
        print("Loading data...")
        df = pd.read_csv('customer_isp.csv')
        analyze_data(df)
        
        # 2. Preprocess data
        X_scaled, y, features = preprocess_data(df)
        
        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # 5. Initialize XGBoost with optimized parameters
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0.1,
            random_state=42,
            eval_metric=['mlogloss', 'merror']
        )
        
        # 6. Train model
        print("\nTraining XGBoost model...")
        eval_set = [(X_test, y_test)]
        model.fit(
            X_train_balanced, 
            y_train_balanced,
            eval_set=eval_set,
            verbose=True
        )
        
        # 7. Save model
        joblib.dump(model, 'models/best_model.pkl')
        
        # 8. Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # 9. Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        })
        print("\nTop 10 Most Important Features:")
        print(feature_importance.sort_values('importance', ascending=False).head(10))
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()