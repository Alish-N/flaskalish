import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf	
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

def generate_complex_isp_data(n_rows=10000):
    np.random.seed(42)
    
    # Timestamp generation with minute-level granularity
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=n_rows)
    
    # Basic Features
    data = {
        'timestamp': timestamps,
        'customer_id': np.random.randint(1000, 9999, n_rows),
        'plan': np.random.choice(['Basic', 'Standard', 'Premium', 'Enterprise'], n_rows, 
                               p=[0.3, 0.4, 0.2, 0.1]),
        
        # Network Performance Metrics
        'download_speed': np.random.normal(100, 15, n_rows),
        'upload_speed': np.random.normal(50, 10, n_rows),
        'latency': np.random.normal(20, 5, n_rows),
        'packet_loss': np.random.normal(0.5, 0.2, n_rows),
        'jitter': np.random.normal(5, 1, n_rows),
        'throughput': np.random.normal(80, 10, n_rows),
        
        # Usage Patterns
        'data_usage': np.random.normal(500, 150, n_rows),
        'peak_hours_usage': np.random.normal(60, 15, n_rows),
        'streaming_usage': np.random.normal(40, 10, n_rows),
        'gaming_usage': np.random.normal(20, 8, n_rows),
        'voip_usage': np.random.normal(10, 3, n_rows),
        
        # Network Infrastructure
        'distance_to_node': np.random.normal(2, 0.5, n_rows),
        'node_capacity': np.random.normal(1000, 100, n_rows),
        'fiber_quality': np.random.normal(90, 5, n_rows),
        
        # Environmental Factors
        'temperature': np.random.normal(25, 5, n_rows),
        'humidity': np.random.normal(60, 10, n_rows),
        'weather_condition': np.random.choice(['Clear', 'Rain', 'Storm'], n_rows),
        
        # Service Quality
        'downtime_minutes': np.random.exponential(30, n_rows),
        'packet_retransmission': np.random.normal(1, 0.3, n_rows),
        'dns_latency': np.random.normal(10, 2, n_rows),
        'connection_stability': np.random.normal(95, 2, n_rows),
        
        # Customer Metrics
        'ticket_count': np.random.poisson(2, n_rows),
        'customer_tenure': np.random.normal(24, 12, n_rows),
        'customer_satisfaction': np.random.normal(8, 1, n_rows).round(),
        'payment_history': np.random.normal(95, 3, n_rows),
        
        # Financial Metrics
        'monthly_revenue': np.zeros(n_rows),  # Will be calculated based on plan
        'service_costs': np.random.normal(30, 5, n_rows),
        'maintenance_costs': np.random.normal(15, 3, n_rows)
    }
    
    # Calculate monthly revenue based on plan
    plan_prices = {
        'Basic': (50, 5),
        'Standard': (80, 8),
        'Premium': (120, 12),
        'Enterprise': (200, 20)
    }
    
    for plan, (base_price, std) in plan_prices.items():
        mask = data['plan'] == plan
        data['monthly_revenue'][mask] = np.random.normal(base_price, std, mask.sum())
    
    # Create complex issue flags based on multiple conditions
    data['network_issues'] = np.where(
        (data['packet_loss'] > 1) |
        (data['latency'] > 30) |
        (data['jitter'] > 7) |
        (data['connection_stability'] < 90) |
        (data['packet_retransmission'] > 1.5) |
        ((data['weather_condition'] == 'Storm') & (data['connection_stability'] < 93)),
        1, 0
    )
    
    return pd.DataFrame(data)

# Advanced Preprocessing
def preprocess_data(df):
    df_processed = df.copy()
    
    # Time-based feature extraction
    df_processed['hour'] = df_processed['timestamp'].dt.hour
    df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
    df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
    df_processed['is_peak_hour'] = df_processed['hour'].between(17, 23).astype(int)
    
    # Weather encoding
    weather_mapping = {'Clear': 0, 'Rain': 1, 'Storm': 2}
    df_processed['weather_encoded'] = df_processed['weather_condition'].map(weather_mapping)
    
    # Plan encoding
    plan_mapping = {'Basic': 0, 'Standard': 1, 'Premium': 2, 'Enterprise': 3}
    df_processed['plan_encoded'] = df_processed['plan'].map(plan_mapping)
    
    # Feature engineering
    df_processed['network_load'] = df_processed['data_usage'] / df_processed['node_capacity']
    df_processed['revenue_per_gb'] = df_processed['monthly_revenue'] / df_processed['data_usage']
    df_processed['service_efficiency'] = (df_processed['connection_stability'] * 
                                        df_processed['throughput'] / 
                                        df_processed['latency'])
    
    # Select features for model
    feature_columns = [
        'download_speed', 'upload_speed', 'latency', 'packet_loss', 'jitter',
        'throughput', 'data_usage', 'peak_hours_usage', 'streaming_usage',
        'gaming_usage', 'voip_usage', 'distance_to_node', 'node_capacity',
        'fiber_quality', 'temperature', 'humidity', 'weather_encoded',
        'downtime_minutes', 'packet_retransmission', 'dns_latency',
        'connection_stability', 'customer_tenure', 'payment_history',
        'network_load', 'revenue_per_gb', 'service_efficiency',
        'hour', 'day_of_week', 'is_weekend', 'is_peak_hour', 'plan_encoded'
    ]
    
    # Scale features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_processed[feature_columns])
    y = df_processed['network_issues']
    
    return X, y, scaler, feature_columns

# Deep Learning Model
def create_deep_learning_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

# Main execution
if __name__ == "__main__":
    # Generate complex dataset
    print("Generating complex ISP data...")
    df = generate_complex_isp_data(10000)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y, scaler, feature_columns = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    print("\nTraining deep learning model...")
    model = create_deep_learning_model(X_train.shape[1])
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save model and scaler
    model.save('isp_network_predictor.h5')
    import joblib
    joblib.dump(scaler, 'feature_scaler.joblib')
    
    # Feature importance analysis using permutation importance
    from sklearn.inspection import permutation_importance
    
    # Replace the feature importance analysis section with this:

    # Feature importance analysis using a custom approach
    def get_feature_importance(model, X, feature_columns):
        base_pred = model.predict(X)
        importance_scores = []
        
        for i in range(X.shape[1]):
            # Create a copy and permute one feature
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Get new predictions
            new_pred = model.predict(X_permuted)
            
            # Calculate importance as the mean absolute difference
            importance = np.mean(np.abs(base_pred - new_pred))
            importance_scores.append(importance)
        
        return importance_scores

    # Calculate feature importance
    print("\nCalculating feature importance...")
    importance_scores = get_feature_importance(model, X_test, feature_columns)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("\nFeature importance saved to feature_importance.csv")

    # Visualize feature importance
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance['feature'].head(10), feature_importance['importance'].head(10))
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')

    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Additional metrics
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # Get predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

