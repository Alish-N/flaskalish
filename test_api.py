# test_api.py
import requests
import json
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

def format_probability(prob):
    return f"{prob * 100:.2f}%"

def test_prediction():
    # Test data matching the features used in model training
    test_data = {
        "service_plan": "Basic",
        "connection_type": "DSL",
        "monthly_charges": 99.16,
        "bandwidth_mb": 10,
        "avg_monthly_gb_usage": 1230.99,
        "customer_rating": 4,
        "support_tickets_opened": 15
    }
    
    try:
        # First test health check
        health_response = requests.get('http://localhost:5000/health')
        print(f"\n{Fore.CYAN}=== Health Check ==={Style.RESET_ALL}")
        print(f"Status: {Fore.GREEN if health_response.status_code == 200 else Fore.RED}"
              f"{health_response.status_code}{Style.RESET_ALL}")
        print(f"Response: {health_response.json()}")
        
        # Test prediction endpoint
        print(f"\n{Fore.CYAN}=== Making Prediction ==={Style.RESET_ALL}")
        print(f"Input Data: {json.dumps(test_data, indent=2)}\n")
        
        response = requests.post('http://localhost:5000/predict', 
                               json=test_data,
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status Code: {Fore.GREEN if response.status_code == 200 else Fore.RED}"
              f"{response.status_code}{Style.RESET_ALL}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n{Fore.YELLOW}Prediction Results:{Style.RESET_ALL}")
            print(f"Churn Risk Level: {Fore.CYAN}{result['churn_risk']}{Style.RESET_ALL}")
            print(f"\nProbabilities:")
            print(f"├── Low Risk:    {Fore.GREEN}{format_probability(result['probability']['Low'])}{Style.RESET_ALL}")
            print(f"├── Medium Risk: {Fore.YELLOW}{format_probability(result['probability']['Medium'])}{Style.RESET_ALL}")
            print(f"└── High Risk:   {Fore.RED}{format_probability(result['probability']['High'])}{Style.RESET_ALL}")
            print(f"\nConfidence: {Fore.CYAN}{format_probability(result['confidence'])}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Error:{Style.RESET_ALL} {response.json()}")
            
    except Exception as e:
        print(f"{Fore.RED}Error testing API: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    test_prediction()