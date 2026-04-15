"""
ANN Meta-Model for Battery Parameter Identification
Replaces expensive physics-based simulations with fast neural network predictions
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

class ANNMetaModel:
    """
    Artificial Neural Network meta-model for battery discharge curve prediction
    Maps parameter space to voltage-time characteristics
    """
    
    def __init__(self, hidden_layers=(128, 64, 32), activation='relu', 
                 max_iter=2000, early_stopping=True):
        """
        Initialize ANN meta-model
        
        Parameters:
        -----------
        hidden_layers : tuple
            Number of neurons in each hidden layer
        activation : str
            Activation function ('relu', 'tanh', 'logistic')
        """
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        self.is_trained = False
        self.training_history = {'loss': [], 'val_loss': []}
        
    def generate_lhs_samples(self, param_bounds, n_samples):
        """
        Generate Latin Hypercube Sampling (LHS) parameter combinations
        
        Parameters:
        -----------
        param_bounds : dict
            Dictionary with parameter names and (min, max) bounds
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        samples : ndarray
            LHS samples of shape (n_samples, n_params)
        param_names : list
            List of parameter names in order
        """
        param_names = list(param_bounds.keys())
        n_params = len(param_names)
        
        # Generate LHS samples
        samples = np.zeros((n_samples, n_params))
        
        for i in range(n_params):
            # Stratified sampling
            bounds = param_bounds[param_names[i]]
            cut = np.linspace(bounds[0], bounds[1], n_samples + 1)
            samples[:, i] = cut[:-1] + np.random.uniform(0, cut[1] - cut[0], n_samples)
            np.random.shuffle(samples[:, i])
        
        return samples, param_names
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the ANN meta-model
        
        Parameters:
        -----------
        X_train : ndarray
            Training input parameters (n_samples, n_params)
        y_train : ndarray
            Training output features (n_samples, n_features)
        """
        # Scale inputs and outputs
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train)
        
        # Train model
        self.model.fit(X_scaled, y_scaled)
        
        # Store training metrics
        self.training_history['loss'] = self.model.loss_curve_
        
        # Calculate validation score if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val)
            val_score = self.model.score(X_val_scaled, y_val_scaled)
            self.training_history['val_score'] = val_score
        
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """
        Predict output features for given parameters
        
        Parameters:
        -----------
        X : ndarray
            Input parameters (n_samples, n_params)
            
        Returns:
        --------
        y_pred : ndarray
            Predicted output features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_scaled)
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        X_scaled = self.scaler_X.transform(X_test)
        y_scaled = self.scaler_y.transform(y_test)
        
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        mse = np.mean((y_pred - y_test)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_test))
        r2 = self.model.score(X_scaled, y_scaled)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def save(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'is_trained': self.is_trained,
                'history': self.training_history
            }, f)
    
    def load(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler_X = data['scaler_X']
            self.scaler_y = data['scaler_y']
            self.is_trained = data['is_trained']
            self.training_history = data['history']
        return self


class ParameterToCurveMapper:
    """
    Maps battery parameters to discharge curve features
    Used for training data generation
    """
    
    def __init__(self, model_simulator):
        """
        Initialize with battery model simulator
        
        Parameters:
        -----------
        model_simulator : object
            Battery model with simulate_discharge method
        """
        self.simulator = model_simulator
        self.soc_points = np.linspace(0, 1, 21)  # 5% SOC intervals
        
    def extract_curve_features(self, simulation_result):
        """
        Extract features from simulation result
        
        Features extracted:
        - Voltage at SOC points (21 values)
        - Discharge capacity (1 value)
        - Temperature rise (1 value)
        """
        t = simulation_result['time']
        v = simulation_result['voltage']
        T = simulation_result['temperature']
        
        # Calculate SOC from time
        soc = 1 - (t - t.min()) / (t.max() - t.min() + 1e-10)
        
        # Sort by SOC for interpolation
        sort_idx = np.argsort(soc)
        soc_sorted = soc[sort_idx]
        v_sorted = v[sort_idx]
        
        # Interpolate voltage at SOC points
        v_features = np.interp(self.soc_points, soc_sorted, v_sorted, 
                               left=v_sorted[0], right=v_sorted[-1])
        
        # Additional features
        capacity = t[-1] * 1.0 / 3600  # Simplified capacity calculation
        temp_rise = T.max() - T.min()
        
        features = np.concatenate([v_features, [capacity, temp_rise]])
        
        return features
    
    def generate_training_data(self, param_samples, param_names, 
                               current=1.0, T_sim=3600):
        """
        Generate training data by running simulations
        
        Parameters:
        -----------
        param_samples : ndarray
            Array of parameter combinations
        param_names : list
            List of parameter names
        current : float
            Discharge current (A)
        T_sim : float
            Simulation time (s)
            
        Returns:
        --------
        X : ndarray
            Parameter inputs
        y : ndarray
            Curve features outputs
        """
        n_samples = param_samples.shape[0]
        n_features = len(self.soc_points) + 2
        
        X = param_samples
        y = np.zeros((n_samples, n_features))
        
        for i, params in enumerate(param_samples):
            # Set parameters
            param_dict = {name: val for name, val in zip(param_names, params)}
            self.simulator.set_params(param_dict)
            
            # Run simulation
            try:
                result = self.simulator.simulate_discharge(current, T_sim)
                features = self.extract_curve_features(result)
                y[i] = features
            except Exception as e:
                # If simulation fails, use default features
                y[i] = np.zeros(n_features)
                y[i][:len(self.soc_points)] = 3.7  # Default voltage
        
        return X, y
