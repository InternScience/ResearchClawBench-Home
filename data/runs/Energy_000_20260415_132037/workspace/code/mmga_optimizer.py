"""
MMGA (Multi-objective Modified Genetic Algorithm) for Battery Parameter Identification
"""

import numpy as np
from scipy.interpolate import interp1d

class MMGAOptimizer:
    """
    Multi-objective Modified Genetic Algorithm for parameter identification
    Optimizes both voltage error and capacity error simultaneously
    """
    
    def __init__(self, ann_model, param_bounds, pop_size=100, 
                 n_generations=200, crossover_rate=0.8, mutation_rate=0.1,
                 elitism_ratio=0.1):
        """
        Initialize MMGA optimizer
        
        Parameters:
        -----------
        ann_model : ANNMetaModel
            Trained ANN meta-model for fast predictions
        param_bounds : dict
            Parameter bounds {name: (min, max)}
        pop_size : int
            Population size
        n_generations : int
            Number of generations
        crossover_rate : float
            Probability of crossover
        mutation_rate : float
            Probability of mutation
        elitism_ratio : float
            Fraction of elite individuals to preserve
        """
        self.ann_model = ann_model
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        
        # Convergence history
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'pareto_front': [],
            'best_params': []
        }
    
    def initialize_population(self):
        """Initialize random population within bounds"""
        population = np.zeros((self.pop_size, self.n_params))
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[name]
            population[:, i] = np.random.uniform(low, high, self.pop_size)
        return population
    
    def decode_parameters(self, individual):
        """Decode individual to parameter dictionary"""
        return {name: individual[i] for i, name in enumerate(self.param_names)}
    
    def evaluate_fitness(self, individual, target_features):
        """
        Evaluate multi-objective fitness
        
        Objectives:
        1. Voltage error (RMSE)
        2. Capacity error
        """
        # Predict using ANN meta-model
        X = individual.reshape(1, -1)
        predicted = self.ann_model.predict(X)[0]
        
        # Extract predicted features
        n_soc_points = len(target_features) - 2
        v_pred = predicted[:n_soc_points]
        cap_pred = predicted[n_soc_points]
        temp_pred = predicted[n_soc_points + 1]
        
        # Extract target features
        v_target = target_features[:n_soc_points]
        cap_target = target_features[n_soc_points]
        temp_target = target_features[n_soc_points + 1]
        
        # Calculate errors
        voltage_error = np.sqrt(np.mean((v_pred - v_target)**2))
        capacity_error = np.abs(cap_pred - cap_target) / (cap_target + 1e-6)
        
        return np.array([voltage_error, capacity_error])
    
    def non_dominated_sort(self, objectives):
        """
        Non-dominated sorting for multi-objective optimization
        Returns Pareto fronts (ranks)
        """
        n = len(objectives)
        domination_count = np.zeros(n)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                obj_i = objectives[i]
                obj_j = objectives[j]
                
                # Check dominance
                dominates_ij = np.all(obj_i <= obj_j) and np.any(obj_i < obj_j)
                dominates_ji = np.all(obj_j <= obj_i) and np.any(obj_j < obj_i)
                
                if dominates_ij:
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif dominates_ji:
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        fronts.pop()  # Remove empty last front
        return fronts
    
    def crowding_distance(self, objectives, front):
        """Calculate crowding distance for diversity preservation"""
        if len(front) <= 2:
            return np.full(len(front), np.inf)
        
        distances = np.zeros(len(front))
        n_obj = objectives.shape[1]
        
        for m in range(n_obj):
            sorted_idx = np.argsort(objectives[front, m])
            sorted_front = [front[i] for i in sorted_idx]
            
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            obj_range = objectives[sorted_front[-1], m] - objectives[sorted_front[0], m]
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distances[sorted_idx[i]] += (
                        objectives[sorted_front[i + 1], m] - 
                        objectives[sorted_front[i - 1], m]
                    ) / obj_range
        
        return distances
    
    def tournament_selection(self, population, fitness, tournament_size=3):
        """Tournament selection for parent selection"""
        selected = []
        for _ in range(2):  # Select 2 parents
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = fitness[tournament_idx]
            # For minimization, select lowest fitness
            winner_idx = tournament_idx[np.argmin(np.sum(tournament_fitness, axis=1))]
            selected.append(population[winner_idx])
        return selected
    
    def crossover(self, parent1, parent2):
        """Simulated binary crossover (SBX)"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        eta_c = 20  # Distribution index
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            if np.random.random() <= 0.5:
                if np.abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    beta = 1.0 + (2.0 * (y1 - self.param_bounds[self.param_names[i]][0]) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta_c + 1))
                    rand = np.random.random()
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
                    
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    
                    beta = 1.0 + (2.0 * (self.param_bounds[self.param_names[i]][1] - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta_c + 1))
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
                    
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    child1[i] = np.clip(c1, self.param_bounds[self.param_names[i]][0], 
                                       self.param_bounds[self.param_names[i]][1])
                    child2[i] = np.clip(c2, self.param_bounds[self.param_names[i]][0], 
                                       self.param_bounds[self.param_names[i]][1])
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        
        return child1, child2
    
    def mutate(self, individual):
        """Polynomial mutation"""
        eta_m = 20  # Distribution index
        
        for i in range(len(individual)):
            if np.random.random() <= self.mutation_rate:
                y = individual[i]
                y_low, y_high = self.param_bounds[self.param_names[i]]
                delta1 = (y - y_low) / (y_high - y_low)
                delta2 = (y_high - y) / (y_high - y_low)
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                    delta_q = 1.0 - val ** mut_pow
                
                y = y + delta_q * (y_high - y_low)
                individual[i] = np.clip(y, y_low, y_high)
        
        return individual
    
    def optimize(self, target_features, verbose=True):
        """
        Run MMGA optimization
        
        Parameters:
        -----------
        target_features : ndarray
            Target discharge curve features from experimental data
        verbose : bool
            Print progress
            
        Returns:
        --------
        best_params : dict
            Best identified parameters
        pareto_front : list
            Pareto-optimal solutions
        """
        # Initialize population
        population = self.initialize_population()
        
        for generation in range(self.n_generations):
            # Evaluate fitness for all individuals
            objectives = np.array([self.evaluate_fitness(ind, target_features) 
                                  for ind in population])
            
            # Non-dominated sorting
            fronts = self.non_dominated_sort(objectives)
            
            # Calculate crowding distance
            crowding_distances = np.zeros(len(population))
            for front in fronts:
                if len(front) > 0:
                    distances = self.crowding_distance(objectives, front)
                    for i, idx in enumerate(front):
                        crowding_distances[idx] = distances[i]
            
            # Store best fitness
            first_front = fronts[0]
            best_fitness = np.min(objectives[first_front], axis=0)
            avg_fitness = np.mean(objectives, axis=0)
            
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            
            # Store Pareto front
            pareto_objectives = objectives[first_front]
            pareto_solutions = population[first_front]
            self.history['pareto_front'].append({
                'objectives': pareto_objectives,
                'solutions': pareto_solutions
            })
            
            if verbose and generation % 20 == 0:
                print(f"Generation {generation}: "
                      f"Best Voltage Error = {best_fitness[0]:.4f} V, "
                      f"Best Capacity Error = {best_fitness[1]:.4f}")
            
            # Create offspring
            offspring = []
            
            # Elitism: keep best individuals
            n_elite = int(self.elitism_ratio * self.pop_size)
            elite_indices = first_front[:n_elite]
            for idx in elite_indices:
                offspring.append(population[idx].copy())
            
            # Generate rest through crossover and mutation
            while len(offspring) < self.pop_size:
                # Parent selection
                parent1, parent2 = self.tournament_selection(population, objectives)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                offspring.append(child1)
                if len(offspring) < self.pop_size:
                    offspring.append(child2)
            
            population = np.array(offspring[:self.pop_size])
        
        # Final evaluation
        final_objectives = np.array([self.evaluate_fitness(ind, target_features) 
                                     for ind in population])
        final_fronts = self.non_dominated_sort(final_objectives)
        
        # Select best solution (minimum combined error)
        first_front = final_fronts[0]
        combined_error = np.sum(final_objectives[first_front], axis=1)
        best_idx = first_front[np.argmin(combined_error)]
        best_params = self.decode_parameters(population[best_idx])
        
        self.history['best_params'].append(best_params)
        
        return best_params, {
            'pareto_objectives': final_objectives[first_front],
            'pareto_solutions': population[first_front]
        }
