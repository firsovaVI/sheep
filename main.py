import os
import json
import numpy as np
import matplotlib.pyplot as plt
from optimizer import HybridOptimizer
from data_utils import generate_synthetic_data, load_breed_data
from model_comparison import compare_models
from matplotlib.gridspec import GridSpec


def numpy_to_list(obj):
    """Рекурсивно преобразует numpy массивы в списки для JSON сериализации"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_list(x) for x in obj]
    return obj


def brody_model(params, age):
    A, B, k = params
    return A * (1 - B * np.exp(-k * age))


def gompertz_model(params, age):
    A, B, k = params
    return A * np.exp(-B * np.exp(-k * age))


def logistic_model(params, age):
    A, B, k = params
    return A / (1 + B * np.exp(-k * age))


def negative_exponential_model(params, age):
    A, k = params
    return A * (1 - np.exp(-k * age))


def richards_model(params, age):
    A, B, k, m = params
    return A * (1 - B * np.exp(-k * age)) ** (1 / m)


def von_bertalanffy_model(params, age):
    A, B, k = params
    return A * (1 - B * np.exp(-k * age)) ** 3


def objective_function(params, target_weights, age_points, model_type="brody"):
    try:
        predicted_weights = []
        params = np.array(params)

        for age in age_points:
            if model_type == "brody":
                predicted = brody_model(params, age)
            elif model_type == "gompertz":
                predicted = gompertz_model(params, age)
            elif model_type == "logistic":
                predicted = logistic_model(params, age)
            elif model_type == "negative_exponential":
                predicted = negative_exponential_model(params, age)
            elif model_type == "richards":
                predicted = richards_model(params, age)
            elif model_type == "von_bertalanffy":
                predicted = von_bertalanffy_model(params, age)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            predicted_weights.append(predicted)

        errors = np.array(predicted_weights) - np.array(target_weights)
        rmse = np.sqrt(np.mean(errors ** 2))
        sd = np.std(errors)

        return rmse, sd
    except Exception as e:
        print(f"Error in objective_function: {str(e)}")
        return float('inf'), float('inf')


def plot_optimization_history(log_file):
    try:
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"File {log_file} not found.")

        iterations = []
        rmse_values = []
        sd_values = []

        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    iterations.append(data['iteration'])
                    rmse_values.append(data['best_rmse'])
                    if 'best_sd' in data:
                        sd_values.append(data['best_sd'])
                except json.JSONDecodeError:
                    continue

        plt.figure(figsize=(12, 6))
        plt.plot(iterations, rmse_values, 'b-', label='RMSE')
        if sd_values:
            plt.plot(iterations, sd_values, 'r--', label='SD')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Optimization History')
        plt.legend()
        plt.grid(True)
        plt.savefig('optimization_history.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error plotting optimization history: {e}")


def plot_growth_curves(results, target_weights, age_points):
    try:
        plt.figure(figsize=(12, 8))
        extended_ages = np.linspace(min(age_points), max(age_points), 100)

        # Plot target weights
        plt.scatter(age_points, target_weights, c='black', s=100,
                    label='Target Data', zorder=5)

        # Plot model curves
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for (model_name, data), color in zip(results.items(), colors):
            params = data['params']
            if model_name == "brody":
                curve = [brody_model(params, age) for age in extended_ages]
            elif model_name == "gompertz":
                curve = [gompertz_model(params, age) for age in extended_ages]
            elif model_name == "logistic":
                curve = [logistic_model(params, age) for age in extended_ages]
            elif model_name == "negative_exponential":
                curve = [negative_exponential_model(params[:2], age) for age in extended_ages]
            elif model_name == "richards":
                curve = [richards_model(params, age) for age in extended_ages]
            elif model_name == "von_bertalanffy":
                curve = [von_bertalanffy_model(params, age) for age in extended_ages]

            plt.plot(extended_ages, curve, color=color, linewidth=2,
                     label=f"{model_name} (RMSE={data['rmse']:.2f}, SD={data['sd']:.2f})")

        plt.xlabel('Age (days)')
        plt.ylabel('Weight (kg)')
        plt.title('Sheep Growth Models Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('growth_curves_comparison.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error plotting growth curves: {e}")


def main():
    try:

        breed_name = "Santa_Ines"
        sex = "m"
        age_points = np.array([1, 60, 180, 210, 360])  # Ensure this is numpy array


        print("Loading data...")
        breed_data = load_breed_data(breed_name, sex)
        if not breed_data:
            breed_data = {'mean': [3.50, 13.50, 22.70, 24.80, 24.90],
                          'std': [0.35, 1.35, 2.27, 2.48, 2.49]}
            print(f"Using default data for {breed_name} {sex}")

        synthetic_data = generate_synthetic_data(
            breed_data['mean'],
            breed_data['std'],
            n_samples=100
        )
        target_weights = np.mean(synthetic_data, axis=0)


        models_to_compare = {
            "brody": {
                "function": brody_model,
                "bounds": [(20, 50), (0.5, 1.5), (0.01, 0.1)]
            },
            "gompertz": {
                "function": gompertz_model,
                "bounds": [(20, 50), (1, 5), (0.01, 0.1)]
            },
            "logistic": {
                "function": logistic_model,
                "bounds": [(20, 50), (1, 5), (0.01, 0.1)]
            }
        }


        results = {}
        for model_name, model_info in models_to_compare.items():
            print(f"\n=== Optimizing {model_name} model ===")

            optimizer = HybridOptimizer(
                lambda params: objective_function(params, target_weights,
                                                  age_points, model_name),
                model_info["bounds"],
                log_file=f"optimization_{model_name}.json"
            )

            best_params, (best_rmse, best_sd) = optimizer.optimize(max_iterations=50)

            results[model_name] = {
                "params": numpy_to_list(best_params),
                "rmse": float(best_rmse),
                "sd": float(best_sd),
                "predictions": numpy_to_list(
                    [model_info["function"](best_params, age) for age in age_points]
                )
            }

            print(f"\nResults for {model_name}:")
            print(f"Parameters: {best_params}")
            print(f"RMSE: {best_rmse:.4f} kg")
            print(f"Standard deviation: {best_sd:.4f} kg")

            plot_optimization_history(f"optimization_{model_name}.json")


        print("\n=== Model Comparison ===")
        model_predictions = {name: data["predictions"] for name, data in results.items()}
        compare_models(model_predictions, target_weights)


        plot_growth_curves(results, target_weights, age_points)


        with open("final_results.json", "w") as f:
            json.dump(numpy_to_list({
                "breed": breed_name,
                "sex": sex,
                "age_points": age_points,
                "target_weights": target_weights,
                "results": results
            }), f, indent=4)

        print("\nAnalysis completed successfully. Results saved to final_results.json")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")


if __name__ == "__main__":
    main()
