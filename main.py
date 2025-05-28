from GNN_tasks import run_classificaton_with_SGNN, run_clustering_with_SGNN, run_classification_with_SGC
import json
from utils import set_arg_parser, get_logger
import torch
import torch.multiprocessing as mp
import utils


def run_experiment(cuda_num, exp_times, config, dataset_decision, model_decision, task_type, is_DDP, experiment_name, logger=None):
    accuracy_list = []
    efficiency_list = []
    nmi_list = []
    time_taken_list = []
    total_accuracy = 0
    total_efficiency = 0
    total_time_taken = 0
    total_nmi = 0
    world_size = None
    return_queue = None

    if is_DDP:
        world_size = torch.cuda.device_count()
        mp.set_start_method("spawn", force=True)
        return_queue = mp.Queue()

    logger.info(f"Starting experiment: {experiment_name}")
    for time in range(exp_times):
        accuracy = 0
        efficiency = 0
        nmi = 0
        time_taken = 0
        logger.info(f"Running trial {time + 1} of {exp_times} for experiment '{experiment_name}'")
        if task_type == 'Clustering':
            if model_decision == 'SGNN':
                accuracy, efficiency, nmi, dataset_name = run_clustering_with_SGNN(dataset_decision, config)
            accuracy_list.append(accuracy)
            efficiency_list.append(efficiency)
            nmi_list.append(nmi)
        elif task_type == 'Classification':
            if model_decision == 'SGNN':
                if is_DDP:
                    mp.spawn(run_classificaton_with_SGNN, args=(world_size, dataset_decision, config, return_queue),
                             nprocs=world_size, join=True)
                    accuracy, efficiency, time_taken = return_queue.get()
                else:
                    accuracy, efficiency, time_taken = run_classificaton_with_SGNN(world_size, cuda_num,
                                                                                   dataset_decision, config,
                                                                                   return_queue)
            elif model_decision == 'SGC':
                if is_DDP:
                    mp.spawn(run_classification_with_SGC, args=(world_size, dataset_decision, config, return_queue),
                             nprocs=world_size, join=True)
                    accuracy, efficiency, time_taken = return_queue.get()
                else:
                    accuracy, efficiency, time_taken = run_classification_with_SGC(world_size, cuda_num,
                                                                                   dataset_decision, config,
                                                                                   return_queue)
            else:
                exit()
            accuracy_list.append(accuracy)
            efficiency_list.append(efficiency)
            time_taken_list.append(time_taken)

        total_accuracy += accuracy
        total_efficiency += efficiency
        total_time_taken += time_taken
        total_nmi += nmi

    average_accuracy = total_accuracy / exp_times
    average_efficiency = total_efficiency / exp_times
    average_time_taken = total_time_taken / exp_times
    average_nmi = total_nmi / exp_times

    logger.info(f"\nEXPERIMENT RESULTS for '{experiment_name}'")
    logger.info(f'Dataset used: {dataset_decision}')
    logger.info(f'Model used: {model_decision}')
    logger.info(f'Task type: {task_type}')
    logger.info(f'Experiment count: {exp_times}')
    logger.info(f"All the accuracies: {accuracy_list}")
    logger.info(f"All the efficiencies: {efficiency_list}")
    logger.info(f"All the times taken: {time_taken_list}")
    logger.info(f"All the nmi: {nmi_list}")
    logger.info(f"The average accuracy is: {average_accuracy}")
    logger.info(f"The average efficiency is: {average_efficiency}")
    logger.info(f"The average time taken is: {average_time_taken}")
    logger.info(f"The average nmi is: {average_nmi}")

    return average_accuracy, average_efficiency, average_nmi, average_time_taken


def main_multiple_experiments(experiments, logger):
    for i, exp in enumerate(experiments):
        experiment_name = exp.get('experiment_name', f"Experiment_{i+1}")
        logger.info(f"\n===== Running experiment {i+1} of {len(experiments)}: {experiment_name} =====")
        cuda_num = exp.get('cuda_num')
        dataset_decision = exp.get('dataset')
        model_decision = exp.get('model')
        task_type = exp.get('task_type')
        exp_times = exp.get('exp_times', 1)
        isTuning = exp.get('is_tuning')
        is_ddp = exp.get('is_ddp', False)
        mm_op = exp.get('multi_model_operation')
        mm_structure = exp.get('multi_model_structure')
        tuning_params = exp.get('tuning_parameters', None)

        main_single_experiment(cuda_num, dataset_decision, model_decision, task_type, exp_times, isTuning, is_ddp,
                               mm_op, mm_structure, experiment_name, logger=logger, tuning_params=tuning_params)


def main_single_experiment(cuda_num, dataset_decision, model_decision, task_type, exp_times, isTuning, is_ddp,
                           mm_op, mm_structure, experiment_name, logger=None, tuning_params=None):
    with open('./config.json', 'r') as file:
        settings = json.load(file)
    dataset_config = settings[model_decision][task_type][dataset_decision]

    if isTuning is None:
        if mm_op is not None and mm_structure is not None:
            dataset_config = utils.modify_and_return_config(dataset_config, mm_structure, mm_op)
        logger.info(json.dumps(dataset_config, indent=4))
        run_experiment(cuda_num, exp_times, dataset_config, dataset_decision, model_decision, task_type, is_ddp,
                       experiment_name, logger=logger)
    else:
        # Load hyperparameter ranges
        with open("ranges.json", "r") as f:
            hyperparam_data = json.load(f)["Test"]

        # Generate all combinations of hyperparameters
        all_combinations = utils.generate_hyperparam_combinations(dataset_config, hyperparam_data, tuning_params=tuning_params)

        # Ensure we don't exceed the number of combinations
        isTuning = min(isTuning, len(all_combinations))
        logger.info(f"Total configurations available: {len(all_combinations)}")
        logger.info(f"Running {isTuning} tuning iterations.")

        tuning_results = []  # To store configurations and results

        for i, config in enumerate(all_combinations[:isTuning], 1):
            logger.info(f"\n=======\nRunning tuning iteration {i}/{isTuning} for '{experiment_name}'\n=======")
            logger.info(json.dumps(config, indent=4))
            average_accuracy, average_efficiency, average_nmi, average_time_taken = run_experiment(
                cuda_num, exp_times, config, dataset_decision, model_decision, task_type, is_ddp, experiment_name,
                logger=logger)

            # Record results with the configuration
            tuning_results.append({
                "config": config,
                "average_accuracy": average_accuracy,
                "average_efficiency": average_efficiency,
                "average_nmi": average_nmi,
                "average_time_taken": average_time_taken
            })

        # Log results
        logger.info(f"\nFINAL TUNING RESULTS for '{experiment_name}'")
        for i, result in enumerate(tuning_results, 1):
            logger.info(f"Experiment {i}:")
            logger.info(f"Configuration: {json.dumps(result['config'], indent=4)}")
            logger.info(f"Average Accuracy: {result['average_accuracy']}")
            logger.info(f"Average Efficiency: {result['average_efficiency']}")
            logger.info(f"Average NMI: {result['average_nmi']}")
            logger.info(f"Average Time Taken: {result['average_time_taken']}\n")

        # Log best result
        best_result = max(tuning_results, key=lambda x: x["average_accuracy"])
        logger.info(f"Best configuration by accuracy: {json.dumps(best_result['config'], indent=4)}")
        logger.info(f"Accuracy: {best_result['average_accuracy']}")


if __name__ == "__main__":
    (cuda_num, dataset_decision, model_decision, task_type, exp_times,
     logPath, isTuning, ddp, mm_op, mm_structure, log_level, experiments_file) = set_arg_parser()

    logger_settings = {
        "logger": {
            "model": model_decision,
            "log_path": logPath,
            "dataset": dataset_decision,
            "log_level": log_level.upper()
        },
        "ddp": ddp
    }

    with open("global_settings.json", "w") as file:
        json.dump(logger_settings, file, indent=4)

    logger = get_logger()

    logger.info(f"CUDA num: {cuda_num}")
    logger.info(f"DDP: {ddp}")
    logger.info(f"Log level: {log_level}")

    if experiments_file:
        # Load multiple experiments from JSON file and run them all
        with open(experiments_file, 'r') as f:
            experiments = json.load(f)  # Ensure this line parses correctly
            if not isinstance(experiments, dict) or 'experiments' not in experiments:
                logger.error("Invalid JSON structure in experiments file.")
                exit(1)
            experiments = experiments['experiments']  # Extract the list of experiments
        main_multiple_experiments(experiments, logger)

    else:
        # Run the single experiment
        experiment_name = f"Single_Experiment_{task_type}_{model_decision}_{dataset_decision}"
        logger.info(f"Dataset: {dataset_decision}")
        logger.info(f"Model: {model_decision}")
        logger.info(f"Task: {task_type}")
        logger.info(f"Number of experiments: {exp_times}")
        logger.info(f"Multi-Model Operation: {mm_op}")
        logger.info(f"Multi-Model Structure: {mm_structure}")

        main_single_experiment(cuda_num, dataset_decision, model_decision, task_type, exp_times, isTuning, ddp, mm_op,
                               mm_structure, experiment_name, logger=logger)

