import os
import pickle
import cvxpy as cp
import copy
import pathlib


# run_fn =  "data/online/ltv/disturbance,noise-0.2,horizon-70,seed-203"
# run = pickle.load(open(run_fn, "rb"))

def correct_step_costs(reference, run):
    out = copy.deepcopy(run)
    for t in range(len(run.step_costs) - 1):  # last one is terminal cost, excluded
        p = reference.opt_datas.parameters[t]
        real_step_cost = cp.quad_form(run.states[t] - p.x_bar, p.Q) + cp.quad_form(run.inputs[t], p.R)
        out.step_costs[t] = real_step_cost.value
    return out

def correct_noisy_prediction_cost():
    out_dir = "./data/online_cost-corrected/ltv"
    os.makedirs(out_dir, exist_ok=True)
    ref_fn = "data/offline/ltv/seed-100"
    reference = pickle.load(open(ref_fn, "rb"))
    files = list(sorted(pathlib.Path('data/online/ltv').glob('*')))
    for f in files:
        print(f"Handling {str(f)}")
        run = pickle.load(open(f, "rb"))
        run = correct_step_costs(reference, run)
        pickle.dump(run, open(os.path.join(out_dir, f.name), "wb"))

def correction_nn_prediction_cost():
    out_dir = "data/neural_networks/run_results_cost-corrected"
    ref_fn = "data/offline/ltv/seed-100"
    reference = pickle.load(open(ref_fn, "rb"))

    files = list(sorted(pathlib.Path('data/neural_networks/run_results').glob('*')))
    for f in files:
        print(f"Handling {str(f)}.")
        run_dict = pickle.load(open(f, "rb"))
        runs = run_dict["mpc_results"]
        evaluation_iterations = run_dict["evaluation_iterations"]
        evaluation_data = run_dict["nn_eval_data"]
        cost_corrected_runs = [correct_step_costs(reference, e) for e in runs]
        f_output_dir = os.path.join(out_dir, f.stem)
        os.makedirs(f_output_dir, exist_ok=True)
        pickle.dump(evaluation_iterations, open(os.path.join(f_output_dir, "evaluation_iterations.pkl"), "wb"))
        pickle.dump(evaluation_data, open(os.path.join(f_output_dir, "evaluation_data.pkl"), "wb"))
        for eval_iter, run in zip(evaluation_iterations, cost_corrected_runs):
            pickle.dump(run, open(os.path.join(f_output_dir, str(eval_iter)), "wb"))

if __name__ == '__main__':
    correct_noisy_prediction_cost()
    correction_nn_prediction_cost()