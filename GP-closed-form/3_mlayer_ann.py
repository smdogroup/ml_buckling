# run_ann_grid.py
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append("src/")
from closed_form_dataset import get_closed_form_data, split_data
from eval_anns import eval_ann
from plot_GPs import plot_GPs
from eval_utils import eval_Rsquared

# -----------------------------
# Argparse
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # dataset / transforms
    p.add_argument("--axial", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--affine", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--log", action=argparse.BooleanOptionalAction, default=False)

    # training setup
    p.add_argument("--activ", type=str, default="relu", choices=["relu", "tanh"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--kfolds", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--random", action="store_false", default=True,
                   help="If passed, disables random (keeps seed deterministic).")

    # data sizing
    p.add_argument("--dataf", type=float, default=1.0,
                   help="fraction of full dataset size (out of 1000) for doing less trials")

    # ANN architecture knobs you wanted
    p.add_argument("--neurons", type=int, default=64,
                   help="Width per layer (used with --nlayers).")
    p.add_argument("--nlayers", type=int, default=1,
                   help="Number of hidden layers.")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout probability after each hidden layer.")
    p.add_argument("--ntrials", type=int, default=5,
                   help="Number of eval trials inside eval_ann().")

    # output
    p.add_argument("--outdir", type=str, default="output")
    p.add_argument("--csv", type=str, default="ann_results.csv",
                   help="CSV file to append results to (created if missing).")
    p.add_argument("--no_plot", action="store_true",
                   help="Skip making prediction plots to save time.")

    return p.parse_args()

args = parse_args()

# -----------------------------
# Reproducibility
# -----------------------------
if not args.random:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

# -----------------------------
# Folder setup
# -----------------------------
os.makedirs(args.outdir, exist_ok=True)

folder_name = os.path.join(args.outdir, f"ANN_{args.activ}")
os.makedirs(folder_name, exist_ok=True)
os.makedirs(os.path.join(folder_name, "opt"), exist_ok=True)

# base name for each file
axial_str = "axial" if args.axial else "shear"
log_str = "log" if args.log else "nolog"
affine_str = "affine" if args.affine else "noaffine"
base_name = f"tf_{axial_str}_{affine_str}_{log_str}_L{args.nlayers}_N{args.neurons}_do{args.dropout:.1f}"

txt_file = os.path.join(folder_name, f"{base_name}.txt")
txt_hdl = open(txt_file, "w")

# -----------------------------
# epochs default (your logic)
# -----------------------------
if args.epochs is None:
    epochs = 400 if args.activ == "tanh" else 100
    if args.activ == "relu":
        epochs = 100 if args.log else 200
else:
    epochs = args.epochs

# -----------------------------
# Build neurons list from nlayers, neurons
# -----------------------------
neurons = [args.neurons] * args.nlayers

# -----------------------------
# Build dataset
# -----------------------------
n_data_mult = args.dataf ** (1.0 / 3.0)
n_rho = int(20 * n_data_mult)
n_gamma = int(10 * n_data_mult)
n_xi = max(int(5 * n_data_mult), 3)
n_data = n_rho * n_gamma * n_xi

print(f"{n_data_mult=} => {n_data=}")

X_interp, Y_interp = get_closed_form_data(
    axial=args.axial,
    include_extrapolation=False,
    affine_transform=args.affine,
    log_transform=args.log,
    n_rho0=n_rho, n_gamma=n_gamma, n_xi=n_xi,
)

# -----------------------------
# Evaluate metrics (R^2)
# -----------------------------
interp_Rsq, extrap_Rsq = eval_ann(
    neurons=neurons,
    epochs=epochs,
    activation=args.activ,
    dropout=args.dropout,
    n_trials=args.ntrials,
    train_test_frac=0.8,
    shear_ks_param=None,
    axial=args.axial,
    affine=args.affine,
    log=args.log,
    n_rho0=n_rho, n_gamma=n_gamma, n_xi=n_xi,
    metric_func=eval_Rsquared,
    percentile=50.0,
)

txt_hdl.write("R^2 metrics:\n")
txt_hdl.write(f"\t{interp_Rsq=}\n\n")
txt_hdl.write(f"\t{extrap_Rsq=}\n\n")
txt_hdl.write("--------------------------------------------\n\n")

# -----------------------------
# Train a model (for nparams + optional plots)
# -----------------------------
X_train, Y_train, X_test, Y_test = split_data(X_interp, Y_interp, train_test_split=0.9)

X_plot, Y_plot = get_closed_form_data(
    axial=args.axial,
    include_extrapolation=True,
    affine_transform=args.affine,
    log_transform=args.log,
    n_rho0=n_rho, n_gamma=n_gamma, n_xi=n_xi,
)

start_time = time.time()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(3,)))

for _ in neurons:
    model.add(tf.keras.layers.Dense(args.neurons, activation=args.activ))
    model.add(tf.keras.layers.Dropout(args.dropout))

model.add(tf.keras.layers.Dense(1))

loss_fn = tf.keras.losses.MSE
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=epochs, verbose=0)
model.evaluate(X_test, Y_test, verbose=0)

train_time_sec = time.time() - start_time
trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))

print(f"{train_time_sec=:.4e}")
print("Num Trainable params:", trainable_params)

# Optional plotting
if not args.no_plot:
    Y_plot_pred = model(X_plot).numpy()
    plot_GPs(
        X_plot, Y_plot, Y_plot_pred,
        folder_name=folder_name,
        base_name=base_name,
        axial=args.axial,
        affine=args.affine,
        log=args.log,
        nx1=n_rho, nx2=n_gamma,
        show=False,
    )

# -----------------------------
# Append to CSV (your schema)
# -----------------------------
row = {
    "load": "axial" if args.axial else "shear",
    "log": "yes" if args.log else "no",
    "affine": "yes" if args.affine else "no",
    "dropout": float(args.dropout),
    "nlayers": int(args.nlayers),
    "neurons": int(args.neurons),
    "nparams": int(trainable_params),
    "intR2": float(interp_Rsq),
    "extR2": float(extrap_Rsq),
}

csv_path = args.csv
write_header = not os.path.exists(csv_path)

df_out = pd.DataFrame([row])
df_out.to_csv(csv_path, mode="a", header=write_header, index=False)

txt_hdl.close()
