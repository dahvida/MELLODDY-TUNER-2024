import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from scipy import sparse
import json

parser = argparse.ArgumentParser(description="Arguments for running descriptor normalization before starting a MELLODDY training run")

parser.add_argument("--path", type=str, 
                    help="Path to folder where output of prepare_4_training was stored, or to json config file")

parser.add_argument("--fill_na", action="store_true", 
                    help="Whether to fill NaN in X matrix with value defined with --nan_value")

parser.add_argument("--nan_value", type=float,
                    help="Value to use when filling NaN",
                    default=0.0)

parser.add_argument("--neginf_value", type=float,
                    help="Minimum value allowed for the dataset (used to catch neginf or similar outliers)",
                    default=1e-8)

parser.add_argument("--posinf_value", type=float,
                    help="Maximum value allowed for the dataset (used to catch posinf or similar outliers)",
                    default=1e8)

parser.add_argument("--standard_scale", action="store_true", 
                    help="Whether to use a standardscaler on the dataset, fitting on folds [1,2,3]")

parser.add_argument("--minmax_scale", action="store_true", 
                    help="Whether to use a minmaxscaler on the dataset, fitting on folds [1,2,3]")

parser.add_argument("--quantile_scale", action="store_true", 
                    help="Whether to use a quantilescaler on the dataset, fitting on folds [1,2,3]")

parser.add_argument("--variance_threshold", type=float,
                    help="Variance threshold for feature selection. If set to 0.0, then no feature selection will be performed.",
                    default=0.0)

args = parser.parse_args()

def get_subpaths(path: str):
    """Utility function to get paths to relevant matrices, given a root  folder
    """
    path_dict = {}
    root = f"{path}/matrices/wo_aux"
    
    prefixes = ["cls", "reg"]

    for prefix in prefixes:
        path_dict[prefix] = [f"{root}/{prefix}/{prefix}_T11_x.npz", f"{root}/{prefix}/{prefix}_T11_fold_vector.npy", f"{root}/{prefix}/{prefix}_T10_y.npz"]
    
    return path_dict

def load_npz_matrix(path: str):
    """Utility function to load a npz matrix and convert it to numpy
    """
    x = sparse.load_npz(path)
    return x.toarray()

def main(args):
    """Script to preprocess molecular descriptors before running a MELLODDY training run.

    Steps:
        1. Get relevant paths
        2. Load matrices
        3. Deal with NaNs with either:
            a. remove rows with at least one NaN
            b. fill according to args.nan_value
        4. Bound dataset as args.neginf_value <= x <= args.posinf_value
        5. Scale values with either:
            a. standardscaler
            b. quantilescaler
            c. minmaxscaler
        6. Save output in original folder, using same names with "_processed" appended 
    """
    
    # check if path corresponds to json, if yes load flags from there
    if "json" in args.path:
        with open(args.path, 'r') as f:
            args = json.load(f)
        args = argparse.Namespace(**args)
        print("Loading flags from json specified as --path")

    # get paths
    path_dict = get_subpaths(args.path)

    # loop over dataset types (cls, reg, hyb)
    for key in list(path_dict.keys()):
        print(f"Processing {key} folder...")
        path_list = path_dict[key]

        # load matrices to process
        print("Loading original matrices...")
        x = load_npz_matrix(path_list[0])
        fold = np.load(path_list[1])
        y = load_npz_matrix(path_list[2])

        # deal with NaNs and outliers
        print("Starting cleanup...")
        if args.fill_na is True:
            x = np.nan_to_num(x,
                              nan = args.nan_value,
                              posinf = args.posinf_value,
                              neginf = args.neginf_value)
            print(f"NaN replaced with {args.nan}, values bound between {args.neginf_value} and {args.posinf_value}")
        else:
            nan_idx = np.where(np.isnan(x))[0]
            x = x[~nan_idx]
            y = y[~nan_idx]
            fold = fold[~nan_idx]
            x[x > args.posinf_value] = args.posinf_value
            x[x < args.neginf_value] = args.neginf_value
            print(f"Compounds with NaN removed, values bound between {args.neginf_value} and {args.posinf_value}")
            print(f"{len(nan_idx)} compounds were removed in the process")

        # run feature selection
        print(f"Running feature selection with threshold {args.variance_threshold}...")
        start_n = x.shape[1]
        train_idx = np.where(np.isin(x, [1, 2, 3]))[0]
        x_train = x[train_idx]
        selector = VarianceThreshold(threshold=args.variance_threshold)
        selector.fit(x_train)
        x = selector.transform(x)
        x_train = selector.transform(x_train)
        end_n = x.shape[1]
        print(f"Feature selection finished, {start_n - end_n} columns removed")

        # run standardization
        if args.standard_scale is True:
            scaler = StandardScaler()
            print("Data will be scaled with the StandardScaler")
        elif args.minmax_scale is True:
            scaler = MinMaxScaler()
            print("Data will be scaled with the MinMaxScaler")
        elif args.quantile_scale is True:
            scaler = QuantileTransformer()
            print("Data will be scaled with the QuantileTransformer")
        scaler.fit(x_train)
        x = scaler.transform(x)
        print("Dataset scaled using folds [1,2,3] as training set")
        
        # convert numpy back to sparse
        x = sparse.csr_matrix(x)
        y = sparse.csr_matrix(y)

        # save output in same path, with "_processed" naming convention
        sparse.save_npz(f'{path_list[0][:-4]}_processed.npz', x)
        print(f"Processed X matrix saved at {path_list[0][:-4]}_processed.npz")
        np.save(f'{path_list[1][:-4]}_processed.npy', fold)
        print(f"Processed X matrix saved at {path_list[1][:-4]}_processed.npy")
        sparse.save_npz(f'{path_list[2][:-4]}_processed.npz', y)
        print(f"Processed X matrix saved at {path_list[2][:-4]}_processed.npz")

if __name__ == "__main__":
    main(args)