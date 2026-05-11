import pickle
import numpy as np
from logging import INFO
from argparse import ArgumentParser
from src.dataset.processing import Processing
from src.utils.logger import log
from bson.binary import Binary



def main():
    def save_batch(X, y, _type: str, _id: str, loc: str):
        if _type in ["train", "val"]:
            path = f"dataset/pecanstreet/15min/{loc}/train/"
            np.save(f"{path}/{_id}-{_type}-X.npy", X)
            np.save(f"{path}/{_id}-{_type}-y.npy", y)
        elif _type == "test":
            path = f"dataset/pecanstreet/15min/{loc}/test/"
            np.save(f"{path}/{_id}-{_type}-X.npy", X)
            np.save(f"{path}/{_id}-{_type}-y.npy", y)
        log(INFO, f"Saved {path} - X Shape: {X.shape} - y Shape: {y.shape}")

    parser = ArgumentParser()
    parser.add_argument("--mongo_uri", type=str, default='mongodb://localhost:27017')
    parser.add_argument("--model_name", type=str, default='lstm')
    parser.add_argument("--loc", type=str, default='austin')
    parser.add_argument("--data_path", type=str, default='dataset/pecanstreet/15min/austin/train/')
    parser.add_argument("--test_path", type=str, default='dataset/pecanstreet/15min/austin/test/')
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--targets", type=list, default=['consumption'])
    parser.add_argument("--num_lags", type=int, default=96)
    parser.add_argument("--filter_bs", type=str, default=661)
    parser.add_argument("--identifier", type=str, default='cid')
    parser.add_argument("--nan_constant", type=int, default=0)
    parser.add_argument("--x_scaler", type=str, default='minmax')
    parser.add_argument("--y_scaler", type=str, default='minmax')
    parser.add_argument("--outlier_detection", type=any, default=None)
    
    args = parser.parse_args()
    log(INFO, f"Migrating {args.filter_bs} data to Numpy")
    print(args)
    processing = Processing(args=args, data_path=args.data_path)
    X_train, X_val, y_train, y_val, x_scaler, y_scaler = processing.make_preprocessing(filter_bs=args.filter_bs, per_area=False, peek=False)
    X_test, y_test, _ = processing.make_test_processing(filter_data=args.filter_bs,
                                                                                  x_scaler=x_scaler,
                                                                                  y_scaler=y_scaler)

    input_dim, output_dim = processing.get_input_dims(X_train), y_train.shape[1]

    meta_obj = {
        "client_id": args.filter_bs,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "x_scaler": Binary(pickle.dumps(x_scaler, protocol=4)),
        "y_scaler": Binary(pickle.dumps(y_scaler, protocol=4)),
        "num_train_samples": len(X_train),
        "num_val_samples": len(X_val)
    }

    with open(f"dataset/pecanstreet/15min/{args.loc}/train/{args.filter_bs}_metadata.pkl", "wb") as f:
        pickle.dump(meta_obj, f)

    save_batch(X_train, y_train, "train", args.filter_bs, loc=args.loc)
    save_batch(X_val, y_val, "val", args.filter_bs, loc=args.loc)
    save_batch(X_test, y_test, "test", args.filter_bs, loc=args.loc)
    log(INFO, f"Migration for {args.filter_bs}'s data finished")


if "__main__" == __name__:
    main()