
import pickle
from tqdm import tqdm
from logging import INFO
from pymongo import MongoClient
from argparse import ArgumentParser
from src.dataset.processing import Processing
from src.utils.logger import log
from bson.binary import Binary



def main():
    def save_batch(X, y, _type: str, _id: str):
        documents = []
        data_col.delete_many({"client_id": _id, "type": _type})
        for i in tqdm(range(len(X)), total=len(X)):
            _doc = {
                "client_id": args.filter_bs,
                "type": _type,
                "X": Binary(pickle.dumps(X[i], protocol=4)),
                "y": Binary(pickle.dumps(y[i], protocol=4))
            }
            documents.append(_doc)
            if len(documents) >= 1000:
                data_col.insert_many(documents)
                documents = []
        if documents:
            data_col.insert_many(documents)

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
    log(INFO, f"Migrating {args.filter_bs} data to MongoDB")
    print(args)
    processing = Processing(args=args, data_path=args.data_path)
    X_train, X_val, y_train, y_val, x_scaler, y_scaler = processing.make_preprocessing(filter_bs=args.filter_bs, per_area=False, peek=False)
    X_test, y_test, _ = processing.make_test_processing(filter_data=args.filter_bs,
                                                                                  x_scaler=x_scaler,
                                                                                  y_scaler=y_scaler)

    input_dim, output_dim = processing.get_input_dims(X_train), y_train.shape[1]

    client = MongoClient(args.mongo_uri)
    db = client["pecanstreet"]
    data_col = db[f"{args.loc}-samples"]
    meta_col = db[f"{args.loc}-metadata"]

    meta_obj = {
        "client_id": args.filter_bs,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "x_scaler": Binary(pickle.dumps(x_scaler, protocol=4)),
        "y_scaler": Binary(pickle.dumps(y_scaler, protocol=4)),
        "num_train_samples": len(X_train),
        "num_val_samples": len(X_val)
    }

    meta_col.replace_one({"client_id": args.filter_bs}, meta_obj, upsert=True)

    save_batch(X_train, y_train, "train", args.filter_bs)
    save_batch(X_val, y_val, "val", args.filter_bs)
    save_batch(X_test, y_test, "test", args.filter_bs)
    log(INFO, f"Migration for {args.filter_bs}'s data finished")


if "__main__" == __name__:
    main()