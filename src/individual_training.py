import json

import pandas as pd

from src.utils.functions import inverse_transform_test, make_plot, get_model, mkdir_if_not_exists
from src.base.trainers import Trainers
from src.models.model_serializer import ModelSerializer
from src.dataset.processing import Processsing
from src.data import TimeSeriesLoader
from src.utils.logger import log
from logging import INFO



class IndividualTraining(Trainers):
    def __init__(self, args):
        super(IndividualTraining, self).__init__(args=args)
        self.seed_all(args.seed)

        self.processing = Processsing(args=self.args, data_path=self.args.data_path)

        X_train, X_val, y_train, y_val, self.x_scaler, self.y_scaler = self.processing.make_preprocessing(
            filter_bs=self.args.filter_bs, per_area=False
        )

        self.X_train, self.X_val, self.y_train, self.y_val, self.area_X_train, self.area_X_val, self.area_y_train, self.area_y_val = (
            self.processing.make_postprocessing(X_train, X_val, y_train, y_val))

        self.input_dim = self.processing.get_input_dims(self.X_train)

        # print(self.input_dim)

        self.model = get_model(args=self.args, model=args.model_name,
                          input_dim=self.input_dim,
                          out_dim=self.y_train.shape[1],
                          lags=args.num_lags)



    def fit(self, idxs=[0], log_per=1):

        num_features = len(self.X_train[0][0])

        train_loader = TimeSeriesLoader(X=self.X_train,
                                       y=self.y_train,
                                       num_lags=self.args.num_lags,
                                       num_features=num_features,
                                       indices=idxs, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers).get_dataloader()

        val_loader = TimeSeriesLoader(X=self.X_val,
                                       y=self.y_val,
                                       num_lags=self.args.num_lags,
                                       num_features=num_features,
                                       indices=idxs, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers).get_dataloader()

        self.model, train_mse_curve, val_mse_curve = self.train(model=self.model,
                      train_loader=train_loader, test_loader=val_loader,
                      epochs=self.args.epochs,
                      optimizer=self.args.optimizer, lr=self.args.lr,
                      criterion=self.args.criterion,
                      early_stopping=self.args.early_stopping,
                      patience=self.args.patience,
                      device=self.args.device, log_per=log_per, cid=self.args.filter_bs)

        mkdir_if_not_exists(f'etc/in/ckpts/{self.args.model_name}/best/')
        client_model_serializer = ModelSerializer(model=self.model, path=f'etc/in/ckpts/{self.args.model_name}/best/')
        client_model_serializer.save(f'{self.args.model_name}_{self.args.filter_bs}.h5')
        log(INFO, f"Client model saved on etc/in/ckpts/{self.args.model_name}/best/{self.args.model_name}_{self.args.filter_bs}.h5")

        return self.model, train_mse_curve, val_mse_curve

    def predict(self, model, cid, X_test, y_test, client_scaler, exogenous_data_test, plot, apply_round=False, idxs=[0], round_dimensions=[0], invert_scale=False):

        num_features = len(X_test[0][0])

        test_loader = TimeSeriesLoader(X=X_test, y=y_test,
                         num_lags=self.args.num_lags,
                         num_features=num_features, indices=idxs, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False).get_dataloader()

        test_mse, test_rmse, test_mae, test_mape, test_r2, test_nrmse, pinball, _, y_pred_test = Trainers.test(model, test_loader, None, device=self.args.device)

        inverted_y_test, inverted_y_pred_test = inverse_transform_test(
            y_test, y_pred_test, client_scaler, round_preds=apply_round, dims=round_dimensions
        )

        inverted_test_mse, inverted_test_rmse, inverted_test_mae, inverted_test_mape, inverted_test_r2, inverted_test_nrmse, inverted_test_pinball, inverted_test_res_per_dim = Trainers.accumulate_metrics(inverted_y_test, inverted_y_pred_test, log_per_output=True, return_all=True
        )
        for i in range(y_pred_test.shape[1]):
            inverted_client_preds = {"y_true": inverted_y_test[:, i].tolist(), "y_pred": inverted_y_pred_test[:, i].tolist(), "client": cid}
            mkdir_if_not_exists(f'etc/in/results/{self.args.model_name}/preds/inverted')
            with open(f'etc/in/results/{self.args.model_name}/preds/inverted/{cid}_preds.json', 'w') as file:
                json.dump(inverted_client_preds, file)

        print(f"Final Prediction in {cid}")
        print(f"[Test]: mse: {test_mse}, rmse: {test_rmse}, mae {test_mae}, "
              f"r2: {test_r2}, nrmse: {test_nrmse}\n\n")


        for i in range(y_pred_test.shape[1]):
            client_preds = {"y_true": y_test[:, i].tolist(), "y_pred": y_pred_test[:, i].tolist(), "client": cid}
            with open(f'etc/in/results/{self.args.model_name}/preds/{cid}_preds.json', 'w') as file:
                json.dump(client_preds, file)

        results = pd.DataFrame([{'mse': test_mse, 'rmse': test_rmse, 'mae': test_mae, 'mape': test_mape, 'r2': test_r2,
                                 'nrmse': test_nrmse, 'pinball': pinball, 'client': cid}])
        results.to_json(f"etc/in/results/{self.args.model_name}/{cid}_results.json", index=False, orient="records")
        inverted_values = pd.DataFrame([
            {'mse': inverted_test_mse, 'rmse': inverted_test_rmse, 'mae': inverted_test_mae, 'mape': inverted_test_mape, 'r2': inverted_test_r2,
             'nrmse': inverted_test_nrmse, 'pinball': inverted_test_pinball, 'client': cid}])

        inverted_values.to_json(f"etc/in/results/{self.args.model_name}/{cid}_results_inverted.json", index=False, orient='records')

        return results, inverted_values


    def evaluate(self):
        df_dict = self.processing.get_test_data()
        df = df_dict[str(self.args.filter_bs)].copy()
        test_data = self.processing.handle_nans(train_data=df, constant=self.args.nan_constant,
                                                identifier=self.args.identifier)
        X_test, y_test = self.processing.to_Xy(test_data, targets=self.args.targets)
        X_test = self.processing.scale_features(X_test, scaler=self.x_scaler, per_area=False)
        y_test = self.processing.scale_features(y_test, scaler=self.y_scaler, per_area=False)

        # generate time lags
        X_test = self.processing.generate_time_lags(X_test, self.args.num_lags)
        y_test = self.processing.generate_time_lags(y_test, self.args.num_lags, is_y=True)

        # remove identifiers
        X_test, y_test = self.processing.remove_identifiers(X_test, y_test)

        num_features = len(X_test.columns) // self.args.num_lags

        # to timeseries representation
        X_test = self.processing.to_timeseries_rep(X_test.to_numpy(), self.args.num_lags, num_features=num_features)

        y_test = y_test.to_numpy()

        self.predict(model=self.model, cid=self.args.filter_bs, X_test=X_test, y_test=y_test,
                     exogenous_data_test=None, plot=True, client_scaler=self.y_scaler,
                     invert_scale=self.args.invert_scale)


