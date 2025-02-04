# from __future__ import annotations

import json
import logging as log
import os
import sys
import typing

import numpy as np
import torch
import torch.backends.cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.dataset

if True:
    import sys
    sys.path.append('..')
    import common

SUMMARY_FN = {
    'mean': lambda x: np.array(x).mean(),
    'stdev': lambda x: np.std(np.array(x)),
    'median': lambda x: np.median(np.array(x)),
    'max': lambda x: np.array(x).max(),
    'min': lambda x: np.array(x).min(),
}

STORE_MODEL_AND_LOG = True


class Runner():
    def __init__(self, device: torch.device, job_id: dict, stdin: dict, config_yaml_abspath: str):
        if None in [device, job_id, stdin, config_yaml_abspath]:
            raise ValueError

        self.my_device = device
        self.job_id = job_id
        self.ctx = stdin

        backend_config = common.util.load_yaml(config_yaml_abspath)

        # d_: directory, f_: file
        self.path = {}

        self.path['d_dataset'] = os.path.join(
            os.path.dirname(config_yaml_abspath),
            backend_config['location']['dataset'],
            self.ctx['datasetName'] + "/")

        if STORE_MODEL_AND_LOG:
            result_dir = os.path.join(
                os.path.dirname(config_yaml_abspath),
                backend_config['location']['result'])

            os.makedirs(result_dir, mode=0o777, exist_ok=True)

            my_dir = os.path.join(result_dir, self.job_id)
            self.path['f_model'] = os.path.join(my_dir, 'model.pt')
            self.path['f_log'] = os.path.join(my_dir, "log.json")

            # create a folder for saving the model
            os.makedirs(my_dir, mode=0o777, exist_ok=True)

        self.dataset, self.loader_train, self.loader_test = self._prepare_dataset_loader(
            self.path['d_dataset'],
            self.ctx['inputLength'],
            self.ctx['outputLength'],
            self.ctx['batchSize'],
            self.ctx['trainTestRatio'],
            self.ctx['additionalTemporalEmbed'],
        )
        self.model = self._prepare_model(
            self.ctx, self.dataset, self.my_device)
        self.loss_fn, self.optim = self._prepare_metric_and_optim(
            self.ctx, self.model)
        self.log = []
        self.scaler = common.scaler.StandardScaler_torch()
        self.scaler.fit(self.dataset.X)

    def _prepare_dataset_loader(self, folder_path: str, x_len: int, y_len: int, batch_size: int, train_test_ratio: float, additional_temporal_embed: str)\
            -> typing.Tuple[common.format.TSFMovingWindow, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        # load raw dataset
        dataset = common.format.TSFMovingWindow(
            folder_path=folder_path, x_len=x_len, y_len=y_len)
        dataset.embed_time(additional_temporal_embed)

        # train-test split by temporal order (e.g. timestamps of the test dataset are newer than those of the train dataset)
        total_len = len(dataset)
        train_len = int(total_len * train_test_ratio)

        dataset_train = torch.utils.data.Subset(dataset, range(train_len))
        dataset_test = torch.utils.data.Subset(
            dataset, range(train_len, total_len))

        # make data loaders
        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2,
            shuffle=True)

        loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2,
            shuffle=True)

        return dataset, loader_train, loader_test

    def _prepare_model(self, ctx: typing.Dict[str, typing.Any], dataset: common.format.TSFMovingWindow, device: torch.device) -> torch.nn.Module:
        model_name = ctx['modelName']

        if 'modelConfig' not in ctx:
            model_conf_path = f'./common/model/{model_name}.yaml'
            if os.path.exists(model_conf_path):
                ctx['modelConfig'] = common.util.load_yaml(model_conf_path)
            else:
                ctx['modelConfig'] = {}

        ctx['modelConfig']['x_len'] = int(ctx['inputLength'])
        ctx['modelConfig']['y_len'] = int(ctx['outputLength'])
        ctx['modelConfig']['x_dim'] = len(ctx['targetSensorAttributes'])
        ctx['modelConfig']['y_dim'] = len(ctx['targetSensorAttributes'])

        my_model = getattr(common.model, model_name)(ctx, dataset, device)
        if my_model is None:
            raise BaseException(f'no such model: {model_name}')
        my_model: common.model.ModelBase
        my_model = my_model.to(device)

        return my_model

    def _prepare_metric_and_optim(self, ctx: typing.Dict[str, typing.Any], model: torch.nn.Module) -> typing.Tuple[torch.nn.Module, torch.optim.Optimizer]:
        loss_fn: torch.nn.Module = common.metric.MaskedRegressionLossCollection()

        assert ctx['optimizer'] in [
            'Adam', 'SGD'], f"unsupported optimizer: {ctx['optimizer']}"

        # optimizer is required only for training
        if ctx['optimizer'] == 'Adam':
            optim = torch.optim.Adam(
                model.parameters(),
                lr=ctx['learningRate'],
                weight_decay=ctx['weightDecay'] if ctx['weightDecay'] else 0)

        elif ctx['optimizer'] == 'SGD':
            optim = torch.optim.SGD(
                model.parameters(),
                lr=ctx['learningRate'],
                weight_decay=ctx['weightDecay'] if ctx['weightDecay'] else 0)

        return loss_fn, optim

    def _loop_train(self) -> dict:
        self.model.train()
        # log.debug(f"{self.model._get_name()}'s model mode = {mode}")

        train_loss = {key: [] for key in self.ctx['lossFunction']}
        loss_obj_temp = {key: None for key in self.ctx['lossFunction']}

        batches_seen = 0

        for _, item in enumerate(self.loader_train):
            x: torch.Tensor = item['x']
            y: torch.Tensor = item['y']
            x_time: torch.Tensor = item['x_time'] if 'x_time' in item else None
            y_time: torch.Tensor = item['y_time'] if 'y_time' in item else None

            # reset gradient
            self.optim.zero_grad()

            # z-score scaling
            x, y = self.scaler.transform(x), self.scaler.transform(y)

            # move input data to GPU
            x_cuda, y_cuda = x.to(self.my_device), y.to(self.my_device)

            x_time_cuda = x_time.to(
                self.my_device) if x_time is not None else None
            y_time_cuda = y_time.to(
                self.my_device) if y_time is not None else None

            # forward propagation
            y_hat = self.model.forward(x_cuda, {
                'y': y_cuda,
                'x_time': x_time_cuda,
                'y_time': y_time_cuda,
                'batches_seen': batches_seen
            })

            # calculate loss
            for key in self.ctx['lossFunction']:
                if key != 'MLE':
                    # reverse scaling before calculating loss
                    # note that MLE suppose that the data is normalized
                    _y_hat = self.scaler.inverse_transform(y_hat)
                    _y_cuda = self.scaler.inverse_transform(y_cuda)

                loss_obj_temp[key] = self.loss_fn(key, _y_hat, _y_cuda)

            # backward propagation
            loss_obj_temp[self.ctx['targetLossFunction']].backward()

            # clip gradient
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)

            # update model parameter
            self.optim.step()

            # append loss result
            for key in self.ctx['lossFunction']:
                train_loss[key].append(loss_obj_temp[key].item())

            # update batches_seen
            batches_seen += 1

        # complete this epoch

        return train_loss

    @torch.no_grad()
    def _loop_test(self) -> dict:
        self.model.eval()

        test_loss = {key: [] for key in self.ctx['lossFunction']}
        loss_obj_temp = {key: None for key in self.ctx['lossFunction']}

        for _, item in enumerate(self.loader_test):
            x: torch.Tensor = item['x']
            y: torch.Tensor = item['y']
            x_time: torch.Tensor = item['x_time'] if 'x_time' in item else None
            y_time: torch.Tensor = item['y_time'] if 'y_time' in item else None

            # z-score scaling
            x, y = self.scaler.transform(x), self.scaler.transform(y)

            # move input data to GPU
            x_cuda, y_cuda = x.to(self.my_device), y.to(self.my_device)

            x_time_cuda = x_time.to(
                self.my_device) if x_time is not None else None
            y_time_cuda = y_time.to(
                self.my_device) if y_time is not None else None

            # forward propagation
            y_hat = self.model.forward(x_cuda, {
                'y': y_cuda,
                'x_time': x_time_cuda,
                'y_time': y_time_cuda,
            })

            # calculate loss
            for key in self.ctx['lossFunction']:
                if key != 'MLE':
                    # reverse scaling before calculating loss
                    # note that MLE suppose that the data is normalized
                    _y_hat = self.scaler.inverse_transform(y_hat)
                    _y_cuda = self.scaler.inverse_transform(y_cuda)
                loss_obj_temp[key] = self.loss_fn(key, _y_hat, _y_cuda)

            # append loss result
            for key in self.ctx['lossFunction']:
                # _loss_fn[key]: torch.Tensor
                test_loss[key].append(loss_obj_temp[key].item())

        return test_loss

    def train(self) -> dict:
        test_loss_target_key = f'test_{self.ctx["targetLossFunction"]}_mean'

        # start from zero
        best_loss = np.inf
        start_epoch = 0

        if STORE_MODEL_AND_LOG:
            # load incompleted job records
            if os.path.exists(self.path['f_log']) and os.path.exists(self.path['f_model']):
                # load previous model
                self.model.load_state_dict(torch.load(
                    self.path['f_model'], map_location=self.my_device))

                # load previous log
                with open(self.path['f_log'], 'r') as f:
                    self.log = json.load(f)

                best_loss = max([epoch_result[test_loss_target_key]
                                for epoch_result in self.log])
                start_epoch = max([epoch_result['epoch']
                                   for epoch_result in self.log]) + 1

        # epoch loop
        for epoch in range(start_epoch, self.ctx['maxEpoch']):
            # training
            with common.util.timer() as time_train:
                train_loss = self._loop_train()

            # testing
            with common.util.timer() as time_test:
                test_loss = self._loop_test()

            # summarize result
            epoch_result = {'epoch': epoch}

            # for train
            for _lf_name in self.ctx['lossFunction']:
                for _sf_name, _sf in SUMMARY_FN.items():
                    # {mode}_{lossFunction}_{summaryFunction}; ex) train_MAE_mean
                    key = f'train_{_lf_name}_{_sf_name}'
                    epoch_result[key] = _sf(train_loss[_lf_name])

            # for test
            for _lf_name in self.ctx['lossFunction']:
                for _sf_name, _sf in SUMMARY_FN.items():
                    key = f'test_{_lf_name}_{_sf_name}'
                    epoch_result[key] = _sf(test_loss[_lf_name])

            epoch_result['time_train'] = time_train()
            epoch_result['time_test'] = time_test()
            epoch_result['cuda_mem_usage'] = torch.cuda.max_memory_allocated(
                device=self.my_device)

            self.log.append(epoch_result)

            if STORE_MODEL_AND_LOG:
                test_loss = epoch_result[test_loss_target_key]
                if (best_loss > test_loss):
                    # save model only having the best loss
                    best_loss = test_loss
                    torch.save(self.model.state_dict(), self.path['f_model'])

                # save each epoch summary to file
                with open(self.path['f_log'], 'w') as f:
                    json.dump(self.log, f)

        return self.log


def main():
    # set torch options
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_printoptions(profile="full", precision=3, sci_mode=False)

    log.basicConfig(
        format="[%(levelname)s][%(asctime)s.%(msecs)03d][%(filename)s:%(lineno)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log.INFO)

    # select device
    if torch.cuda.is_available():
        my_device = torch.device('cuda:0')
    else:
        my_device = torch.device('cpu')

    # check arguments
    if not (len(sys.argv) == 2):
        # should be ./main.py <job_id>
        raise ValueError(f'not enough arguments, {sys.argv=}')

    job_id = sys.argv[1]
    if not job_id.isdigit():
        raise ValueError(
            f'arguments is wrong. second argument should be a number format, {sys.argv=}')

    # stdin
    stdin = json.loads(input())

    # create Runner
    r = Runner(
        device=my_device,
        job_id=job_id,
        stdin=stdin,
        config_yaml_abspath=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            './config.yaml'))

    # start training
    result = r.train()
    print(json.dumps(result))

    # clear the GPU
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
