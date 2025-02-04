# 👢 SPATS

![](./img/overview.png)

This is the official implementation of "**SPA**tio-**T**emporal graph **S**ystem (SPATS)" in the following paper:

- Yoon, Heeyong, et al. "SPATS: A practical system for comparative analysis of Spatio-Temporal Graph Neural Networks." _Journal Name Here_ (2025): _Issue and Volume Here._

## 📝 Citation (Not yet published)

If you use this repository in your research project, please cite the following BiBTeX in your paper:

```bibtex
(TBD)
```

## 📜 How to use

### ☑️ Prerequisites

> ⚠ This is not a commercial system, so we do not guarantee its behavior in all environments. Please handle exceptional cases with appropriate external knowledge, official documentation, and experience.

> ⚠ Some parts contain template text; do not copy and paste them directly, and make sure to replace the template text appropriately.

> ⚠ Because NFS recognizes users only by ID, it is strongly recommended to set all nodes' user and group IDs to the same value. Otherwise, unexpected behavior can occur (search about the Linux command `id`, `usermod` or `groupmod`).

1. Prepare your GPU cluster and set one node as **_master server_** and other nodes as **_worker server_**. The **_master server_** and **_worker server_** roles can be overlapped, but separating the roles is recommended because CPU usage might affect model training.

   Our testing environment for each node:

   - Ubuntu 18.04

   - CUDA 11.3 to 12.1 (different per node)

   - Python 3.10 (installed by miniconda) and necessary packages (PyTorch, Numpy, ...)

2. Install `NFS` server on the **_master server_**

   a. Make the repository visible to anyone.

   ```bash
      sudo apt install nfs-kernel-server -y # install NFS server
   ```

   b. Modify `/etc/exports`.

   ```bash
      sudo vim /etc/exports
   ```

   c. Write the following text line at the bottom of the `/etc/exports/`.

   ```vim
   /your_master_server_folder/SPATS/ *(rw,sync,no_subtree_check)
   ```

   d. Restart `NFS`.

   ```bash
      sudo exportfs -a
      sudo systemctl restart nfs-kernel-server
   ```

   e. Download this repository on the **_master server_**.

   ```bash
      cd /your_master_server_folder/
      git clone https://github.com/sunrise2575/SPATS
   ```

   f. Make the repository visible to anyone.

   ```bash
      sudo chmod 777 ./SPATS/ # Make the repository visible to anyone
   ```

3. Install `NFS` client and mount the remote folder on each **_worker server_**

   a. Install `NFS` client.

   ```bash
      sudo apt install nfs-common -y
   ```

   b. Mount the **_master server_**'s folder.

   ```bash
      mkdir -p /your_worker_server_folder/SPATS/
      mount <master_server_IP>:/your_master_server_folder/SPATS /your_worker_server_folder/SPATS/
   ```

   For checking IP, use `ip a` or `ifconfig -a` command.

   c. Make sure the remote folder is mounted well.

   ```bash
      ls -al /your_worker_server_folder/SPATS/
   ```

4. Install `Python` dependencies on every server (all **_master server_** and **_worker server_**)

   ```bash
      conda activate base # or specify different virtualenv
      pip install -r requirements.txt
   ```

   Some Python packages in `requirements.txt` are not version-sensitive, so you can selectively modify `requirements.txt` not to upgrade or downgrade your existing packages.

### 📖 Using SPATS

1. Launch Broker and Worker

   a. Launch Broker process on **_master server_**

   ```bash
      python ./broker.py --port <broker_port>
   ```

   The default port number is `9999`; if you do not specify the port number, you can type like this:

   ```bash
      python ./broker.py
   ```

   b. Launch Worker process on each **_worker server_**

   ```bash
      python ./worker.py --broker <broker_IP>:<broker_port> --gpu <GPU_indices_as_you_wish>
      # python ./worker.py --broker 1.2.3.4:9999 --gpu 0,2,5 # example; using 0,2,5-th GPU only
   ```

   We recommend using `tmux` to manage processes on each node efficiently.

2. Insert jobs to Broker

   > ⚠ The following content may be lengthy and complex. Since this was developed for research purposes and aims to execute queries in Python without using a structured language like SQL, the details can be intricate. It is recommended to read through carefully and learn by using `query-bulk-insert.py` yourself.

   a. Find the following text lines in `query.py` and `query-bulk-insert.py`

   ```python
      BROKER_IP='127.0.0.1'
      BROKER_PORT='9999'
   ```

   and replace it with the IP and port number of your Broker.

   ```python
      BROKER_IP=<broker_IP>
      BROKER_PORT=<broker_port>
   ```

   b. Insert multiple jobs using `query-bulk-insert.py`

   In `query-bulk-insert.py`, the variable `DEFAULT` is a default job setting. By changing some values of `DEFAULT` and `VARIATION`, `query-bulk-insert.py` generates thousands of combinations of jobs.

   The following example shows how to use `VARIATION` properly.

   ```python
      VARIATION = {
          'maxEpoch': [10],
          'batchSize': [64, 128], # 2 variations
          'datasetName': ['METR-LA', 'PEMS-BAY'], # 2 variations
          'modelName': ['Seq2Seq_RNN', 'ASTGCN'], # 2 variations
          'adjacencyMatrixThresholdValue': [float('0.' + str(i)) for i in range(2)], # 2 variations
          # -> 2x2x2x2=16 variations
      }
   ```

   In this case, SPATS sets `maxEpoch` to `10`, while `batchSize` to `64` and `128` both. The candidate `datasetName` and `modelName` work similarly. You can write Python generator syntax on `VARIATION` such as `adjacencyMatrixThresholdValue` of the example.

   On the other hand, `DATASET_DEPENDENT_SETTING` and `MODEL_DEPENDENT_SETTING` in `query-bulk-insert.py` are variables to help change other values in `DEFAULT` comfortably, which are affected by `datasetName` and `modelName` in `VARIATION`, respectively. For example, the above `VARIATION` example specifies that candidate datasets are `METR-LA` and `PEMS-BAY`, following items in `DATASET_DEPENDENT_SETTING` are selected,

   ```python
      DATASET_DEPENDENT_SETTING={
        ...
          'METR-LA': {
              'additionalTemporalEmbed': ['timestamp_in_day'], # 1 variation
              'targetSensorAttributes': [['speed']], # 1 variation
              'inputLength': [12], 'outputLength': [12], # 1 variation each
              # -> 1x1x1x1=1 variation
        },
          'PEMS-BAY': {
              'additionalTemporalEmbed': ['timestamp_in_day'], # 1 variation
              'targetSensorAttributes': [['speed']], # 1 variation
              'inputLength': [12], 'outputLength': [12], # 1 variation each
              # -> 1x1x1x1=1 variation
        },
        ...
       }
   ```

   Similarly, the following items in `MODEL_DEPENDENT_SETTING` are selected because the `modelName` is `['Seq2Seq_RNN', 'ASTGCN']`.

   ```python
      MODEL_DEPENDENT_SETTING={
        ...
          'Seq2Seq_RNN': { 'adjacencyMatrixLaplacianMatrix': [None] }, # 1 variation
          'ASTGCN': { 'adjacencyMatrixLaplacianMatrix': ['cheb_poly'], }, # 1 variation
        ...
      }
   ```

   If the default value is like this:

   ```python
      DEFAULT={
          'trainTestRatio': 0.7,
          'adjacencyMatrixThresholdValue': 0.7,

          'maxEpoch': 100,
          'batchSize': 64,

          'lossFunction': ["MAE", "MSE", "MAAPE"], # this should be list; 1 variation
          'targetLossFunction': "MSE",

          # about optimizer
          'optimizer': 'Adam',
          'learningRate': 0.001,
          'weightDecay': 0.0005,
      }
   ```

   By the power of algorithm `stdin_generator()` and additional model configuration, which is stored in `.yaml` files, 16 variations of jobs are generated, such as

   ```python
      stdin = {
          'datasetName': 'METR-LA', # set by VARIATION
          'adjacencyMatrixThresholdValue': 0.0, # changed by VARIATION
          'modelName': 'Seq2Seq_RNN', # set by VARIATION
          'maxEpoch': 10, # changed by VARIATION
          'batchSize': 64, # changed by VARIATION

          'adjacencyMatrixLaplacianMatrix': None, # set by MODEL_DEPENDENT_SETTING
          'modelConfig': {}, # automatically filled by common/model/<MODEL_NAME>.yaml. if it doesn't exist, it becomes an empty dict.

          'additionalTemporalEmbeds': ['timestamp_in_day'], # set by DATASET_DEPENDENT_SETTING
          'inputLength': 12, # set by DATASET_DEPENDENT_SETTING
          'outputLength': 12, # set by DATASET_DEPENDENT_SETTING
          'targetSensorAttributes': ['speed'],  # set by DATASET_DEPENDENT_SETTING

          'trainTestRatio': 0.7, # from DEFAULT
          'learningRate': 0.001, # from DEFAULT
          'weightDecay': 0.0005, # from DEFAULT
      }
   ```

   or

   ```python
      stdin = {
          'datasetName': 'PEMS-BAY', # set by VARIATION
          'adjacencyMatrixThresholdValue': 0.1, # changed by VARIATION
          'modelName': 'ASTGCN', # set by VARIATION
          'maxEpoch': 10, # changed by VARIATION
          'batchSize': 128, # changed by VARIATION

          'adjacencyMatrixLaplacianMatrix': 'cheb_poly', # set by MODEL_DEPENDENT_SETTING
          'modelConfig': {
              'time_strides': 1,
              'nb_block': 2
              'nb_chev_filter': 64,
              'nb_time_filter': 64
        }, # automatically filled by common/model/<MODEL_NAME>.yaml. if it doesn't exist, it becomes an empty dict.

          'additionalTemporalEmbeds': ['timestamp_in_day'], # set by DATASET_DEPENDENT_SETTING
          'inputLength': 12, # set by DATASET_DEPENDENT_SETTING
          'outputLength': 12, # set by DATASET_DEPENDENT_SETTING
          'targetSensorAttributes': ['speed'],  # set by DATASET_DEPENDENT_SETTING

          'maxEpoch': 10, # changed by VARIATION
          'batchSize': 128, # changed by VARIATION

          'trainTestRatio': 0.7, # from DEFAULT
          'learningRate': 0.001, # from DEFAULT
          'weightDecay': 0.0005, # from DEFAULT
      }
   ```

   and so on. you can add `print(stdin)` at the loop of `main()` in `query-bulk-insert.py` to print generated stdin for better understanding and debugging

   c. After `DEFAULT` and `VARIATION` are ready, you can insert jobs and wait for the completion.

   ```bash
      python ./query-bulk-insert.py # run SPATS!
   ```

   Broker works as a queue, so you can insert more jobs without waiting for the completion of previously inserted jobs.

3. Get the job information and delete job

   a. Full job info. (only shows recently inserted 100 jobs)

   ```bash
      python ./query.py list
   ```

   b. Job info with filtered type (only shows recently inserted 100 jobs).

   ```bash
      python ./query.py list <started|pending|success|failure>
   ```

   c. Single job info.

   ```bash
      python ./query.py select <job_id>
   ```

   d. Delete a job.

   ```bash
      python ./query.py delete <job_id>
   ```

4. Visualize results

   a. Copy `broker.sqlite3` related files to `visualize/` folder

   ```bash
   cp broker.sqlite3* visualize/.
   ```

   b. Run extractor

   ```bash
   cd visualize/
   python ./extractor.py broker.sqlite3 success
   ```

   c. Run `1-replace-model-and-dataset.ipynb` and `2-concat-results.ipynb` in order to get `result.csv`

   d. Use `make-fig*.ipynb` files to generate comparison results similar to our paper.
