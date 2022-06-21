import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from data import load_data, preprocess, create_tf_dataset, compute_adjacency_matrix
from model import GraphInfo, get_model
from matplotlib import pyplot as plt
import numpy as np



def plot_speed_array(speeds_array):
    plt.figure(figsize=(18, 6))
    plt.plot(speeds_array[:, [0, -1]])
    plt.legend(["route_0", "route_25"])
    plt.show()


def main(url):
    route_distances_array, speeds_array = load_data(url)

    # plot_speed_array(data[1])

    # Data preprocessing
    train_size, val_size = 0.5, 0.2
    train_array, val_array, test_array = preprocess(speeds_array, train_size, val_size)
    
    print(f"train set size: {train_array.shape}")
    print(f"validation set size: {val_array.shape}")
    print(f"test set size: {test_array.shape}")


    # Creating TensorFlow Datasets
    batch_size = 64
    input_sequence_length = 12
    forecast_horizon = 3
    multi_horizon = False

    
    train_dataset, val_dataset = (
        create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
        for data_array in [train_array, val_array]
    )

    test_dataset = create_tf_dataset(
        test_array,
        input_sequence_length,
        forecast_horizon,
        batch_size=test_array.shape[0],
        shuffle=False,
        multi_horizon=multi_horizon,
    )

    sigma2 = 0.1
    epsilon = 0.5
    adjacency_matrix = compute_adjacency_matrix(route_distances_array, sigma2, epsilon)
    node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
    graph = GraphInfo(
        edges=(node_indices.tolist(), neighbor_indices.tolist()),
        num_nodes=adjacency_matrix.shape[0],
    )
    print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")

    epochs = 20

    model = get_model(graph)

    tf.keras.utils.plot_model(model, 'LSTM_GC.jpg', dpi=600)
    # model.compile(
    #     optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0002),
    #     loss=tf.keras.losses.MeanSquaredError(),
    # )

    # model.fit(
    #     train_dataset,
    #     validation_data=val_dataset,
    #     epochs=epochs,
    #     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
    # )


if __name__ == '__main__':
    url = "https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/data_loader/PeMS-M.zip"
    main(url)