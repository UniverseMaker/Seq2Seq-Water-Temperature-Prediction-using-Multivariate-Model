import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import tensorflow as tf
from tensorflow.keras.utils import plot_model
#import keras
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from general_v2 import (data_preprocess, data_preprocess_v2, data_preprocess_v3, data_sequencing, confirm_result,
                        data_visual, model_loss_visual, act_pred_visual, time_preprocess, plot_parity, prep, prep3)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

f_path = "C:/windows/Fonts/malgun.ttf"
fm.FontProperties(fname=f_path).get_name()
plt.rc('font', family='Malgun Gothic')

score_df_all = pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'Encoder Input', 'Output', 'Decoder Input'])

def Seq2Seq_model(epoch, batch, X_train, Y_train, X_train_di, Y_train_di, X_test, feature_cnt_input, feature_cnt_output):
    layers = [35, 35]
    learning_rate = 0.01
    decay = 0
    optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay)
    regulariser = None
    loss = "mse"

    encoder_inputs = keras.layers.Input(shape=(None, feature_cnt_input))

    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    encoder = keras.layers.RNN(encoder_cells, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]

    decoder_inputs = keras.layers.Input(shape=(None, feature_cnt_output))

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = decoder_outputs_and_states[0]

    decoder_dense = keras.layers.Dense(feature_cnt_output,
                                        #units=predict_periods,
                                        activation='linear',
                                        kernel_regularizer=regulariser,
                                        bias_regularizer=regulariser)

    decoder_outputs = decoder_dense(decoder_outputs)

    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=optimiser, loss=loss)
    model.summary()

    train_decoder_inputs_zeros = Y_train.copy()
    train_decoder_inputs_zeros.fill(0.0)

    #hist = model.fit([X_train, train_decoder_inputs_zeros], Y_train, epochs=epoch, batch_size=batch, validation_split=0.2, verbose=1)
    hist = model.fit([X_train, Y_train_di], Y_train, epochs=epoch, batch_size=batch, validation_split=0.2, verbose=1)

    return encoder, decoder, encoder_inputs, encoder_states, decoder_inputs, decoder_outputs, decoder_dense, layers, model, hist


def Seq2Seq_run(args, X_train, Y_train, X_train_di, Y_train_di, X_test, Y_test, X_test_di, Y_test_di, ts_test, feature_sc):
    encoder, decoder, encoder_inputs, encoder_states, decoder_inputs, decoder_outputs, decoder_dense, layers, model, hist = Seq2Seq_model(args['epoch'], args['batch'], X_train, Y_train, X_train_di, Y_train_di, X_test, args['feature_cnt_input'], args['feature_cnt_output'])

    plot_model(model, to_file=os.path.join(args['save_path'], f"{args['model']}_plot.png"), show_shapes=True, show_layer_names=True)

    test_decoder_inputs_zeros = Y_test.copy()
    test_decoder_inputs_zeros.fill(0.0)
    y_test_predicted = model.predict([X_test, test_decoder_inputs_zeros])
    #y_test_predicted = model.predict([X_test, Y_test_di])
    y_test_predicted = y_test_predicted.reshape(y_test_predicted.shape[0], y_test_predicted.shape[1])

    # create empty table with feature_cnt fields
    gru_pred = np.zeros(shape=(y_test_predicted.shape[0], args['feature_cnt_input'] + 1))
    y_test = np.zeros(shape=(Y_test.shape[0], args['feature_cnt_input'] + 1))

    gru_pred[:, -1] = y_test_predicted[:, 0]
    y_test[:, -1] = Y_test[:, 0]

    for i in range(args['feature_cnt_di']):
        gru_pred = np.c_[gru_pred, np.zeros(gru_pred.shape[0])]
        y_test = np.c_[y_test, np.zeros(y_test.shape[0])]

    y_pred = feature_sc.inverse_transform(gru_pred)[:, args['feature_cnt_input']]
    y_test = feature_sc.inverse_transform(y_test)[:, args['feature_cnt_input']]

    predict_result = pd.DataFrame()
    predict_result['date'] = pd.to_datetime(ts_test[-len(y_test):].index.values)
    predict_result['actual'] = y_test
    predict_result['predict'] = y_pred
    predict_result.set_index('date', inplace=True)

    score = confirm_result(y_test, y_pred)

    return encoder, decoder, encoder_inputs, encoder_states, decoder_inputs, decoder_outputs, decoder_dense, layers, predict_result, model, hist, score


def train_test(file_name, args, allset_df, data_df):
    fname = file_name.split(".")[0]
    initial_time = time.time()
    score_df = pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'Encoder Input', 'Output', 'Decoder Input'])

    tf.random.set_seed(13)
    np.random.seed(13)
    random.seed(13)

    with tf.device("/gpu:0"):
        case_df = data_df.copy()
        train_split = round(args['split_ratio'] * len(case_df))
        allset_plt = data_visual(case_df[args['input_col']])
        allset_plt.savefig(os.path.join(args['save_path'], f"{fname}_{args['model']}_input_visualize.png"))

        ## Pre-processing
        feature_sc, ts_train, org_test, ts_test = data_preprocess_v3(case_df, args['input_col'], args['target_col'], args['target_col_di'], train_split, args['hist_p'])
        allset_scaled_plt = data_visual(pd.concat([ts_train.iloc[:, :-1], ts_test.iloc[:, :-1]], axis=0))
        allset_scaled_plt.savefig(os.path.join(args['save_path'], f"{fname}_{args['model']}_input_visualize(scaled).png"))

        ## Sequential Dataset
        feature_cnt_input = ts_train[args['input_col']].shape[1]
        feature_cnt_output = ts_train[args['target_col']].shape[1]
        feature_cnt_di = ts_train[args['target_col_di']].shape[1]
        args['feature_cnt_input'] = feature_cnt_input
        args['feature_cnt_output'] = feature_cnt_output
        args['feature_cnt_di'] = feature_cnt_di
        args['io_col'] = args['input_col'] + ['target']

        ts_train_io = ts_train[args['io_col']]
        ts_train_di = ts_train[args['target_col_di']]
        ts_test_io = ts_test[args['io_col']]
        ts_test_di = ts_test[args['target_col_di']]
        X_train, Y_train, X_train_di, Y_train_di, X_test, Y_test, X_test_di, Y_test_di = data_sequencing(feature_sc, ts_train_io.values, ts_train_di.values, ts_test_io.values, ts_test_di.values, args['hist_p'], args['pred_p'], feature_cnt_input)

        encoder, decoder, encoder_inputs, encoder_states, decoder_inputs, decoder_outputs, decoder_dense, layers, predict_result, model, hist, score = Seq2Seq_run(args, X_train, Y_train, X_train_di, Y_train_di, X_test, Y_test, X_test_di, Y_test_di, ts_test, feature_sc)

        ## Result Visualize
        loss_plt = model_loss_visual(hist, args['epoch'], args['batch'], args['model'])
        loss_plt.savefig(os.path.join(args['save_path'], f"{fname}_{args['model']}_loss.png"))
        loss_plt.show()

        predict_result.to_csv(os.path.join(args['save_path'], f"{fname}_{args['model']}_testset_predict_result.csv"), sep=",")
        predict_result_plt = act_pred_visual(f"Full {args['target_col'][0]} Data / EPO{args['epoch']}_BAT{args['batch']}_MODEL{args['model_name']}_{fname}", predict_result, score)
        predict_result_plt.savefig(os.path.join(args['save_path'], f"{fname}_{args['model']}_testset_predict_result.png"))
        predict_result_plt.show()

        predict_result_expand_plt = act_pred_visual(f"First 3 months of the {args['target_col'][0]} / EPO{args['epoch']}_BAT{args['batch']}_MODEL{args['model_name']}_{fname}", predict_result.head(528), score)
        predict_result_expand_plt.savefig(os.path.join(args['save_path'], f"{fname}_{args['model']}_testset_predict_expand_result.png"))
        predict_result_expand_plt.show()

        ax1, plt1 = plot_parity(f"Full {args['target_col'][0]} Data / EPO{args['epoch']}_BAT{args['batch']}_MODEL{args['model_name']}_{fname}", predict_result, score, 1.5)
        plt1.savefig(os.path.join(args['save_path'], f"{fname}_{args['model']}_testset_predict_parity_result.png"))
        plt1.show()

        score.append(",".join(args["input_col"]))
        score.append(",".join(args["target_col"]))
        score.append(",".join(args["target_col_di"]))
        ## Save Result Score(MAE, RMSE, R2)
        score_df.loc[fname.split('.')[0]] = score
        score_df_all.loc[fname.split('.')[0]] = score
        score_df.to_csv(os.path.join(args['save_path'], f"{fname}_{args['model']}_testset_score.csv"), sep=",", encoding="utf-8-sig")
        score_df_all.to_csv(os.path.join(args['save_path'], f"{fname}_{args['model']}_testset_score_all.csv"), sep=",", encoding="utf-8-sig")

        ## Future prediction
        # input_x : 현시점 ~ hist_p 까지의 데이터
        input_x = ts_test.iloc[-args['hist_p']:, :feature_cnt_input]
        input_x = np.expand_dims(input_x.values, axis=0)
        test_decoder_inputs_zeros = np.zeros(shape=(input_x.shape[0], args['pred_p']))
        test_decoder_inputs_zeros.fill(0.0)

        model_predict = model.predict([input_x, test_decoder_inputs_zeros])
        predict_y = np.zeros(shape=(model_predict.shape[1], feature_cnt_input + 1))
        predict_y[:, -1] = model_predict.reshape(model_predict.shape[0], model_predict.shape[1])

        for i in range(args['feature_cnt_di']):
            predict_y = np.c_[predict_y, np.zeros(predict_y.shape[0])]

        future_predict = feature_sc.inverse_transform(predict_y)[:, -1]

        future_predict_df = time_preprocess(allset_df, future_predict, args['pred_p'])
        future_predict_df.to_csv(os.path.join(args['save_path'], f"{fname}_{args['model']}_future_predict.csv"), sep=",")
        pretty_df = tabulate(future_predict_df, headers='keys', tablefmt='psql')
        print(f"Forecasts for the future {args['pred_p']} hours \n {pretty_df}")

        time_elapsed = time.time() - initial_time
        print(f"The whole process runs for {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {((time_elapsed % 3600) % 60) % 60:0f}s")

    return future_predict_df


def load_data(file_path, file_name, save_path, hist_p, pred_p, max_day, ext):
    file = os.path.join(file_path, file_name)

    try:
        read_df = pd.read_csv(file, header=0, encoding='CP949')
    except FileNotFoundError as e:
        print(str(e))

    save_folder_name = f'real_test/{hist_p}history_{pred_p}period_result_{ext}'
    save_path = os.path.join(save_path, save_folder_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    read_df.rename(columns={'일시': 'time'}, inplace=True)
    read_df.rename(columns={'풍속(m/s)': 'Wind Speed'}, inplace=True)
    read_df.rename(columns={'풍향(deg)': 'Wind Direction'}, inplace=True)
    read_df.rename(columns={'GUST풍속(m/s)': 'GUST Wind Speed'}, inplace=True)
    read_df.rename(columns={'현지기압(hPa)': 'Atmospheric Pressure'}, inplace=True)
    read_df.rename(columns={'습도(%)': 'Humidity'}, inplace=True)
    read_df.rename(columns={'기온(°C)': 'Temperature'}, inplace=True)
    read_df.rename(columns={'수온(°C)': 'Water Temperature'}, inplace=True)
    read_df.rename(columns={'최대파고(m)': 'Maximum Wave Height'}, inplace=True)
    read_df.rename(columns={'유의파고(m)': 'Significant Wave Height'}, inplace=True)
    read_df.rename(columns={'평균파고(m)': 'Average Wave Height'}, inplace=True)
    read_df.rename(columns={'파주기(sec)': 'Wave Period'}, inplace=True)
    read_df.rename(columns={'파향(deg)': 'Wave Direction'}, inplace=True)

    read_df.set_index('time', inplace=True)
    read_df.index.name = 'time'

    data_df = read_df.copy()
    if len(read_df) > max_day:
        data_df = data_df.iloc[:max_day]
    data_df = data_df.dropna(axis=0, how='any')
    data_df_index = data_df[data_df['Water Temperature'] < 8.8].index
    data_df = data_df.drop(data_df_index)
    data_df['C0000'] = np.arange(len(data_df))
    data_df = prep(data_df, 'Water Temperature', 0.3)
    data_df = prep3(data_df, 'Temperature', 10)
    data_df['predict'] = np.nan

    data_df.to_csv(os.path.join(save_path, f"combined_data_prep.csv"), sep=",", encoding="utf-8-sig")

    read_df['Water Temperature (Out)'] = read_df['Water Temperature']
    read_df['Temperature (Out)'] = read_df['Temperature']

    data_df['Water Temperature (Out)'] = data_df['Water Temperature']
    data_df['Temperature (Out)'] = data_df['Temperature']

    return read_df, data_df, save_path

def run():
    # building detection and predict result save
    model_name = 'Seq2Seq Water Temperature'
    split_ratio = 0.9
    max_day = 24 * 365 * 3          # 3 years
    hist_p = 9                      # history period
    pred_p = 3                      # future predict period
    epoch = 25
    batch = 64
    model = 'seq2seq'

    file_path = 'Y:\laboratory\Python\수온예보/20230807_buoy155102/'

    model = 'seq2seq_ei수온di기온3y'
    save_path = 'Y:\laboratory\Python\수온예보/20231108_buoy155102_predict/'
    file_name = 'combined_data.csv'
    read_df, data_df, save_path = load_data(file_path, file_name, save_path, hist_p, pred_p, max_day, f"{epoch}_{batch}")

    ### 입력 컬럼을 맞게 수정하면 됩니다.
    input_base_col = ['Water Temperature']

    ### 고정 Target Column
    target_col = ['Water Temperature']

    ### Decoder Input
    target_col_di = ['Temperature (Out)']

    args = {'split_ratio': split_ratio, 'save_path': save_path, 'hist_p': hist_p, 'pred_p': pred_p, 'model': model,
            'epoch': epoch, 'batch': batch, 'input_col': input_base_col, 'target_col': target_col,
            'target_col_di': target_col_di,
            'model_name': model_name, 'file_name': file_name}

    future_predict = train_test(file_name, args, read_df, data_df)


    # model = 'seq2seq_ei수온기온di기온'
    # save_path = 'X:/Python/수온예보/20230807_buoy155102_predict/'
    # file_name = 'combined_data.csv'
    # read_df, data_df, save_path = load_data(file_path, file_name, save_path, hist_p, pred_p, max_day, f"{epoch}_{batch}")
    #
    # ### 입력 컬럼을 맞게 수정하면 됩니다.
    # input_base_col = ['Water Temperature', 'Temperature']
    #
    # ### 고정 Target Column
    # target_col = ['Water Temperature']
    #
    # ### Decoder Input
    # target_col_di = ['Water Temperature (Out)', 'Temperature (Out)']
    #
    # args = {'split_ratio': split_ratio, 'save_path': save_path, 'hist_p': hist_p, 'pred_p': pred_p, 'model': model,
    #         'epoch': epoch, 'batch': batch, 'input_col': input_base_col, 'target_col': target_col,
    #         'target_col_di': target_col_di,
    #         'model_name': model_name, 'file_name': file_name}
    #
    # future_predict = train_test(file_name, args, read_df, data_df)





if __name__ == '__main__':
    run()


















