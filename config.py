import torch


class config(object):
    # file path
    df_all_path = "datasets/dataframes/df_all.csv"
    df_cal_path = "datasets/dataframes/df_cal.csv"
    df_event_path = "datasets/dataframes/df_event.csv"
    df_flight_path = "datasets/dataframes/df_flight.csv"
    df_map_path = "datasets/dataframes/df_map.csv"
    df_nav_path = "datasets/dataframes/df_nav.csv"

    # flight line
    flights_20 = ["Flt1002", "Flt1003", "Flt1004", "Flt1005", "Flt1006", "Flt1007"]
    flights_21 = ["Flt2001", "Flt2002", "Flt2004", "Flt2005", "Flt2006", "Flt2007", "Flt2008", "Flt2015", "Flt2016", "Flt2017"]
    flight_nums = [2, 3, 4, 6, 7]

    # test_lines = {"Flt1002": ['1002.17'], "Flt1003": ['1003.08'], "Flt1004": ['4012.00', '4001.00'],
    #               "Flt1005": ['4004.00', '4003.00', '4002.00'], "Flt1006": ['1006.08'], "Flt1007": ['1007.06']}
    # train_ttlim = {"Flt1002": [[57489.00, 59946.00], [61360.00, 63035.00]],
    #                "Flt1003": [[59926.00, 60105.00], [50713.00, 54497.00]],
    #                "Flt1004": [[45175.33, 52862.14]],
    #                "Flt1005": [[48085.00, 51848.30]],
    #                "Flt1006": [[49000.00, 53286.00], [53855.00, 54510.00]],
    #                "Flt1007": [[48024.00, 51880.00]]}
    # test_ttlim = {"Flt1002": [[63935.00, 65812.00]], "Flt1003": [[60243.00, 64586.00]], "Flt1004": [[54518.19, 55891.29]],
    #               "Flt1005": [[53471.63, 55605.59]], "Flt1006": [[55770.00, 56609.00]], "Flt1007": [[57770.00, 63010.00]]}

    # features
    all_sensors = ['mag_2_uc', 'mag_3_uc', 'mag_4_uc', 'mag_5_uc', 'cur_com_1', 'cur_ac_hi', 'cur_ac_lo', 'cur_tank', 'cur_flap',
                   'cur_strb', 'cur_srvo_o', 'cur_srvo_m', 'cur_srvo_i', 'cur_heat', 'cur_acpwr', 'cur_outpwr', 'cur_bat_1',
                   'cur_bat_2', 'vol_acpwr', 'vol_outpwr', 'vol_bat_1', 'vol_bat_2', 'vol_res_p', 'vol_res_n', 'vol_back_p',
                   'vol_back_n', 'vol_gyro_1', 'vol_gyro_2', 'vol_acc_p', 'vol_acc_n', 'vol_block', 'vol_back', 'vol_srvo',
                   'vol_cabt', 'vol_fan']

    selected_features = ['mag_3_c', 'mag_4_c', 'mag_5_c', 'vol_bat_1', 'vol_bat_2', 'ins_vn', 'ins_vw', 'ins_vu', 'cur_heat',
                         'cur_flap', 'cur_ac_lo', 'cur_tank', 'ins_pitch', 'ins_roll', 'ins_yaw', 'baro', 'line', 'model_y']

    # run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    lr = 1e-3
    weight_decay = 1e-3
    epochs = 50
    is_pca = False
    model_type = 'CNN'  # ['MLP', 'CNN', 'Transformer']
    is_PINNs = False
    test_is_Flt1007 = True
    seq_len = 20
