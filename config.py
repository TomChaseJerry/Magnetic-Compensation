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
    batch_size = 64
    lr = 1e-3
    weight_decay = 1e-3
    epochs = 10
    is_pca = False
    model_type = 'CNN'  # ['MLP', 'CNN', 'Transformer']
    is_PINNs = False
    test_is_Flt1007 = True
    seq_len = 20
