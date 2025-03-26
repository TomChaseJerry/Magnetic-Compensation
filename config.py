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
    flights = ["Flt1003"]
    test_lines = {"Flt1002": ['1002.02'], "Flt1003": ['1003.04', '1003.08'], "Flt1004": ['4015.00']}
    train_ttlim = {"Flt1003": [[50713.00, 54497.00]]}
    # test_ttlim = {"Flt1002": [[46370.00, 47600.00]], "Flt1003": [[54639.00, 59475.00], [60243.00, 64586.00]],
    #               "Flt1004": [[52256.06, 52862.14]]}
    test_ttlim = {"Flt1003": [[54639.00, 59475.00]]}

    # features
    all_features = ['utm_x', 'utm_y', 'utm_z', 'msl', 'lat', 'lon', 'baro', 'dem', 'mag_2_uc', 'mag_3_uc', 'mag_4_uc', 'mag_5_uc',
                    'cur_com_1', 'cur_ac_hi', 'cur_ac_lo', 'cur_tank', 'cur_flap', 'cur_strb', 'cur_srvo_o', 'cur_srvo_m',
                    'cur_srvo_i', 'cur_heat', 'cur_acpwr', 'cur_outpwr', 'cur_bat_1', 'cur_bat_2', 'vol_acpwr', 'vol_outpwr',
                    'vol_bat_1', 'vol_bat_2', 'vol_res_p', 'vol_res_n', 'vol_back_p', 'vol_back_n', 'vol_gyro_1', 'vol_gyro_2',
                    'vol_acc_p', 'vol_acc_n', 'vol_block', 'vol_back', 'vol_srvo', 'vol_cabt', 'vol_fan']
    selected_features = ['vn', 've', 'vd', 'baro', 'ins_pitch', 'ins_roll', 'ins_yaw', 'cur_ac_hi', 'cur_strb', 'cur_heat',
                         'vol_bat_1', 'vol_block', 'cur_com_1', 'cur_ac_lo', 'cur_tank', 'cur_flap', 'vol_bat_2', 'mag_5_uc']

    # run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    lr = 1e-3
    epochs = 30
    is_pca = False
    model_type = 'MLP'  # ['MLP', 'CNN']
    seq_len = 20
