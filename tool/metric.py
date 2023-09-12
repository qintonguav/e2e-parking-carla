import math
import logging
import pathlib
import pandas as pd
import glob


eva_result_path = pathlib.Path('xxx/eva_result/2023_09_07_21_55_25')

if __name__ == '__main__':
    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    csv_files = glob.glob(f'{eva_result_path}/*_result.csv')
    for i in range(16):
        df_row_i = pd.DataFrame()
        for csv in csv_files:
            df_csv = pd.read_csv(csv)
            row_i = df_csv.iloc[[i]]
            df_row_i = df_row_i.append(row_i)

        row_i_mean = df_row_i.mean(axis=0).to_frame().T
        row_i_std = (df_row_i.std(axis=0, ddof=0).to_frame().T / math.sqrt(6))

        df_mean = pd.concat([df_mean, row_i_mean], axis=0)
        df_std = pd.concat([df_std, row_i_std], axis=0)

    row_mean = df_mean.mean(axis=0).to_frame().T
    row_std = df_std.mean(axis=0).to_frame().T
    df_mean = pd.concat([df_mean, row_mean], axis=0)
    df_std = pd.concat([df_std, row_std], axis=0)

    name = ['2-1', '2-3', '2-5', '2-7', '2-9', '2-11', '2-13', '2-15',
            '3-1', '3-3', '3-5', '3-7', '3-9', '3-11', '3-13', '3-15', 'Avg']
    df_mean.index = name
    df_std.index = name

    pd.set_option('display.max_columns', 1000)
    pd.options.display.float_format = '{:,.3f}'.format

    logging.info('Mean')
    print(df_mean)

    logging.info('Std')
    print(df_std)

    df_mean.to_csv(eva_result_path / 'result_mean.csv')
    df_std.to_csv(eva_result_path / 'result_std.csv')
