"""
Modified by Suji Lee
"""

"""
Index Celeb-DF v2
Image and Sound Processing Lab - Politecnico di Milano
Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini

"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from isplutils.utils import extract_meta_av, extract_meta_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, help='Source dir',
                        required=True)
    # --source 인자를 통해 소스 디렉토리 경로를 전달받습니다. 이 소스 디렉토리가 Celeb-df
    # parser.add_argument('--videodataset', type=Path, default='data/celebdf_videos.pkl',
    #                     help='Path to save the videos DataFrame')
    parser.add_argument('--videodataset', type=Path, default='/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/data/celebdf_videos.pkl',
                        help='Path to save the videos DataFrame')
     

    args = parser.parse_args()

    ## Parameters parsing
    source_dir: Path = args.source
    videodataset_path: Path = args.videodataset

    # Create ouput folder (if doesn't exist)
    videodataset_path.parent.mkdir(parents=True, exist_ok=True)
    # parents=True:
    # 상위 디렉토리가 존재하지 않더라도 경로에 있는 모든 디렉토리를 순서대로 생성합니다.
    # 예를 들어, data 디렉토리가 존재하지 않으면 data부터 만들어 줍니다.
    # exist_ok=True:
    # 디렉토리가 이미 존재하더라도 오류를 발생시키지 않고 넘어갑니다.
    # 디렉토리가 존재하지 않는 경우에만 디렉토리를 생성하도록 설정하는 옵션입니다.

    ## DataFrame
    print("videodataset_path:", videodataset_path)
    if videodataset_path.exists():
        print('Loading video DataFrame')
        df_videos = pd.read_pickle(videodataset_path) 
    else:
        print('Creating video DataFrame')

        split_file = Path(source_dir).joinpath('List_of_testing_videos.txt') #read txt file in the source dir
        if not split_file.exists():
            raise FileNotFoundError('Unable to find "List_of_testing_videos.txt" in {}'.format(source_dir))
        test_videos_df = pd.read_csv(split_file, delimiter=' ', header=0, index_col=1)
        # 2nd column of 'List_of_testing_videos.txt' file.

        ff_videos = Path(source_dir).rglob('*.mp4')
        df_videos = pd.DataFrame(
            {'path': [f.relative_to(source_dir) for f in ff_videos]})

        df_videos['height'] = df_videos['width'] = df_videos['frames'] = np.zeros(len(df_videos), dtype=np.uint16)
        with Pool() as p:
            meta = p.map(extract_meta_av, df_videos['path'].map(lambda x: str(source_dir.joinpath(x))))
        meta = np.stack(meta)
        df_videos.loc[:, ['height', 'width', 'frames']] = meta

        # Fix for videos that av cannot decode properly
        for idx, record in df_videos[df_videos['frames'] == 0].iterrows():
            meta = extract_meta_cv(str(source_dir.joinpath(record['path'])))
            df_videos.loc[idx, ['height', 'width', 'frames']] = meta

        df_videos['class'] = df_videos['path'].map(lambda x: x.parts[0]).astype('category')
        # df_videos['path']: 각 비디오 파일의 상대 경로입니다.
        # lambda x: x.parts[0]: path 열의 경로에서 최상위 폴더명을 추출. 
        # .astype('category'): 결과를 범주형(category) 데이터로 변환하여 저장 (메모리 최적화)

        df_videos['label'] = df_videos['class'].map(
            lambda x: True if x == 'Celeb-synthesis' else False)  # True is FAKE, False is REAL
            # if class(folder name) == 'Celeb-synthesis', label == True (FAKE)
            # if class(folder name) != 'Celeb-synthesis', label == False (REAL)
        df_videos['split'] = df_videos['path'].map(lambda x: x.parts[1])
        df_videos['name'] = df_videos['path'].map(lambda x: x.with_suffix('').name)

        df_videos['original'] = -1 * np.ones(len(df_videos), dtype=np.int16)
        df_videos.loc[(df_videos['label'] == True), 'original'] = \
            df_videos[(df_videos['label'] == True)]['name'].map(
                lambda x: df_videos.index[
                    np.flatnonzero(df_videos['name'] == '_'.join([x.split('_')[0], x.split('_')[2]]))[0]]
            )

        df_videos['test'] = df_videos['path'].map(str).isin(test_videos_df.index)
        # 'path' example: 'Celeb-synthesis/id3_id1_0001.mp4'
        # If path is in 2nd column of txt file (the list) -> true (= test dataset), else false (= train dataset)
        # List_of_testing_videos.txt
        # df_videos['path']의 경로들을 문자열로 변환한 뒤, test_videos_df.index와 비교하여 True/False 값을 반환합니다.
        # test_videos_df에는 test 데이터에 해당하는 파일 이름들이 인덱스로 지정되어 있어, 
        # test 비디오와 일치하는 파일 경로는 True, 일치하지 않으면 False로 저장됩니다.

        print('Saving video DataFrame to {}'.format(videodataset_path))
        df_videos.to_pickle(str(videodataset_path)) # error의 문제가 되는 부분; videodataset_path 위치에 data/celebdf_videos.pkl 파일이 생성; 해결

    print('Real videos: {:d}'.format(sum(df_videos['label'] == 0)))
    print('Fake videos: {:d}'.format(sum(df_videos['label'] == 1)))

    # Check dataframe on terminal
    pd.set_option('display.max_columns', None)
    print(df_videos.head(20))
    print(df_videos.info())

    # Check dataframe on excel format
    df_videos.to_excel("celebdf_videos.xlsx", index=False)


if __name__ == '__main__':
    main()