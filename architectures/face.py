import pandas as pd

# .pkl 파일 경로
checkpoint_pkl_path = "/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/CHECKPOINT_DST/Celeb-real/id16_0000.faces.pkl"
# faces_pkl_path = "/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/FACES_DST/Celeb-real"

# Excel 파일 저장 경로
excel_file_path = "/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/Understanding/id16_0000.xlsx"

# .pkl 파일을 읽어서 DataFrame으로 변환
df = pd.read_pickle(checkpoint_pkl_path)

# DataFrame을 Excel 파일로 저장
df.to_excel(excel_file_path, index=False)

print(f"Excel 파일이 저장되었습니다: {excel_file_path}")