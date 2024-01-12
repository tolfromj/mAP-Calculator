import streamlit as st
import pandas as pd
from tqdm import tqdm
import numpy as np
from mAP import mean_average_precision_for_boxes

def gtb(pred_df2):
    gt = []

    file_names2 = pred_df2['image_id'].values.tolist()
    bboxes2 = pred_df2['PredictionString'].values.tolist()

    for i, bbox in enumerate(bboxes2):
        if isinstance(bbox, float):
            # print(f'{file_names2[i]} empty box')
            print()

    for file_name, bbox in tqdm(zip(file_names2, bboxes2)):
        boxes = np.array(str(bbox).strip().split(' '))

        if len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        elif isinstance(bbox, float):
            # print(f'{file_name} empty box')
            print()
            continue
        else:
            raise Exception('error', 'invalid box count')
        for box in boxes:
            gt.append([file_name, box[0], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
    return gt


def pred(pred_df):
    new_pred = []
    file_names = pred_df['image_id'].values.tolist()
    bboxes = pred_df['PredictionString'].values.tolist()

    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            # print(f'{file_names[i]} empty box')
            print()

    for file_name, bbox in tqdm(zip(file_names, bboxes)):
        boxes = np.array(str(bbox).strip().split(' '))

        if len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        elif isinstance(bbox, float):
            # print(f'{file_name} empty box')
            print()
            continue
        else:
            raise Exception('error', 'invalid box count')
        for box in boxes:
            new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
    return new_pred

######################################################

st.title("mAP Calculator")
"""
기준값을 Ground Truth로 삼고, \n
예측대상의 mAP를 계산한다.
"""
st.write('')
"""
리더보드에서 가장 높은 mAP의 csv파일을 기준값에 넣고, \n
리더보드 점수를 예상하고 싶은 csv파일을 비교대상값에 넣는다.
"""
st.write('')
"""
실제 리더보드 점수와는 많은 차이가 있다.\n
0점인지 아닌지만 판단하는 용도로 사용하자.\n
(0점은 잘 잡는다.)
"""
st.write('')
st.write('')
st.write('')
st.subheader("기준값(GT_csv)")
GT_uploaded_file = st.file_uploader("Choose a GT file", type = ["csv"])
if GT_uploaded_file is None:
    st.write('.')
else:
    gt_df = pd.read_csv(GT_uploaded_file)
    st.write('GT.csv has been uploaded. success')
st.write('')
st.write('')
st.write('')
st.subheader("비교대상값(pred_csv)")
PRED_uploaded_file = st.file_uploader("Choose a pred file", type = ["csv"])
if PRED_uploaded_file is None:
    st.write('.')
else:
    pred_df = pd.read_csv(PRED_uploaded_file)
    st.write('pred.csv has been uploaded. success')
    
if GT_uploaded_file and PRED_uploaded_file:
    st.write('')
    st.write('')
    st.write('버튼이 활성화 될 때까지 기달려주세요...')
st.write('')
st.write('')
st.write('')
st.write('')

new_pred = pred(pred_df)
gt = gtb(gt_df)

try : 
    st.write('버튼을 누르면 mAP 계산을 시작합니다.')
    if st.button("calculator", type="primary"):
        if GT_uploaded_file and PRED_uploaded_file:
            st.write('계산중...')
            st.write('(5~40초 정도 걸립니다.)')
            mean_ap1, average_precisions1 = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)
            st.header(f'mAP : {mean_ap1:.4f}')
        else: 
            raise
except:
    st.write('input data and try again')
    st.stop




