# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="6个月PFS风险预测计算器", page_icon="📊", layout="centered")
st.title("📊 6个月无进展生存（PFS）风险预测模型")
st.markdown("**基于治疗前临床及影像特征 + 3个月LI‑RADS 2024评估**")
st.markdown("---")

@st.cache_resource
def load_model():
    model = joblib.load('model_6m_pfs.pkl')
    scaler = joblib.load('scaler_6m_pfs.pkl')
    feature_names = joblib.load('feature_names_6m.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_model()
st.success("模型加载成功！")

label_mapping = {
    '年龄': '年龄',
    '性别': '性别',
    'BMI（正常范围=0，低于正常=1，高于正常=2）': 'BMI分类',
    '吸烟史（有=1，无=0）': '吸烟史',
    '病因（乙肝=1，其他=0）': '病因（乙肝）',
    'Child-Pugh分级': 'Child-Pugh分级',
    'ALBI分级（纳入模型）': 'ALBI分级',
    '体能状态评分（ECOGPS）': 'ECOG PS',
    '总胆红素': '总胆红素 (μmol/L)',
    '白蛋白': '白蛋白 (g/L)',
    'AFP': 'AFP (ng/mL)',
    'NLR': 'NLR',
    '凝血酶原延长秒数': '凝血酶原延长 (s)',
    '肿瘤数目（1个=0，多个=1）': '肿瘤数目',
    '肿瘤最大径': '肿瘤最大径 (mm)',
    'LR-TIV分级': 'LR-TIV分级',
    'BCLC分期': 'BCLC分期',
    '肝外转移（基线）': '肝外转移',
    '肝硬化': '肝硬化',
    '门脉高压': '门脉高压',
    'TACE次数': 'TACE次数',
    'TACE类型（AB=1，BC/AC=2，单项=0）': 'TACE类型',
    '是否接受肝移植/肝切除': '肝移植/肝切除史',
    '靶向药物': '靶向药物',
    '免疫药物（双达=1（信迪利+贝伐）；其他=0）': '免疫药物（双达）',
    '肿瘤部位': '肿瘤部位',
    '瘤周高灌注（有=1，无=0）': '瘤周高灌注',
    '快进（有=1，无=0）': '快进',
    '快速廓清（有=1，无=0）': '快速廓清',
    '包膜强化（有=1，无=0）': '包膜强化',
    '内部有无坏死（有=1，无=0）': '内部坏死',
    '环形强化': '环形强化',
    'LI-RADS2024': 'LI-RADS 2024分类'
}

input_config = {
    '年龄': {'type': 'number', 'min': 20.0, 'max': 90.0, 'value': 55.0},
    '性别': {'type': 'select', 'options': [0, 1], 'labels': ['女', '男']},
    'BMI（正常范围=0，低于正常=1，高于正常=2）': {'type': 'select', 'options': [0, 1, 2], 'labels': ['正常', '低于正常', '高于正常']},
    '吸烟史（有=1，无=0）': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    '病因（乙肝=1，其他=0）': {'type': 'select', 'options': [0, 1], 'labels': ['其他', '乙肝']},
    'Child-Pugh分级': {'type': 'select', 'options': [0, 1], 'labels': ['A级', 'B级']},
    'ALBI分级（纳入模型）': {'type': 'number', 'min': -3.5, 'max': 1.0, 'value': -2.5, 'step': 0.1},
    '体能状态评分（ECOGPS）': {'type': 'number', 'min': 0.0, 'max': 4.0, 'value': 1.0},
    '总胆红素': {'type': 'number', 'min': 0.0, 'max': 500.0, 'value': 20.0},
    '白蛋白': {'type': 'number', 'min': 10.0, 'max': 60.0, 'value': 38.0},
    'AFP': {'type': 'number', 'min': 0.0, 'max': 100000.0, 'value': 10.0},
    'NLR': {'type': 'number', 'min': 0.0, 'max': 20.0, 'value': 2.5, 'step': 0.1},
    '凝血酶原延长秒数': {'type': 'number', 'min': 0.0, 'max': 10.0, 'value': 1.0, 'step': 0.1},
    '肿瘤数目（1个=0，多个=1）': {'type': 'select', 'options': [0, 1], 'labels': ['单发', '多发']},
    '肿瘤最大径': {'type': 'number', 'min': 0.0, 'max': 300.0, 'value': 60.0},
    'LR-TIV分级': {'type': 'select', 'options': [0, 1, 2, 3], 'labels': ['0', '1', '2', '3']},
    'BCLC分期': {'type': 'select', 'options': [0, 1, 2, 3], 'labels': ['0', 'A', 'B', 'C']},
    '肝外转移（基线）': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    '肝硬化': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    '门脉高压': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    'TACE次数': {'type': 'number', 'min': 1.0, 'max': 20.0, 'value': 2.0},
    'TACE类型（AB=1，BC/AC=2，单项=0）': {'type': 'select', 'options': [0, 1, 2], 'labels': ['单项', 'AB', 'BC/AC']},
    '是否接受肝移植/肝切除': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    '靶向药物': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    '免疫药物（双达=1（信迪利+贝伐）；其他=0）': {'type': 'select', 'options': [0, 1], 'labels': ['其他', '双达']},
    '肿瘤部位': {'type': 'select', 'options': ['A', 'B', 'AB', '其他'], 'labels': ['A', 'B', 'AB', '其他']},
    '瘤周高灌注（有=1，无=0）': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    '快进（有=1，无=0）': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    '快速廓清（有=1，无=0）': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    '包膜强化（有=1，无=0）': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    '内部有无坏死（有=1，无=0）': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    '环形强化': {'type': 'select', 'options': [0, 1], 'labels': ['无', '有']},
    'LI-RADS2024': {'type': 'select', 'options': [0, 1, 2], 'labels': ['无活性', '不确定', '有活性']},
}

with st.form(key='prediction_form'):
    st.subheader("请输入患者特征")
    col1, col2 = st.columns(2)
    feature_keys = feature_names
    mid = len(feature_keys) // 2
    left_features = feature_keys[:mid]
    right_features = feature_keys[mid:]
    input_values = {}
    with col1:
        for feat in left_features:
            config = input_config.get(feat, {})
            label = label_mapping.get(feat, feat)
            if config.get('type') == 'number':
                val = st.number_input(label, min_value=float(config.get('min', 0.0)), max_value=float(config.get('max', 100.0)), value=float(config.get('value', 0.0)), step=float(config.get('step', 1.0)), key=feat)
            else:
                options = config.get('options', [0, 1])
                labels = config.get('labels', [str(o) for o in options])
                idx = st.selectbox(label, range(len(options)), format_func=lambda i: labels[i], key=feat)
                val = options[idx]
            input_values[feat] = val
    with col2:
        for feat in right_features:
            config = input_config.get(feat, {})
            label = label_mapping.get(feat, feat)
            if config.get('type') == 'number':
                val = st.number_input(label, min_value=float(config.get('min', 0.0)), max_value=float(config.get('max', 100.0)), value=float(config.get('value', 0.0)), step=float(config.get('step', 1.0)), key=feat)
            else:
                options = config.get('options', [0, 1])
                labels = config.get('labels', [str(o) for o in options])
                idx = st.selectbox(label, range(len(options)), format_func=lambda i: labels[i], key=feat)
                val = options[idx]
            input_values[feat] = val
    submitted = st.form_submit_button("预测6个月PFS风险", type="primary")

if submitted:
    input_array = np.array([input_values[feat] for feat in feature_names]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prob_progress = model.predict_proba(input_scaled)[0, 1]
    prob_no_progress = 1 - prob_progress
    st.markdown("---")
    st.subheader("📈 预测结果")
    col1, col2, col3 = st.columns(3)
    col1.metric("6个月内无进展概率", f"{prob_no_progress*100:.1f}%")
    col2.metric("6个月内进展概率", f"{prob_progress*100:.1f}%")
    if prob_progress > 0.5:
        col3.error("🔴 高风险")
        st.warning("该患者6个月内进展风险较高，建议密切随访或考虑调整治疗策略。")
    else:
        col3.success("🟢 低风险")
        st.success("该患者6个月内进展风险较低，可继续当前治疗方案。")
    with st.expander("查看输入特征值"):
        df_input = pd.DataFrame({'特征': [label_mapping.get(f, f) for f in feature_names], '值': [input_values[f] for f in feature_names]})
        st.dataframe(df_input)
    st.caption("⚠️ 本工具仅基于本研究数据开发，仅供科研参考，实际临床决策需结合患者具体情况。")

st.markdown("---")
st.caption("模型基于154例患者数据开发，AUC=0.821。如有疑问，请联系作者。")