import streamlit as st
import pandas as pd
import plotly.express as px


############################################
# 1. 페이지 설정
############################################
st.set_page_config(layout="wide", page_title="교과과정 대시보드")


# 주요 과목 (노란색으로 강조된 과목) 리스트
key_subjects = ['산업공학특강', '산업시스템공학', '제조공학', '확률통계', '경영과학2', '실험계획법',
                '산업공학SW활용', '시스템공학특강', '산업공학종합설계(캡스톤디자인)', '복합시스템공학', '산업시스템사례연구']
# 이후에 전필, 전선 추가해서 나눠도 될듯

df = pd.read_csv('learnProcess.csv')
df['주요과목'] = df['교과목 명'].isin(key_subjects)



############################################
# 2. 필터 설정정
############################################
st.sidebar.header('🔍 필터 설정')

df = df.fillna('')  # 안전하게 공란 처리

# 학년 필터 (선택 안 하면 전체 적용)
years = sorted(df['학년'].unique())
selected_years = st.sidebar.multiselect('학년 선택 \n (미선택 시 전체)', years, default=[])

# 학기 필터 (선택 안 하면 전체 적용)
semesters = sorted(df['학기'].unique())
selected_semesters = st.sidebar.multiselect('학기 선택 \n (미선택 시 전체)', semesters, default=[])

# 분야 필터
fields = ['시스템 최적화', '생산 및 물류', '품질 및 응용 통계', 'IT융합', '인간 및 시스템', '시스템 경영']

# 단계 기호 상수화
STAGE_VALUES = ['◎', '○']  # 전공 심화, 전공 핵심 (순서: 심화 우선)

# 필드 컬럼 공백 표준화
for _f in fields:
    if _f in df.columns:
        df[_f] = df[_f].astype(str).str.strip()
selected_fields = st.sidebar.multiselect('분야 선택 \n (미선택 시 전체 분포만)', fields, default=[])

# 주요과목 필터
show_key_only = st.sidebar.checkbox('주요 과목만 보기')


# ---------- 표 스타일링 ----------
def highlight_stage(val: str):
    if val == '◎':  # 전공 심화
        return 'background-color:#ffe3ba; font-weight:bold;'
    if val == '○':  # 전공 핵심
        return 'background-color:#d9ecff;'
    return ''


def styled_table(df_sub: pd.DataFrame, highlight_cols=None):
    if highlight_cols is None:
        highlight_cols = fields
    highlight_target = [c for c in highlight_cols if c in df_sub.columns]
    styler = df_sub.style
    if highlight_target:
        styler = styler.applymap(highlight_stage, subset=highlight_target)
    # 가운데 정렬 (교과목 명 제외)
    center_cols = [c for c in df_sub.columns if c != '교과목 명']
    if center_cols:
        styler = styler.set_properties(subset=center_cols, **{'text-align': 'center'})
    styler = styler.set_properties(subset=['교과목 명'], **{'text-align': 'left'})
    return styler


############################################
# 3. 필터링 로직
############################################

# 학년 / 학기 선택이 없으면 전체 허용
if selected_years:
    year_mask = df['학년'].isin(selected_years)
else:
    year_mask = pd.Series([True] * len(df))

if selected_semesters:
    semester_mask = df['학기'].isin(selected_semesters)
else:
    semester_mask = pd.Series([True] * len(df))

filtered_df = df[year_mask & semester_mask]

# 주요과목 필터
if show_key_only:
    filtered_df = filtered_df[filtered_df['주요과목']]

#############################
# 분야 필터 (OR) & 공란 제거 조건
#############################
if selected_fields:
    # 선택된 모든 과목 중에서 ANY 선택 분야에 단계 기호(◎/○) 있는 행만 유지
    mask = pd.Series(False, index=filtered_df.index)
    for field in selected_fields:
        if field in filtered_df.columns:
            mask = mask | (filtered_df[field].isin(STAGE_VALUES))
    filtered_df = filtered_df[mask]



############################################
# 4. 대시보드 메인 화면 구성
############################################
st.title('📖 교과과정 대시보드')
st.write('좌측 사이드바의 필터를 사용하여 원하는 과목을 탐색할 수 있습니다.')
st.caption('◎ = 전공 심화 / ○ = 전공 핵심')

# 필터링 결과 요약
st.subheader(f"📊 검색 결과: 총 {len(filtered_df)}개 과목")

# 기본 표 (필드 미선택 시 전체 개요 제공)
if not selected_fields:
    base_cols = ['교과목 명', '학년', '학기'] + fields + ['주요과목']
    display_df = filtered_df[base_cols]
    st.dataframe(styled_table(display_df), use_container_width=True)

# 분야 선택 시 단계(○/◎)별 그룹 표시
stage_order = STAGE_VALUES  # ['◎','○'] 심화 -> 핵심
only_fields_selected = selected_fields and (not selected_years) and (not selected_semesters)
if selected_fields:
    # 1) 분야만 선택된 경우: 학년 / 학기별 그룹핑
    if only_fields_selected:
        for field in selected_fields:
            st.markdown(f"### 🔎 {field} 학년·학기별 과목")
            field_df = filtered_df[filtered_df[field].isin(STAGE_VALUES)]
            if field_df.empty:
                st.info(f"{field} 관련 표시할 과목이 없습니다.")
                continue
            field_df = field_df.sort_values(['학년', '학기'])
            for (year, semester), grp in field_df.groupby(['학년', '학기'], sort=True):
                st.markdown(f"- **{year}학년 {semester}학기** ({len(grp)}과목)")
                st.dataframe(styled_table(grp[['교과목 명', '학년', '학기', field, '주요과목']], highlight_cols=[field]), use_container_width=True)
    # 2) 학년/학기 필터도 함께 선택된 경우: 단계(심화/핵심)별 그룹핑 유지
    else:
        for field in selected_fields:
            st.markdown(f"### 🔎 {field} 단계별 과목")
            field_df = filtered_df[filtered_df[field].isin(STAGE_VALUES)]
            if field_df.empty:
                st.info(f"{field} 관련 표시할 과목이 없습니다.")
                continue
            for stage in stage_order:
                stage_df = field_df[field_df[field] == stage]
                if stage_df.empty:
                    continue
                label = '전공 심화' if stage == '◎' else '전공 핵심'
                st.markdown(f"- **{label} ({stage})** ({len(stage_df)}과목)")
                st.dataframe(styled_table(stage_df[['교과목 명', '학년', '학기', field, '주요과목']], highlight_cols=[field]), use_container_width=True)




############################################
# 5. 시각화
############################################

st.subheader('분야별 과목 분포')

target_df_for_chart = filtered_df.copy()

# 아무 필터도 선택하지 않은 경우 (학년/학기/분야 모두 미선택 & 주요과목 체크 X) -> 전체 데이터 분포
no_filter = (not selected_years) and (not selected_semesters) and (not selected_fields) and (not show_key_only)
if no_filter:
    target_df_for_chart = df

if not target_df_for_chart.empty:
    counts = []
    for field in fields:
        count = target_df_for_chart[field].isin(['○', '◎']).sum()
        counts.append({'분야': field, '과목 수': count})
    chart_df = pd.DataFrame(counts)
    fig = px.bar(
        chart_df,
        x='분야',
        y='과목 수',
        title='분야별 과목 분포 (현재 필터 적용)',
        labels={'분야': '과목 분야', '과목 수': '개설된 과목 수'},
        color='분야',
        text='과목 수'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
else:

    st.warning('선택된 조건에 해당하는 과목이 없습니다.')
