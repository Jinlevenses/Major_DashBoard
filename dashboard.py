import streamlit as st
import pandas as pd
import plotly.express as px

# 0. 페이지 설정
st.set_page_config(layout="wide", page_title="교과과정 대시보드")

# 1. 데이터 로드 및 전처리
# 이미지의 데이터를 수동으로 DataFrame으로 변환했습니다.


# 주요 과목 (노란색으로 강조된 과목) 리스트
key_subjects = ['산업공학특강', '산업시스템공학', '제조공학', '확률통계', '경영과학2', '실험계획법',
                '산업공학SW활용', '시스템공학특강', '산업공학종합설계(캡스톤디자인)', '복합시스템공학', '산업시스템사례연구']
df = pd.read_csv('learnProcess.csv')
df['주요과목'] = df['교과목 명'].isin(key_subjects)

# 2. 사이드바 필터 설정
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
STAGE_VALUES = ['◎', '○']  # 전공 심화, 전공 핵심 (순서 중요: 심화 우선)

# 필드 컬럼 공백 표준화 (예방적)
for _f in fields:
    if _f in df.columns:
        df[_f] = df[_f].astype(str).str.strip()
selected_fields = st.sidebar.multiselect('분야 선택 \n (미선택 시 전체 분포만)', fields, default=[])

# 주요과목 필터
show_key_only = st.sidebar.checkbox('주요 과목만 보기')

# 표 형태 선택
view_mode = st.sidebar.radio('표 형태 선택', ['과목 별', '분야 - 학년/학기'], index=1)


# ---------- 표 스타일링 헬퍼 ----------
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


def build_matrix_html(src_df: pd.DataFrame) -> str:
    """분야 - 학년&학기 """
    if src_df.empty:
        return '<p>데이터 없음</p>'
    # 사용되는 (학년,학기) 조합 추출 (정렬)
    year_sem = sorted(src_df[['학년', '학기']].drop_duplicates().itertuples(index=False, name=None))
    # 헤더 (2줄 구조) 직접 HTML 구성
    years = sorted({y for y, _ in year_sem})
    # 학년별 학기 목록
    year_to_sems = {y: [s for (yy, s) in year_sem if yy == y] for y in years}

    def color_course(row_val: str, course: str) -> str:
        if row_val == '◎':  # 심화
            return f'<span style="color:#d40000;font-weight:600;">{course}</span>'
        if row_val == '○':  # 핵심
            return f'<span style="color:#0041d9;">{course}</span>'
        return f'<span style="color:#000;">{course}</span>'

    # 셀 구성: 해당 필드 컬럼이 ◎/○ 인 과목만 포함
    table_rows = []
    for field in fields:
        row_cells = []
        for (y, s) in year_sem:
            sub = src_df[(src_df['학년'] == y) & (src_df['학기'] == s) & (src_df[field].isin(STAGE_VALUES))]
            if sub.empty:
                row_cells.append('')
            else:
                rendered = [color_course(sub.iloc[i][field], sub.iloc[i]['교과목 명']) for i in range(len(sub))]
                row_cells.append('<br>'.join(rendered))
        table_rows.append((field, row_cells))

    # HTML 렌더링
    # 스타일 (스크롤 및 셀 스타일)
    html = ["""
    <style>
    .matrix-table {border-collapse:collapse; width:100%; font-size:13px; background:#fff;}
    .matrix-table th, .matrix-table td {border:1px solid #ddd; padding:6px 8px; vertical-align:top;}
    .matrix-table th {background:#f1f3f8; text-align:center;}
    .matrix-table td {min-width:110px;}
    .matrix-top-header {background:#e2e8f5;font-weight:600;}
    .field-col {background:#fafafa;font-weight:600; position:sticky; left:0;}
    .matrix-wrapper {overflow-x:auto; border:1px solid #ddd;}
    </style>
    <div class="matrix-wrapper">
    <table class="matrix-table">
    """]

    # 첫째 줄: 학년 헤더 (colspan)
    html.append('<thead>')
    html.append('<tr>')
    html.append('<th class="matrix-top-header" rowspan="2" style="left:0;">분야</th>')
    for y in years:
        span = len(year_to_sems[y])
        html.append(f'<th class="matrix-top-header" colspan="{span}">{y}학년</th>')
    html.append('</tr>')
    # 둘째 줄: 학기
    html.append('<tr>')
    for y in years:
        for s in year_to_sems[y]:
            html.append(f'<th>{s}학기</th>')
    html.append('</tr>')
    html.append('</thead>')

    # 몸체
    html.append('<tbody>')
    for field, cells in table_rows:
        html.append('<tr>')
        html.append(f'<td class="field-col">{field}</td>')
        for c in cells:
            html.append(f'<td>{c}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    return ''.join(html)


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


# 4. 대시보드 메인 화면 구성
st.title('📖 교과과정 대시보드')
st.write('좌측 사이드바의 필터를 사용하여 원하는 과목을 탐색할 수 있습니다.')
st.caption('◎ = 전공 심화 / ○ = 전공 핵심')
st.caption('빨간 글씨 - 전공 심화, 파란 글씨 - 전공 핵심')


# 기본 표 (필드 미선택 시 전체 개요 제공)
if view_mode == '과목 별' and not selected_fields:
    base_cols = ['교과목 명', '학년', '학기'] + fields + ['주요과목']
    display_df = filtered_df[base_cols]
    st.dataframe(styled_table(display_df), use_container_width=True)

# 행렬 보기 (필터링된 데이터 전체 사용)
if view_mode == '분야 - 학년/학기':
    st.markdown('#### 📌 분야 x 학년/학기 기준')
    st.markdown(build_matrix_html(filtered_df), unsafe_allow_html=True)

# 분야 선택 시 단계(○/◎)별 그룹 표시
stage_order = STAGE_VALUES  # ['◎','○'] 심화 -> 핵심
only_fields_selected = selected_fields and (not selected_years) and (not selected_semesters)
if selected_fields and view_mode == '과목 별':
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


# 5. 시각화
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
