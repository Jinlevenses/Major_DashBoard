import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==============================
# 0) 페이지 설정
# ==============================
st.set_page_config(layout="wide", page_title="교과과정 대시보드")

# ==============================
# 1) 상수/유틸
# ==============================
FIELDS = ['시스템 최적화', '생산 및 물류', '품질 및 응용 통계', 'IT융합', '인간 및 시스템', '시스템 경영']
LEVEL_SYMBOLS = ['◎', '○']     # 심화, 핵심
LEVEL_MAP = {'◎': '심화', '○': '핵심'}
SEM_ORDER = [(y, s) for y in [1,2,3,4] for s in [1,2]]  # 1-1,1-2,...,4-2 순서

KEY_SUBJECTS = [
   '산업공학입문', '프로그래밍기초', '확률통계', '경영과학1', 
    '생산관리1', '산업공학종합설계(캡스톤디자인)'
]

def _normalize_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace('\n',' ').str.strip().replace({'#N/A':'', 'nan':''})

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 표준화
    df.columns = [c.replace('\n',' ').strip() for c in df.columns]
    for c in ['교과목 명', '학년', '학기']:
        df[c] = _normalize_series(df[c])
    for f in FIELDS:
        if f in df.columns:
            df[f] = _normalize_series(df[f])
        else:
            df[f] = ''  # 누락 방어

    # 숫자형 변환 (학년/학기)
    df['학년'] = pd.to_numeric(df['학년'], errors='coerce')
    df['학기'] = pd.to_numeric(df['학기'], errors='coerce')
    df = df.dropna(subset=['교과목 명', '학년', '학기'])
    df['학년'] = df['학년'].astype(int)
    df['학기'] = df['학기'].astype(int)

    # 주요과목 플래그
    df['주요과목'] = df['교과목 명'].isin(KEY_SUBJECTS)

    # long 형 변환 (과목×분야×심화/핵심)
    wide = df[['교과목 명','학년','학기','주요과목'] + FIELDS].copy()
    long = wide.melt(
        id_vars=['교과목 명','학년','학기','주요과목'],
        value_vars=FIELDS,
        var_name='분야',
        value_name='기호'
    )
    long = long[long['기호'].isin(LEVEL_SYMBOLS)]  # ◎/○만
    long['전문성'] = long['기호'].map(LEVEL_MAP)
    long['학기라벨'] = long['학년'].astype(str) + '-' + long['학기'].astype(str)

    # 정렬키
    sem_key = {f"{y}-{s}": i for i, (y,s) in enumerate(SEM_ORDER)}
    long['sem_order'] = long['학기라벨'].map(sem_key).fillna(9999).astype(int)
    long['field_order'] = long['분야'].apply(lambda x: FIELDS.index(x) if x in FIELDS else 999)

    return df, long

# ==============================
# 2) 데이터 로드
# ==============================
RAW_DF, LONG_DF = load_data('learnProcess.csv')

st.sidebar.header('🔍 필터 설정')

years = sorted(RAW_DF['학년'].unique().tolist())
sems  = sorted(RAW_DF['학기'].unique().tolist())
FIELDS = ['시스템 최적화', '생산 및 물류', '품질 및 응용 통계', 'IT융합', '인간 및 시스템', '시스템 경영']

DEFAULTS = {
    "selected_years": [],
    "selected_sems": [],
    "selected_fields": [],
    "show_key_only": False,
    "view_mode": "분야 - 학년/학기",
}

# 1) 위젯을 만들기 전에 세션 상태 기본값을 보장
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 2) 초기화 콜백(위젯 생성 후 값을 바꿔도 안전함: 콜백 → rerun)
def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    # 콜백 종료 후 Streamlit이 자동 rerun 해줍니다. (st.rerun() 불필요)

# 3) 위젯 생성 (key만 주면 Streamlit이 session_state 값을 자동으로 사용)
st.sidebar.multiselect(
    '학년 선택 (미선택 시 전체)', years,
    default=st.session_state["selected_years"],
    key="selected_years"
)

st.sidebar.multiselect(
    '학기 선택 (미선택 시 전체)', sems,
    default=st.session_state["selected_sems"],
    key="selected_sems"
)

st.sidebar.multiselect(
    '분야 선택 (미선택 시 전체)', FIELDS,
    default=st.session_state["selected_fields"],
    key="selected_fields"
)

st.sidebar.checkbox(
    '주요 과목만 보기',
    value=st.session_state["show_key_only"],
    key="show_key_only"
)

# radio는 value 또는 index 중 하나만 써야 충돌이 없습니다. value로 고정 추천.
st.sidebar.radio(
    '표 형태 선택',
    options=['과목 별', '분야 - 학년/학기'],
    key="view_mode"
)

# 4) 필터 초기화 버튼: on_click 콜백 사용 (직접 session_state 수정 X)
st.sidebar.button('필터 초기화', type='primary', on_click=reset_filters)

# 5) 이후 로직에서 사용할 현재 값
selected_years   = st.session_state["selected_years"]
selected_sems    = st.session_state["selected_sems"]
selected_fields  = st.session_state["selected_fields"]
show_key_only    = st.session_state["show_key_only"]
view_mode        = st.session_state["view_mode"]

# ==============================
# 4) 필터 적용
# ==============================
df = RAW_DF.copy()
if selected_years:
    df = df[df['학년'].isin(selected_years)]
if selected_sems:
    df = df[df['학기'].isin(selected_sems)]
if show_key_only:
    df = df[df['주요과목']]

# 분야 필터(OR) → wide 기준으로 해당 분야에 ◎/○가 하나라도 있는 과목만
if selected_fields:
    mask = np.zeros(len(df), dtype=bool)
    for f in selected_fields:
        if f in df.columns:
            mask |= df[f].isin(LEVEL_SYMBOLS).values
    df = df[mask]

# LONG_DF에도 동일 조건 적용
l = LONG_DF.copy()
if selected_years:
    l = l[l['학년'].isin(selected_years)]
if selected_sems:
    l = l[l['학기'].isin(selected_sems)]
if show_key_only:
    l = l[l['주요과목']]
if selected_fields:
    l = l[l['분야'].isin(selected_fields)]

# ==============================
# 5) 헤더 / KPI
# ==============================
st.title('📖 교과과정 대시보드')
st.write('좌측 사이드바의 필터를 사용하여 원하는 과목을 탐색할 수 있습니다.')
st.caption('◎ = 전공 심화 / ○ = 전공 핵심  ·  빨간 글씨 = 심화, 파란 글씨 = 핵심')

c1, c2, c3, c4 = st.columns(4)
c1.metric('총 과목 수(필터 후)', f"{df['교과목 명'].nunique():,}")
c2.metric('분야 수(필터 후)', f"{l['분야'].nunique():,}")
if len(l) > 0:
    deep_ratio = (l['전문성'].eq('심화').mean()) * 100
    core_ratio = (l['전문성'].eq('핵심').mean()) * 100
else:
    deep_ratio, core_ratio = 0, 0
c3.metric('심화 비중', f"{deep_ratio:.0f}%")
c4.metric('핵심 비중', f"{core_ratio:.0f}%")

# ==============================
# 6) 표/행렬
# ==============================
def highlight_stage(val: str):
    if val == '◎':  # 전공 심화
        return 'background-color:#ffe3ba; font-weight:bold;'
    if val == '○':  # 전공 핵심
        return 'background-color:#d9ecff;'
    return ''

def styled_table(df_sub: pd.DataFrame, highlight_cols=None):
    if highlight_cols is None:
        highlight_cols = FIELDS
    highlight_target = [c for c in highlight_cols if c in df_sub.columns]
    styler = df_sub.style
    if highlight_target:
        styler = styler.applymap(highlight_stage, subset=highlight_target)
    center_cols = [c for c in df_sub.columns if c != '교과목 명']
    if center_cols:
        styler = styler.set_properties(subset=center_cols, **{'text-align': 'center'})
    styler = styler.set_properties(subset=['교과목 명'], **{'text-align': 'left'})
    return styler

def build_matrix_html(src_df: pd.DataFrame, field_rows=None) -> str:
    """분야 - 학년&학기 행렬 (field_rows 지정 시 해당 분야만 표시)."""
    fields = field_rows if field_rows else FIELDS
    if src_df.empty:
        return '<p>데이터 없음</p>'

    used = sorted(set([(int(y), int(s)) for y, s in src_df[['학년','학기']].itertuples(index=False)]),
                  key=lambda ys: SEM_ORDER.index(ys) if ys in SEM_ORDER else 999)

    years = sorted({y for y, _ in used})
    year_to_sems = {y: [s for (yy, s) in used if yy == y] for y in years}

    def color_course(mark: str, course: str) -> str:
        if mark == '◎':
            return f'<span style="color:#d40000;font-weight:600;">{course}</span>'
        if mark == '○':
            return f'<span style="color:#0041d9;">{course}</span>'
        return f'<span style="color:#000;">{course}</span>'

    table_rows = []
    for field in fields:
        row_cells = []
        for (y, s) in used:
            sub = src_df[(src_df['학년'] == y) & (src_df['학기'] == s) & (src_df[field].isin(LEVEL_SYMBOLS))]
            if sub.empty:
                row_cells.append('')
            else:
                rendered = [color_course(sub.iloc[i][field], sub.iloc[i]['교과목 명']) for i in range(len(sub))]
                row_cells.append('<br>'.join(rendered))
        table_rows.append((field, row_cells))

    html = ["""
    <style>
    .matrix-table {border-collapse:collapse; width:100%; font-size:13px; background:#fff;}
    .matrix-table th, .matrix-table td {border:1px solid #ddd; padding:6px 8px; vertical-align:top;}
    .matrix-table th {background:#f1f3f8; text-align:center;}
    .matrix-table td {min-width:120px;}
    .matrix-top-header {background:#e2e8f5;font-weight:600;}
    .field-col {background:#fafafa;font-weight:600; position:sticky; left:0;}
    .matrix-wrapper {overflow-x:auto; border:1px solid #ddd;}
    </style>
    <div class="matrix-wrapper">
    <table class="matrix-table">
    """]
    html.append('<thead><tr>')
    html.append('<th class="matrix-top-header" rowspan="2" style="left:0;">분야</th>')
    for y in years:
        span = len(year_to_sems[y])
        html.append(f'<th class="matrix-top-header" colspan="{span}">{y}학년</th>')
    html.append('</tr><tr>')
    for y in years:
        for s in year_to_sems[y]:
            html.append(f'<th>{s}학기</th>')
    html.append('</tr></thead>')
    html.append('<tbody>')
    for field, cells in table_rows:
        html.append('<tr>')
        html.append(f'<td class="field-col">{field}</td>')
        for c in cells:
            html.append(f'<td>{c}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    return ''.join(html)

# 상단 표/행렬
st.caption('※ 표 또는 행렬이 길 경우, 가로 스크롤로 보세요.')
if view_mode == '과목 별':
    base_cols = ['교과목 명', '학년', '학기'] + FIELDS + ['주요과목']
    show_df = df[base_cols].sort_values(['학년','학기','교과목 명'])
    st.dataframe(styled_table(show_df), use_container_width=True, height=420)

if view_mode == '분야 - 학년/학기':
    st.markdown('#### 📌 분야 × 학년/학기')
    st.markdown(build_matrix_html(df, selected_fields if selected_fields else FIELDS), unsafe_allow_html=True)

# ==============================
# 7) 시각화
# ==============================
st.subheader('분야별 과목 분포')

# (A) 분야별 과목 수 막대 (현재 필터 결과 기준)
counts = []
for f in FIELDS:
    if f in df.columns:
        cnt = df[f].isin(LEVEL_SYMBOLS).sum()
    else:
        cnt = 0
    counts.append({'분야': f, '과목 수': int(cnt)})
bar_df = pd.DataFrame(counts)
fig = px.bar(
    bar_df, x='분야', y='과목 수',
    title='분야별 과목 분포 (현재 필터 적용)',
    labels={'분야':'과목 분야', '과목 수':'개설 과목 수'},
    color='분야', text='과목 수'
)
fig.update_layout(xaxis_tickangle=-45, legend_title_text='')
st.plotly_chart(fig, use_container_width=True)

# (B) Heatmap: 학기×분야 과목 수 (LONG_DF 기준)
if len(l) > 0:
    heat = (l.groupby(['학기라벨','분야'])['교과목 명']
              .nunique().reset_index(name='과목 수'))
    heat['sem_order'] = heat['학기라벨'].map({f"{y}-{s}":i for i,(y,s) in enumerate(SEM_ORDER)}).fillna(9999)
    heat['field_order'] = heat['분야'].apply(lambda x: FIELDS.index(x) if x in FIELDS else 999)
    heat = heat.sort_values(['sem_order','field_order'])

    pivot = heat.pivot(index='분야', columns='학기라벨', values='과목 수').fillna(0)
    fig2 = px.imshow(
        pivot,
        color_continuous_scale='Blues',
        title='Heatmap: 학기 × 분야 (과목 수)',
        labels=dict(x='학기', y='분야', color='과목 수'),
        aspect='auto'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ==============================
# 8) 빈 결과 처리
# ==============================
if df.empty:
    st.warning('선택된 조건에 해당하는 과목이 없습니다. 필터를 완화해 보세요.')


