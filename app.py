import math
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ['ticker', 'qty', 'avg_cost']:
        if col not in df.columns:
            df[col] = '' if col == 'ticker' else 0.0

    df = df[['ticker', 'qty', 'avg_cost']]

    df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(0.0)
    df['avg_cost'] = pd.to_numeric(df['avg_cost'], errors='coerce').fillna(0.0)

    return df


def default_holdings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {'ticker': 'NVDA', 'qty': 10.0, 'avg_cost': 650.25},
            {'ticker': 'AAPL', 'qty': 25.5, 'avg_cost': 175.12},
            {'ticker': 'BTC-USD', 'qty': 0.25, 'avg_cost': 45000.00},
        ]
    )


@st.cache_data(ttl=90)
def fetch_live_prices(tickers: List[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    try:
        import yfinance as yf
        for t in tickers:
            try:
                data = yf.download(
                    t,
                    period='2d',
                    interval='1m',
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
                if 'Close' in data and len(data['Close'].dropna()):
                    prices[t] = float(data['Close'].dropna().iloc[-1])
                else:
                    prices[t] = math.nan
            except Exception:
                prices[t] = math.nan
    except Exception:
        for t in tickers:
            prices[t] = math.nan

    return prices


def build_positions(df_input: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df_input)

    df_calc = df[df['ticker'].str.len() > 0].copy()
    if df_calc.empty:
        return pd.DataFrame(columns=['ticker', 'qty', 'avg_cost', 'cost_basis'])

    df_calc['cost_basis'] = df_calc['qty'] * df_calc['avg_cost']

    pos = (
        df_calc
        .groupby('ticker', as_index=False)
        .agg(
            qty=('qty', 'sum'),
            cost_basis=('cost_basis', 'sum'),
        )
    )

    pos['avg_cost'] = pos.apply(
        lambda r: (r['cost_basis'] / r['qty']) if r['qty'] != 0 else 0.0,
        axis=1,
    )

    pos = pos[['ticker', 'qty', 'avg_cost', 'cost_basis']].sort_values('ticker').reset_index(drop=True)
    return pos


st.set_page_config(page_title='Portfolio Dashboard', layout='wide')
st.title('Portfolio Dashboard')
st.caption('Edit holdings live. Fractional shares supported. Scenario moves up to 400 percent.')

if 'holdings_input' not in st.session_state:
    st.session_state.holdings_input = ensure_columns(default_holdings())

top_a, top_b, top_c, top_d = st.columns([1, 1, 1, 2])

with top_a:
    if st.button('Add row'):
        st.session_state.holdings_input = pd.concat(
            [st.session_state.holdings_input, pd.DataFrame([{'ticker': '', 'qty': 0.0, 'avg_cost': 0.0}])],
            ignore_index=True,
        )

with top_b:
    if st.button('Reset sample'):
        st.session_state.holdings_input = ensure_columns(default_holdings())

with top_c:
    uploaded = st.file_uploader('Import holdings csv', type=['csv'], label_visibility='collapsed')
    if uploaded is not None:
        try:
            imported = pd.read_csv(uploaded)
            st.session_state.holdings_input = ensure_columns(imported)
            st.success('Imported holdings')
        except Exception:
            st.error('Import failed. CSV needs columns: ticker, qty, avg_cost')

with top_d:
    export_csv = st.session_state.holdings_input.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download holdings as CSV',
        data=export_csv,
        file_name='holdings.csv',
        mime='text/csv',
    )

st.subheader('Holdings input')
edited = st.data_editor(
    st.session_state.holdings_input,
    key='holdings_editor',
    use_container_width=True,
    num_rows='dynamic',
    column_config={
        'ticker': st.column_config.TextColumn('Ticker'),
        'qty': st.column_config.NumberColumn('Qty', step=0.0001, format='%.8f'),
        'avg_cost': st.column_config.NumberColumn('Avg cost', step=0.01, format='%.6f'),
    },
)

st.session_state.holdings_input = ensure_columns(edited)

positions = build_positions(st.session_state.holdings_input)
tickers = positions['ticker'].tolist()

st.divider()

left, right = st.columns([1.15, 1])

with right:
    st.subheader('Prices')
    use_live = st.toggle('Try live prices', value=True)

    refresh = st.button('Refresh live prices')
    if refresh:
        st.cache_data.clear()

    live_prices: Optional[Dict[str, float]] = None
    if use_live and tickers:
        live_prices = fetch_live_prices(tickers)

    manual_prices: Dict[str, float] = {}
    for t in tickers:
        default_price = 0.0
        if isinstance(live_prices, dict) and t in live_prices and pd.notna(live_prices[t]):
            default_price = float(live_prices[t])

        manual_prices[t] = st.number_input(
            f'Price for {t}',
            min_value=0.0,
            value=float(default_price),
            step=0.01,
        )

with left:
    st.subheader('Scenario')
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        scenario_scope = st.selectbox('Scope', ['All positions', 'Single ticker'])

    with c2:
        target = st.selectbox('Ticker', tickers) if scenario_scope == 'Single ticker' and tickers else None

    with c3:
        coarse = st.slider('Quick move percent', min_value=-80.0, max_value=400.0, value=0.0, step=0.5)
        pct_move = st.number_input('Exact move percent', min_value=-80.0, max_value=400.0, value=float(coarse), step=0.01)

if positions.empty:
    st.info('Add at least one ticker row to see calculations')
    st.stop()

df = positions.copy()
df['price_now'] = df['ticker'].map(manual_prices).astype(float)

def scenario_price(row: pd.Series) -> float:
    base = float(row['price_now'])
    if scenario_scope == 'All positions':
        return base * (1 + pct_move / 100.0)
    if scenario_scope == 'Single ticker' and target and row['ticker'] == target:
        return base * (1 + pct_move / 100.0)
    return base

df['price_scn'] = df.apply(scenario_price, axis=1)

df['value_now'] = df['qty'] * df['price_now']
df['value_scn'] = df['qty'] * df['price_scn']
df['pnl_now'] = df['value_now'] - df['cost_basis']
df['pnl_scn'] = df['value_scn'] - df['cost_basis']
df['pnl_change'] = df['pnl_scn'] - df['pnl_now']

total_now = float(df['value_now'].sum())
total_scn = float(df['value_scn'].sum())
pnl_now = float(df['pnl_now'].sum())
pnl_scn = float(df['pnl_scn'].sum())
pnl_change = float(df['pnl_change'].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric('Total value now', f'${total_now:,.2f}')
k2.metric('Total value scenario', f'${total_scn:,.2f}', delta=f'${(total_scn - total_now):,.2f}')
k3.metric('Unrealized P L now', f'${pnl_now:,.2f}')
k4.metric('Unrealized P L scenario', f'${pnl_scn:,.2f}', delta=f'${pnl_change:,.2f}')

st.divider()

try:
    import plotly.express as px

    p1, p2 = st.columns(2)

    with p1:
        st.subheader('Allocation now')
        alloc_now = df[df['value_now'] > 0].copy()
        fig_now = px.pie(alloc_now, names='ticker', values='value_now', hole=0.45)
        st.plotly_chart(fig_now, use_container_width=True)

    with p2:
        st.subheader('Allocation scenario')
        alloc_scn = df[df['value_scn'] > 0].copy()
        fig_scn = px.pie(alloc_scn, names='ticker', values='value_scn', hole=0.45)
        st.plotly_chart(fig_scn, use_container_width=True)

    st.subheader('Scenario P L impact by ticker')
    bar_df = df.copy().sort_values('pnl_change')
    fig_bar = px.bar(bar_df, x='ticker', y='pnl_change')
    st.plotly_chart(fig_bar, use_container_width=True)

except Exception:
    st.warning('Plotly missing. Confirm plotly is in requirements.txt')

st.subheader('Positions summary')
show_cols = [
    'ticker', 'qty', 'avg_cost',
    'price_now', 'price_scn',
    'cost_basis', 'value_now', 'value_scn',
    'pnl_now', 'pnl_scn', 'pnl_change',
]
st.dataframe(df[show_cols], use_container_width=True)

st.caption('Edits in the holdings table trigger recalculation automatically.')
