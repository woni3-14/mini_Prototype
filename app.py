import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

st.set_page_config(page_title="Time Step 시계열 분석", layout="wide")
st.title("Time Step 기반 시계열 분석 (라벨 확정 데이터)")

# ============================================================
# 데이터 로드 및 전처리 (캐싱)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("txs_features_labeled.csv")

    ts_group = df.groupby(["Time step", "class"]).size().unstack(fill_value=0)
    ts_group.columns = ["illicit", "licit"]
    ts_group["total"] = ts_group["illicit"] + ts_group["licit"]
    ts_group["illicit_ratio"] = ts_group["illicit"] / ts_group["total"] * 100

    raw_features = ["total_BTC", "fees", "size", "num_input_addresses", "num_output_addresses"]
    ts_illicit = df[df["class"] == 1].groupby("Time step")[raw_features].mean()
    ts_licit = df[df["class"] == 2].groupby("Time step")[raw_features].mean()

    return df, ts_group, ts_illicit, ts_licit

df, ts_group, ts_illicit, ts_licit = load_data()

# ============================================================
# 사이드바 - 그래프 선택
# ============================================================
chart_options = {
    "1. 클래스별 트랜잭션 수": "class_dist",
    "2. Illicit 비율 추이": "illicit_ratio",
    "3. 평균 Total BTC": "total_btc",
    "4. 평균 수수료 (Fees)": "fees",
    "5. 평균 입력 주소 수": "input_addr",
    "6. 평균 출력 주소 수": "output_addr",
}

st.sidebar.header("그래프 선택")
selected = st.sidebar.radio("분석 항목을 선택하세요:", list(chart_options.keys()))
chart_key = chart_options[selected]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**데이터 요약**")
st.sidebar.markdown(f"- 총 트랜잭션: {len(df):,}")
st.sidebar.markdown(f"- Illicit: {(df['class']==1).sum():,}")
st.sidebar.markdown(f"- Licit: {(df['class']==2).sum():,}")
st.sidebar.markdown(f"- Time Step: 1 ~ {int(df['Time step'].max())}")

# ============================================================
# 그래프 렌더링
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

if chart_key == "class_dist":
    ax.bar(ts_group.index, ts_group["licit"], label="Licit (class 2)", color="steelblue", alpha=0.8)
    ax.bar(ts_group.index, ts_group["illicit"], bottom=ts_group["licit"], label="Illicit (class 1)", color="crimson", alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Class Distribution per Time Step")
    ax.legend()

    description = """
    **분석 설명**
    - 각 Time Step별 합법(Licit)과 불법(Illicit) 트랜잭션 수를 스택 바 차트로 나타낸 것입니다.
    - 합법 거래(파란색)가 대부분을 차지하며, 불법 거래(빨간색)는 전체의 약 9.8%에 불과합니다.
    - Time Step 1~10 구간에서 전체 거래량이 많고, 이후 불규칙한 증감을 보입니다.
    - Step 9, 13, 20, 28~29, 32 등에서 불법 거래가 상대적으로 많이 집중되어 있습니다.
    """

elif chart_key == "illicit_ratio":
    ax.plot(ts_group.index, ts_group["illicit_ratio"], color="crimson", marker="o", markersize=5, linewidth=1.5)
    mean_val = ts_group["illicit_ratio"].mean()
    ax.axhline(mean_val, color="gray", linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.1f}%")
    ax.fill_between(ts_group.index, ts_group["illicit_ratio"], alpha=0.15, color="crimson")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Illicit Ratio (%)")
    ax.set_title("Illicit Transaction Ratio Over Time")
    ax.legend()

    description = """
    **분석 설명**
    - 불법 거래 비율은 시간에 따라 극심한 변동을 보입니다 (0.3% ~ 36%).
    - 평균 비율은 약 11.4%이지만, Step 9(31.9%), 13(36.0%), 20(28.9%), 28(29.9%) 등 특정 구간에서 급등합니다.
    - Step 43~46 구간에서는 거의 0%에 가까워지며, 불법 활동이 '버스트(burst)' 패턴으로 특정 시점에 집중됨을 시사합니다.
    - 이는 단속 회피를 위한 간헐적 활동 또는 특정 사건(다크웹 마켓 운영 등)과의 연관 가능성을 보여줍니다.
    """

elif chart_key == "total_btc":
    ax.plot(ts_illicit.index, ts_illicit["total_BTC"], label="Illicit", color="crimson", marker="s", markersize=4)
    ax.plot(ts_licit.index, ts_licit["total_BTC"], label="Licit", color="steelblue", marker="o", markersize=4)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Total BTC")
    ax.set_title("Average Total BTC per Transaction")
    ax.legend()

    description = """
    **분석 설명**
    - 합법 거래의 평균 BTC 규모(파란선)는 10~80 BTC로 큰 변동을 보이며, 거래소나 결제 서비스의 대량 거래가 포함되어 있습니다.
    - 불법 거래(빨간선)는 거의 항상 1~10 BTC 수준을 유지하며 소액으로 일관됩니다.
    - 이는 불법 거래가 추적 회피를 위해 의도적으로 금액을 분산(splitting)하는 전형적인 자금세탁 패턴과 일치합니다.
    - 두 클래스 간 BTC 규모 차이는 분류 모델에서 강력한 판별 피처로 활용될 수 있습니다.
    """

elif chart_key == "fees":
    ax.plot(ts_illicit.index, ts_illicit["fees"], label="Illicit", color="crimson", marker="s", markersize=4)
    ax.plot(ts_licit.index, ts_licit["fees"], label="Licit", color="steelblue", marker="o", markersize=4)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Fees (BTC)")
    ax.set_title("Average Transaction Fees")
    ax.legend()

    description = """
    **분석 설명**
    - 대부분의 구간에서 양 클래스 모두 수수료가 매우 낮은 수준(0.001 BTC 이하)입니다.
    - 그러나 후반부(Step 48 부근)에서 불법 거래의 수수료가 급등하는 이상 스파이크가 관측됩니다.
    - 이는 해당 시점에서 긴급하게 처리해야 하는 트랜잭션(빠른 확인을 위한 높은 수수료)이 존재했음을 의미합니다.
    - 수수료 스파이크는 자금 도피나 긴급 이체 상황과 관련될 수 있습니다.
    """

elif chart_key == "input_addr":
    ax.plot(ts_illicit.index, ts_illicit["num_input_addresses"], label="Illicit", color="crimson", marker="s", markersize=4)
    ax.plot(ts_licit.index, ts_licit["num_input_addresses"], label="Licit", color="steelblue", marker="o", markersize=4)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Input Addresses")
    ax.set_title("Average Number of Input Addresses")
    ax.legend()

    description = """
    **분석 설명**
    - 합법 거래는 평균 3~10개의 입력 주소를 사용하며 비교적 안정적입니다.
    - 불법 거래는 대부분 1~5개로 낮지만, Step 48 부근에서 평균 250개까지 급등하는 극단적 이상치가 있습니다.
    - 이 스파이크는 소수의 트랜잭션이 수백 개의 입력을 집계한 것으로, 믹싱(mixing) 또는 텀블링(tumbling) 서비스의 특징입니다.
    - 믹싱 서비스는 다수의 소액 입력을 하나로 합쳐 자금 출처를 은폐하는 기법입니다.
    """

elif chart_key == "output_addr":
    ax.plot(ts_illicit.index, ts_illicit["num_output_addresses"], label="Illicit", color="crimson", marker="s", markersize=4)
    ax.plot(ts_licit.index, ts_licit["num_output_addresses"], label="Licit", color="steelblue", marker="o", markersize=4)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Output Addresses")
    ax.set_title("Average Number of Output Addresses")
    ax.legend()

    description = """
    **분석 설명**
    - 합법 거래의 출력 주소 수는 5~30개로 변동이 크며, 이는 거래소의 일괄 출금이나 다수 수신자 결제 특성을 반영합니다.
    - 불법 거래는 1~3개로 일관되게 낮습니다. 이는 특정 목적지로만 자금을 이동시키는 단순한 구조를 의미합니다.
    - 출력 주소 수가 낮다는 것은 자금의 최종 수취자가 소수라는 뜻으로, P2P 방식의 직접적인 불법 자금 이동 패턴과 일치합니다.
    """

ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 설명 표시
st.markdown(description)

# 하단 데이터 테이블 (접기)
with st.expander("Time Step별 상세 데이터 보기"):
    st.dataframe(ts_group[["illicit", "licit", "total", "illicit_ratio"]].reset_index(), use_container_width=True)
