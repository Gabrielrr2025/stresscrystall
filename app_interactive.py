# app_interactive.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime, math
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

st.set_page_config(page_title="VaR Monte Carlo Avançado", layout="wide")
st.title("📊 VaR Monte Carlo Avançado - Finhealth")

# ================== Inputs ==================
col1, col2 = st.columns(2)
with col1:
    pl = st.number_input("Patrimônio Líquido (R$)", 0.0, value=1_000_000.0, step=1000.0)
    data_ref = st.date_input("Data de Referência", datetime.date.today())
    horizonte_dias = st.selectbox("Horizonte (dias)", [1, 10, 21])
with col2:
    nivel_conf = st.selectbox("Nível de Confiança", ["95%", "99%"])
    alpha = 0.95 if nivel_conf=="95%" else 0.99
    n_sims = st.number_input("Nº simulações Monte Carlo", 100, 200000, value=10000, step=100,
                             help="Define quantos cenários serão simulados. Mais = mais precisão, mas mais lento.")
    seed = st.number_input("Seed (reprodutibilidade)", 0, 1000000, value=42, step=1,
                           help="Número usado para fixar a aleatoriedade. Com a mesma seed, os resultados se repetem.")

# ================== Carteira ==================
st.subheader("📈 Alocação da Carteira")
classes = ["Ações", "Juros", "Dólar", "Crédito Privado", "Imobiliário", "Multimercado", "Commodities", "Outros"]
cols = st.columns(4)
pesos = []
for i, classe in enumerate(classes):
    with cols[i % 4]:
        p = st.slider(f"{classe} %PL", 0, 100, 0)
        pesos.append(p/100)
pesos = np.array(pesos)
total_aloc = pesos.sum()
if total_aloc != 1.0:
    st.warning(f"⚠️ Alocação total {total_aloc*100:.1f}%. Ajuste para 100%.")

# ================== Simulação ==================
if st.button("🚀 Rodar Simulação VaR", type="primary"):
    np.random.seed(seed)
    vols = np.array([0.25,0.08,0.15,0.05,0.06,0.18,0.20,0.10])  # vols anuais default
    vols_d = vols/np.sqrt(252)*math.sqrt(horizonte_dias)
    R = np.random.normal(0,1,(n_sims,len(classes))) * vols_d
    port_ret = R @ pesos
    pnl = pl*port_ret

    # Métricas
    var_val = -np.quantile(pnl, 1-alpha)
    es_val = -pnl[pnl<=-var_val].mean()
    prob_loss = (pnl<0).mean()*100

    col1, col2, col3 = st.columns(3)
    col1.metric(f"VaR {nivel_conf}", f"R$ {var_val:,.2f}",
                help="Perda máxima esperada com confiança escolhida.")
    col2.metric("ES", f"R$ {es_val:,.2f}",
                help="Média das piores perdas além do VaR.")
    col3.metric("Prob. Perda", f"{prob_loss:.2f}%",
                help="Percentual de simulações com resultado negativo.")

    # ================== Gráficos Interativos ==================
    st.subheader("📊 Gráficos Interativos")

    # Histograma
    fig1 = px.histogram(x=pnl, nbins=50, title="Distribuição de P&L",
                        labels={"x":"P&L (R$)"}, opacity=0.7)
    st.plotly_chart(fig1, use_container_width=True)

    # CDF
    sorted_pnl = np.sort(pnl)
    cdf = np.arange(1,len(sorted_pnl)+1)/len(sorted_pnl)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sorted_pnl, y=cdf, mode="lines"))
    fig2.add_vline(x=-var_val, line_dash="dash", line_color="red")
    fig2.update_layout(title="Função de Distribuição Acumulada (CDF)",
                       xaxis_title="P&L (R$)", yaxis_title="Probabilidade Acumulada")
    st.plotly_chart(fig2, use_container_width=True)

    # Boxplot
    fig3 = px.box(pnl, points="all", title="Boxplot dos Retornos Simulados")
    st.plotly_chart(fig3, use_container_width=True)

    # Rolling VaR
    window_size = max(100, n_sims//50)
    rolling_var = pd.Series(pnl).rolling(window_size).quantile(1-alpha).values * -1
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=rolling_var, mode="lines", name="Rolling VaR"))
    fig4.add_hline(y=var_val, line_dash="dash", line_color="red", annotation_text="VaR Total")
    fig4.update_layout(title="Rolling VaR ao longo das simulações",
                       yaxis_title="VaR (R$)")
    st.plotly_chart(fig4, use_container_width=True)

    # Estatísticas
    st.subheader("📈 Estatísticas Detalhadas")
    stats = {
        "Média": pnl.mean(),
        "Mediana": np.median(pnl),
        "Desvio Padrão": pnl.std(),
        "Mínimo": pnl.min(),
        "Máximo": pnl.max(),
        "VaR": var_val,
        "ES": es_val
    }
    st.dataframe(pd.DataFrame(stats,index=["Valor"]).T)

    # Export PDF
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig, ax = plt.subplots()
        ax.hist(pnl, bins=50)
        ax.axvline(-var_val, color="r", linestyle="--")
        ax.set_title("Histograma P&L")
        pdf.savefig(fig)
        plt.close(fig)
    pdf_buffer.seek(0)
    st.download_button("📥 Baixar Relatório PDF", data=pdf_buffer,
                       file_name="var_report.pdf", mime="application/pdf")
