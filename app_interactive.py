# app_professional.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import math
import json
from scipy import stats
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="VaR Monte Carlo Professional", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para aparência profissional
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header principal
st.markdown('<p class="main-header">📊 VaR Monte Carlo Professional</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema Avançado de Gestão de Risco</p>', unsafe_allow_html=True)

# 🏢 Dados do Fundo
st.subheader("🏢 Dados do Fundo")

col1, col2, col3 = st.columns(3)
with col1:
    nome_projeto = st.text_input("Nome do Projeto", value="Análise de Risco - Portfolio")
with col2:
    responsavel = st.text_input("Responsável pela Análise", value="")
with col3:
    cnpj = st.text_input("CNPJ", value="", placeholder="00.000.000/0000-00")

col1, col2, col3 = st.columns(3)
with col1:
    data_ref = st.date_input("Data de Referência", datetime.date.today())
with col2:
    nome_fundo = st.text_input("Nome do Fundo", value="Fundo Exemplo")
with col3:
    pl = st.number_input(
        "Patrimônio Líquido (R$)", 
        min_value=0.0, 
        value=10_000_000.0, 
        step=100_000.0, 
        format="%.2f"
    )

# SIDEBAR COM PARÂMETROS
with st.sidebar:
    st.header("⚙️ Parâmetros")

    horizonte_dias = st.selectbox(
        "Horizonte Temporal (dias úteis)", 
        options=[1, 5, 10, 15, 21, 42, 63, 126, 252, 504],
        format_func=lambda x: f"{x} dias úteis ({x/21:.1f} meses)" if x > 21 else f"{x} dias úteis",
        index=3
    )

    nivel_conf = st.selectbox(
        "Nível de Confiança", 
        ["90%", "95%", "97.5%", "99%", "99.5%"],
        index=1
    )

    n_sims = st.selectbox(
        "Número de Simulações",
        options=[10_000, 50_000, 100_000, 250_000, 500_000],
        format_func=lambda x: f"{x:,} simulações",
        index=1
    )

    seed = st.number_input(
        "Seed (reprodutibilidade)", 
        min_value=0, 
        max_value=1000000, 
        value=42, 
        step=1
    )

# ALOCAÇÃO POR CLASSE
st.subheader("📊 Alocação por Classe")

st.write("### Ativos")
col1, col2, col3, col4 = st.columns(4)
with col1:
    acoes = st.slider("Ações", 0, 100, 25)
with col2:
    juros = st.slider("Renda Fixa", 0, 100, 20)
with col3:
    credito_privado = st.slider("Crédito Privado", 0, 100, 15)
with col4:
    dolar = st.slider("Moeda Estrangeira", 0, 100, 10)

col1, col2, col3 = st.columns(3)
with col1:
    imobiliario = st.slider("Imobiliário", 0, 100, 15)
with col2:
    commodities = st.slider("Commodities", 0, 100, 5)
with col3:
    alternativos = st.slider("Outros", 0, 100, 5)

total_aloc = acoes + juros + credito_privado + dolar + imobiliario + commodities + alternativos

# Validação visual da alocação
if total_aloc > 100:
    st.error(f"⚠️ Alocação total: {total_aloc}% excede 100%!")
elif total_aloc == 100:
    st.success(f"✅ Carteira totalmente alocada: {total_aloc}%")
else:
    st.info(f"💰 Alocação: {total_aloc}% | Caixa: {100-total_aloc}%")

# COMPOSIÇÃO DA CARTEIRA
st.subheader("📈 Composição da Carteira")

fig_aloc = go.Figure(data=[go.Pie(
    labels=['Ações', 'Renda Fixa', 'Crédito Privado', 'Moeda Estrangeira', 'Imobiliário', 'Commodities', 'Outros', 'Caixa'],
    values=[acoes, juros, credito_privado, dolar, imobiliario, commodities, alternativos, max(0, 100-total_aloc)],
    hole=.3,
    marker_colors=['#FF6B6B', '#4ECDC4', '#95E1D3', '#45B7D1', '#AA96DA', '#FFA07A', '#FCBAD3', '#98D8C8'],
    textposition='auto',
    textinfo='label+percent'
)])
fig_aloc.update_layout(
    title="Distribuição da Carteira por Classe",
    height=350,
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_aloc, use_container_width=True)


pesos = np.array([acoes, juros, credito_privado, dolar, imobiliario, commodities, alternativos])/100

# CONFIGURAÇÕES AVANÇADAS
st.subheader("🔧 Configurações Avançadas do Modelo")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Volatilidades", 
    "🔗 Correlações", 
    "📈 Distribuições", 
    "🎯 Cenários Stress",
    "📉 Backtesting"
])

with tab1:
    st.write("### Parâmetros de Volatilidade")
    
    st.write("#### Ativos Tradicionais")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        vol_acoes = st.number_input("Vol. Ações (%a.a.)", 5.0, 100.0, 25.0, 0.5)
    with col2:
        vol_juros = st.number_input("Vol. Renda Fixa (%a.a.)", 1.0, 50.0, 6.0, 0.5)
    with col3:
        vol_credito = st.number_input("Vol. Crédito Privado (%a.a.)", 3.0, 50.0, 12.0, 0.5)
    with col4:
        vol_dolar = st.number_input("Vol. Moeda (%a.a.)", 5.0, 50.0, 15.0, 0.5)
    
    st.write("#### Ativos Alternativos")
    col1, col2, col3 = st.columns(3)
    with col1:
        vol_imobiliario = st.number_input("Vol. Imobiliário (%a.a.)", 5.0, 50.0, 18.0, 0.5)
    with col2:
        vol_commodities = st.number_input("Vol. Commodities (%a.a.)", 10.0, 100.0, 30.0, 0.5)
    with col3:
        vol_alternativos = st.number_input("Vol. Alternativos (%a.a.)", 10.0, 150.0, 35.0, 0.5)
    
    vols = np.array([vol_acoes, vol_juros, vol_credito, vol_dolar, vol_imobiliario, vol_commodities, vol_alternativos]) / 100
    
    # Mostrar volatilidades ajustadas
    vols_horizonte = vols * np.sqrt(horizonte_dias/252)
    
    # Gráfico de barras das volatilidades
    fig_vol = go.Figure(data=[
        go.Bar(name='Anualizada', 
               x=['Ações', 'RF', 'Créd.Priv', 'Moeda', 'Imob.', 'Commod.', 'Altern.'],
               y=vols*100,
               marker_color='lightblue'),
        go.Bar(name=f'{horizonte_dias} dias',
               x=['Ações', 'RF', 'Créd.Priv', 'Moeda', 'Imob.', 'Commod.', 'Altern.'],
               y=vols_horizonte*100,
               marker_color='darkblue')
    ])
    fig_vol.update_layout(
        title=f"Volatilidades: Anualizada vs {horizonte_dias} dias",
        yaxis_title="Volatilidade (%)",
        barmode='group',
        height=350
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with tab2:
    st.write("### Matriz de Correlação entre Ativos")
    
    usar_correlacao = st.checkbox("Ativar correlações entre ativos", value=True)
    
    if usar_correlacao:
        # Templates pré-definidos para 7 ativos
        template_corr = st.selectbox(
            "Selecionar cenário de correlação:",
            ["Personalizado", "Mercado Normal", "Crise (Flight to Quality)", 
             "Risk-On", "Stress Sistêmico", "Decorrelação"]
        )
        
        # Matriz 7x7 com valores realistas
        if template_corr == "Mercado Normal":
            corr_values = {
                'acoes_rf': -0.15, 'acoes_credito': 0.40, 'acoes_dolar': 0.25, 
                'acoes_imob': 0.60, 'acoes_comm': 0.35, 'acoes_alt': 0.45,
                'rf_credito': 0.60, 'rf_dolar': -0.30, 'rf_imob': 0.20, 
                'rf_comm': -0.10, 'rf_alt': 0.05,
                'credito_dolar': 0.15, 'credito_imob': 0.50, 'credito_comm': 0.20, 
                'credito_alt': 0.35,
                'dolar_imob': 0.10, 'dolar_comm': 0.45, 'dolar_alt': 0.30,
                'imob_comm': 0.25, 'imob_alt': 0.40,
                'comm_alt': 0.50
            }
        elif template_corr == "Crise (Flight to Quality)":
            corr_values = {
                'acoes_rf': 0.30, 'acoes_credito': 0.70, 'acoes_dolar': 0.60, 
                'acoes_imob': 0.80, 'acoes_comm': 0.65, 'acoes_alt': 0.75,
                'rf_credito': -0.20, 'rf_dolar': -0.40, 'rf_imob': -0.10, 
                'rf_comm': -0.30, 'rf_alt': -0.15,
                'credito_dolar': 0.40, 'credito_imob': 0.65, 'credito_comm': 0.50, 
                'credito_alt': 0.60,
                'dolar_imob': 0.45, 'dolar_comm': 0.35, 'dolar_alt': 0.40,
                'imob_comm': 0.55, 'imob_alt': 0.65,
                'comm_alt': 0.60
            }
        else:
            corr_values = {
                'acoes_rf': 0.0, 'acoes_credito': 0.0, 'acoes_dolar': 0.0, 
                'acoes_imob': 0.0, 'acoes_comm': 0.0, 'acoes_alt': 0.0,
                'rf_credito': 0.0, 'rf_dolar': 0.0, 'rf_imob': 0.0, 
                'rf_comm': 0.0, 'rf_alt': 0.0,
                'credito_dolar': 0.0, 'credito_imob': 0.0, 'credito_comm': 0.0, 
                'credito_alt': 0.0,
                'dolar_imob': 0.0, 'dolar_comm': 0.0, 'dolar_alt': 0.0,
                'imob_comm': 0.0, 'imob_alt': 0.0,
                'comm_alt': 0.0
            }
        
        # Interface para editar correlações principais
        st.write("#### Correlações Principais (ajuste fino)")
        st.write("*As correlações mais importantes para o portfolio*")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Ações com outros ativos:**")
            corr_values['acoes_rf'] = st.slider("Ações × RF", -1.0, 1.0, corr_values['acoes_rf'], 0.05, key="ac_rf")
            corr_values['acoes_credito'] = st.slider("Ações × Crédito", -1.0, 1.0, corr_values['acoes_credito'], 0.05, key="ac_cr")
            corr_values['acoes_imob'] = st.slider("Ações × Imobiliário", -1.0, 1.0, corr_values['acoes_imob'], 0.05, key="ac_im")
        
        with col2:
            st.write("**Renda Fixa com outros:**")
            corr_values['rf_credito'] = st.slider("RF × Crédito", -1.0, 1.0, corr_values['rf_credito'], 0.05, key="rf_cr")
            corr_values['rf_dolar'] = st.slider("RF × Moeda", -1.0, 1.0, corr_values['rf_dolar'], 0.05, key="rf_dl")
            corr_values['rf_imob'] = st.slider("RF × Imobiliário", -1.0, 1.0, corr_values['rf_imob'], 0.05, key="rf_im")
        
        with col3:
            st.write("**Alternativos com outros:**")
            corr_values['acoes_alt'] = st.slider("Ações × Alternativos", -1.0, 1.0, corr_values['acoes_alt'], 0.05, key="ac_al")
            corr_values['imob_alt'] = st.slider("Imobiliário × Alternativos", -1.0, 1.0, corr_values['imob_alt'], 0.05, key="im_al")
            corr_values['comm_alt'] = st.slider("Commodities × Alternativos", -1.0, 1.0, corr_values['comm_alt'], 0.05, key="cm_al")
        
        # Construir matriz 7x7 completa
        corr_matrix = np.eye(7)
        
        # Preencher a matriz simétrica
        corr_matrix[0, 1] = corr_matrix[1, 0] = corr_values['acoes_rf']
        corr_matrix[0, 2] = corr_matrix[2, 0] = corr_values['acoes_credito']
        corr_matrix[0, 3] = corr_matrix[3, 0] = corr_values.get('acoes_dolar', 0.25)
        corr_matrix[0, 4] = corr_matrix[4, 0] = corr_values['acoes_imob']
        corr_matrix[0, 5] = corr_matrix[5, 0] = corr_values.get('acoes_comm', 0.35)
        corr_matrix[0, 6] = corr_matrix[6, 0] = corr_values['acoes_alt']
        
        corr_matrix[1, 2] = corr_matrix[2, 1] = corr_values['rf_credito']
        corr_matrix[1, 3] = corr_matrix[3, 1] = corr_values['rf_dolar']
        corr_matrix[1, 4] = corr_matrix[4, 1] = corr_values['rf_imob']
        corr_matrix[1, 5] = corr_matrix[5, 1] = corr_values.get('rf_comm', -0.10)
        corr_matrix[1, 6] = corr_matrix[6, 1] = corr_values.get('rf_alt', 0.05)
        
        corr_matrix[2, 3] = corr_matrix[3, 2] = corr_values.get('credito_dolar', 0.15)
        corr_matrix[2, 4] = corr_matrix[4, 2] = corr_values.get('credito_imob', 0.50)
        corr_matrix[2, 5] = corr_matrix[5, 2] = corr_values.get('credito_comm', 0.20)
        corr_matrix[2, 6] = corr_matrix[6, 2] = corr_values.get('credito_alt', 0.35)
        
        corr_matrix[3, 4] = corr_matrix[4, 3] = corr_values.get('dolar_imob', 0.10)
        corr_matrix[3, 5] = corr_matrix[5, 3] = corr_values.get('dolar_comm', 0.45)
        corr_matrix[3, 6] = corr_matrix[6, 3] = corr_values.get('dolar_alt', 0.30)
        
        corr_matrix[4, 5] = corr_matrix[5, 4] = corr_values.get('imob_comm', 0.25)
        corr_matrix[4, 6] = corr_matrix[6, 4] = corr_values['imob_alt']
        
        corr_matrix[5, 6] = corr_matrix[6, 5] = corr_values['comm_alt']
        
        # Verificar se é positiva definida
        eigenvalues = np.linalg.eigvals(corr_matrix)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue <= 0:
            st.warning(f"⚠️ Ajustando matriz para ser positiva definida...")
            eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
            eigenvalues[eigenvalues < 0.01] = 0.01
            corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            D = np.diag(1.0/np.sqrt(np.diag(corr_matrix)))
            corr_matrix = D @ corr_matrix @ D
            st.success("✅ Matriz ajustada com sucesso")
        
        # Heatmap interativo
        labels = ['Ações', 'RF', 'Créd.Priv', 'Moeda', 'Imob.', 'Commod.', 'Altern.']
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlação"),
            hovertemplate='%{y} × %{x}: %{z:.3f}<extra></extra>'
        ))
        fig_corr.update_layout(
            title="Matriz de Correlação 7×7",
            height=500,
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        corr_matrix = np.eye(7)
        st.info("📊 Ativos configurados como independentes")

with tab3:
    st.write("### Distribuições de Probabilidade")
    
    st.write("#### Ativos Tradicionais")
    col1, col2 = st.columns(2)
    with col1:
        dist_acoes = st.selectbox(
            "Distribuição - Ações",
            ["Normal", "t-Student", "Lognormal", "Normal Mixture"]
        )
        if dist_acoes == "t-Student":
            df_acoes = st.slider("Graus de liberdade (Ações)", 3, 30, 5)
        elif dist_acoes == "Normal Mixture":
            mix_prob = st.slider("Probabilidade regime volátil (%)", 5, 30, 10) / 100
            mix_scale = st.slider("Multiplicador de volatilidade", 2.0, 5.0, 3.0)
        
        dist_juros = st.selectbox(
            "Distribuição - Renda Fixa",
            ["Normal", "t-Student", "Lognormal"]
        )
        if dist_juros == "t-Student":
            df_juros = st.slider("Graus de liberdade (RF)", 3, 30, 10)
        
        dist_credito = st.selectbox(
            "Distribuição - Crédito Privado",
            ["Normal", "t-Student", "Lognormal"],
            index=1
        )
        if dist_credito == "t-Student":
            df_credito = st.slider("Graus de liberdade (Crédito)", 3, 30, 7)
    
    with col2:
        dist_dolar = st.selectbox(
            "Distribuição - Moeda",
            ["Normal", "t-Student", "Lognormal"],
            index=1
        )
        if dist_dolar == "t-Student":
            df_dolar = st.slider("Graus de liberdade (Moeda)", 3, 30, 7)
        
        dist_imobiliario = st.selectbox(
            "Distribuição - Imobiliário",
            ["Normal", "t-Student", "Lognormal"]
        )
        if dist_imobiliario == "t-Student":
            df_imobiliario = st.slider("Graus de liberdade (Imob.)", 3, 30, 8)
            
    st.write("#### Ativos Alternativos")
    col1, col2 = st.columns(2)
    with col1:
        dist_commodities = st.selectbox(
            "Distribuição - Commodities",
            ["Normal", "t-Student", "Lognormal"],
            index=1
        )
        if dist_commodities == "t-Student":
            df_commodities = st.slider("Graus de liberdade (Commod.)", 3, 30, 5)
    
    with col2:
        dist_alternativos = st.selectbox(
            "Distribuição - Alternativos",
            ["Normal", "t-Student", "Lognormal", "Normal Mixture"],
            index=1
        )
        if dist_alternativos == "t-Student":
            df_alternativos = st.slider("Graus de liberdade (Altern.)", 3, 30, 4)

with tab4:
    st.write("### Cenários de Stress Determinísticos")
    
    usar_cenarios = st.checkbox("Incluir cenários de stress históricos", value=True)
    
    if usar_cenarios:
        # Inicializar cenários para 7 ativos
        if 'cenarios' not in st.session_state:
            st.session_state.cenarios = pd.DataFrame({
                'Nome': ['Crise 2008', 'COVID-19', 'Taper Tantrum', 'Brexit', 'Guerra Comercial'],
                'Ações (%)': [-38.5, -33.9, -5.8, -8.5, -15.0],
                'RF (%)': [8.0, -2.0, 12.0, 3.5, 5.0],
                'Crédito (%)': [-25.0, -18.0, 8.0, 2.0, -5.0],
                'Moeda (%)': [25.0, 18.0, 15.0, 12.0, 10.0],
                'Imobiliário (%)': [-35.0, -15.0, -3.0, -5.0, -8.0],
                'Commodities (%)': [-45.0, -30.0, -10.0, -5.0, -20.0],
                'Alternativos (%)': [-40.0, -25.0, -8.0, -10.0, -18.0],
                'Probabilidade (%)': [2.0, 2.0, 5.0, 3.0, 4.0]
            })
        
        # Editor de cenários
        st.write("*Edite os cenários de stress ou adicione novos:*")
        edited_df = st.data_editor(
            st.session_state.cenarios,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Nome": st.column_config.TextColumn("Cenário", width="small"),
                "Ações (%)": st.column_config.NumberColumn("Ações", format="%.1f", width="small"),
                "RF (%)": st.column_config.NumberColumn("RF", format="%.1f", width="small"),
                "Crédito (%)": st.column_config.NumberColumn("Créd.", format="%.1f", width="small"),
                "Moeda (%)": st.column_config.NumberColumn("Moeda", format="%.1f", width="small"),
                "Imobiliário (%)": st.column_config.NumberColumn("Imob.", format="%.1f", width="small"),
                "Commodities (%)": st.column_config.NumberColumn("Comm.", format="%.1f", width="small"),
                "Alternativos (%)": st.column_config.NumberColumn("Alt.", format="%.1f", width="small"),
                "Probabilidade (%)": st.column_config.NumberColumn("Prob", format="%.1f", width="small")
            },
            hide_index=True
        )
        st.session_state.cenarios = edited_df
        
        if len(st.session_state.cenarios) > 0:
            col1, col2 = st.columns(2)
            with col1:
                pct_stress = st.slider("Percentual de cenários de stress", 5, 30, 10)
            with col2:
                total_prob = st.session_state.cenarios['Probabilidade (%)'].sum()
                st.info(f"📊 Probabilidade total: {total_prob:.1f}%")

with tab5:
    st.write("### Configurações de Backtesting")
    
    realizar_backtest = st.checkbox("Realizar backtesting histórico", value=False)
    if realizar_backtest:
        col1, col2 = st.columns(2)
        with col1:
            periodo_backtest = st.selectbox(
                "Período de backtesting",
                ["1 ano", "2 anos", "3 anos", "5 anos"]
            )
        with col2:
            metodo_backtest = st.selectbox(
                "Método de backtesting",
                ["Kupiec (POF)", "Christoffersen", "Ambos"]
            )

# BOTÃO DE SIMULAÇÃO
st.write("---")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    run_simulation = st.button("🚀 Executar Simulação", 
                              type="primary", 
                              use_container_width=True)
with col2:
    generate_report = st.checkbox("📄 Gerar Relatório PDF", value=True)
with col3:
    export_data = st.checkbox("💾 Exportar Dados", value=True)

# EXECUÇÃO DA SIMULAÇÃO
if run_simulation:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Inicializando simulação..."):
        np.random.seed(seed)
        
        # Preparar parâmetros
        vols_d = vols * np.sqrt(horizonte_dias/252)
        
        # Listas para resultados
        all_returns = []
        all_labels = []
        
        # Determinar número de simulações
        if usar_cenarios and len(st.session_state.cenarios) > 0:
            n_stress_total = int(n_sims * pct_stress / 100)
            n_mc = n_sims - n_stress_total
        else:
            n_mc = n_sims
            n_stress_total = 0
        
        progress_bar.progress(10)
        status_text.text("Gerando cenários Monte Carlo...")
        
        # SIMULAÇÃO MONTE CARLO
        if n_mc > 0:
            # Gerar correlações
            if usar_correlacao:
                L = np.linalg.cholesky(corr_matrix)
                Z = np.random.normal(size=(n_mc, 7))
                Z_corr = Z @ L.T
            else:
                Z_corr = np.random.normal(size=(n_mc, 7))
            
            R_mc = np.zeros((n_mc, 7))
            
            # Aplicar distribuições
            distributions = [dist_acoes, dist_juros, dist_credito, dist_dolar, 
                           dist_imobiliario, dist_commodities, dist_alternativos]
            
            for j, (dist, vol) in enumerate(zip(distributions, vols_d)):
                if dist == "Normal":
                    R_mc[:, j] = vol * Z_corr[:, j]
                
                elif dist == "t-Student":
                    df_values = [
                        df_acoes if 'df_acoes' in locals() else 5,
                        df_juros if 'df_juros' in locals() else 10,
                        df_credito if 'df_credito' in locals() else 7,
                        df_dolar if 'df_dolar' in locals() else 7,
                        df_imobiliario if 'df_imobiliario' in locals() else 8,
                        df_commodities if 'df_commodities' in locals() else 5,
                        df_alternativos if 'df_alternativos' in locals() else 4
                    ]
                    df = df_values[j]
                    t_samples = stats.t.ppf(stats.norm.cdf(Z_corr[:, j]), df)
                    scale_factor = np.sqrt(df / (df - 2)) if df > 2 else 1
                    R_mc[:, j] = vol * t_samples / scale_factor
                
                elif dist == "Lognormal":
                    mu_log = -0.5 * vol**2
                    R_mc[:, j] = np.exp(mu_log + vol * Z_corr[:, j]) - 1
                
                elif dist == "Normal Mixture":
                    if 'mix_prob' in locals():
                        is_volatile = np.random.rand(n_mc) < mix_prob
                        normal_part = vol * Z_corr[:, j]
                        volatile_part = vol * mix_scale * Z_corr[:, j]
                        R_mc[:, j] = np.where(is_volatile, volatile_part, normal_part)
                    else:
                        R_mc[:, j] = vol * Z_corr[:, j]
            
            all_returns.append(R_mc)
            all_labels.extend(['Monte Carlo'] * n_mc)
        
        progress_bar.progress(50)
        status_text.text("Aplicando cenários de stress...")
        
        # CENÁRIOS DE STRESS
        if usar_cenarios and n_stress_total > 0:
            cenarios_df = st.session_state.cenarios
            probs = cenarios_df['Probabilidade (%)'].values
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)
            
            indices = np.random.choice(len(cenarios_df), size=n_stress_total, p=probs)
            
            R_stress = []
            for idx in indices:
                row = cenarios_df.iloc[idx]
                R_stress.append([
                    row['Ações (%)'] / 100,
                    row['RF (%)'] / 100,
                    row['Crédito (%)'] / 100,
                    row['Moeda (%)'] / 100,
                    row['Imobiliário (%)'] / 100,
                    row['Commodities (%)'] / 100,
                    row['Alternativos (%)'] / 100
                ])
                all_labels.append(row['Nome'])
            
            R_stress = np.array(R_stress)
            all_returns.append(R_stress)
        
        progress_bar.progress(70)
        status_text.text("Calculando métricas de risco...")
        
        # Combinar retornos
        R_total = np.vstack(all_returns) if len(all_returns) > 0 else np.zeros((n_sims, 7))
        
        # Calcular P&L
        port_ret = R_total @ pesos
        pnl = pl * port_ret
        
        # MÉTRICAS DE RISCO
        var = -np.quantile(pnl, 1-alpha)
        es = -pnl[pnl <= -var].mean() if len(pnl[pnl <= -var]) > 0 else 0
        
        # Métricas adicionais
        mean_return = pnl.mean()
        std_return = pnl.std()
        sharpe = (mean_return / std_return) * np.sqrt(252/horizonte_dias) if std_return > 0 else 0
        
        downside_returns = pnl[pnl < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = (mean_return / downside_std) * np.sqrt(252/horizonte_dias) if downside_std > 0 else 0
        
        max_loss = pnl.min()
        max_gain = pnl.max()
        prob_loss = (pnl < 0).mean() * 100
        
        skewness = stats.skew(port_ret)
        kurtosis_value = stats.kurtosis(port_ret)
        
        progress_bar.progress(100)
        status_text.text("Simulação concluída!")
    
    # EXIBIÇÃO DE RESULTADOS
    st.success(f"✅ Simulação concluída com {n_sims:,} cenários!")
    
    # Métricas principais em cards
    st.subheader("📊 Métricas de Risco Principal")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label=f"VaR {nivel_conf}",
        value=f"R$ {var:,.0f}",
        delta=f"{var/pl*100:.2f}% do PL",
        delta_color="inverse"
    )
    col2.metric(
        label="CVaR/ES",
        value=f"R$ {es:,.0f}",
        delta=f"{es/pl*100:.2f}% do PL",
        delta_color="inverse"
    )
    col3.metric(
        label="Probabilidade de Perda",
        value=f"{prob_loss:.1f}%",
        delta=f"{prob_loss-50:.1f}pp vs 50%",
        delta_color="inverse" if prob_loss > 50 else "normal"
    )
    col4.metric(
        label="Sharpe Ratio",
        value=f"{sharpe:.2f}",
        delta="Bom" if sharpe > 1 else "Baixo" if sharpe < 0.5 else "Médio"
    )
    
    # Métricas secundárias
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sortino Ratio", f"{sortino:.2f}")
    col2.metric("Máxima Perda", f"R$ {max_loss:,.0f}", f"{max_loss/pl*100:.2f}% do PL")
    col3.metric("Assimetria", f"{skewness:.2f}")
    col4.metric("Curtose", f"{kurtosis_value:.2f}")
    
# VISUALIZAÇÕES
col1, col2 = st.columns([6,2])
with col1:
    st.subheader("📈 Análise Visual")
with col2:
    with st.expander("❓"):
        st.markdown("""
        **Objetivo da Análise Visual**  
        Estes gráficos permitem avaliar como os riscos se distribuem no portfólio e onde estão os pontos de atenção.  
        Use cada um deles para embasar decisões de alocação, hedge e monitoramento.  

        **Gráficos e como interpretar para decisão:**

        - **Distribuição de P&L**  
          Mostra a frequência de lucros e perdas simulados.  
          ➝ Se a cauda esquerda é muito longa, há risco elevado de perdas extremas.  
          ➝ **Decisão**: reduzir ativos voláteis ou reforçar hedge.

        - **Q-Q Plot (Normalidade)**  
          Compara os retornos simulados com uma curva normal.  
          ➝ Desvios da reta sugerem caudas pesadas e maior risco de choques.  
          ➝ **Decisão**: revisar premissas de distribuição ou aumentar reservas de capital.

        - **Função de Distribuição (CDF)**  
          Mostra a probabilidade acumulada de perdas e ganhos.  
          ➝ Ajuda a identificar a chance de perdas além do VaR.  
          ➝ **Decisão**: ajustar limites de risco conforme a tolerância do investidor.

        - **Decomposição do Risco**  
          Mostra a contribuição de cada classe de ativo para o risco total.  
          ➝ Se um ativo domina o risco, pode haver concentração perigosa.  
          ➝ **Decisão**: diversificar e rebalancear.

        - **Correlação Ações vs Portfólio**  
          Indica dependência do portfólio em relação às ações.  
          ➝ Correlação alta significa que quedas em ações afetam fortemente o fundo.  
          ➝ **Decisão**: buscar ativos descorrelacionados como proteção.

        - **VaR Móvel (rolling)**  
          Evolução do VaR ao longo do tempo/simulações.  
          ➝ Oscilações grandes indicam instabilidade do portfólio.  
          ➝ **Decisão**: implementar monitoramento mais frequente ou reduzir alavancagem.
        """)

   # Criar gráficos simples com matplotlib para evitar erros
if "pnl" in locals() and run_simulation:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Histograma de P&L
    axes[0, 0].hist(pnl/1000, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(-var/1000, color='red', linestyle='--', label=f'VaR {nivel_conf}')
    axes[0, 0].axvline(-es/1000, color='orange', linestyle='--', label='CVaR')
    axes[0, 0].set_xlabel('P&L (R$ mil)')
    axes[0, 0].set_ylabel('Frequência')
    axes[0, 0].set_title('Distribuição de P&L')
    axes[0, 0].legend()

    # Q-Q Plot
    sm.qqplot(pnl, line='s', ax=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')

    # Função de Distribuição Cumulativa
    sorted_pnl = np.sort(pnl)
    cdf = np.arange(1, len(sorted_pnl)+1) / len(sorted_pnl)
    axes[0, 2].plot(sorted_pnl/1000, cdf, color='blue')
    axes[0, 2].axvline(-var/1000, color='red', linestyle='--', label='VaR')
    axes[0, 2].set_title('Função de Distribuição (CDF)')
    axes[0, 2].set_xlabel('P&L (R$ mil)')
    axes[0, 2].set_ylabel('Probabilidade acumulada')
    axes[0, 2].legend()

    # Decomposição do Risco
    contrib_risco = vols_horizonte * np.array([acoes, juros, credito_privado, dolar, imobiliario, commodities, alternativos]) / 100
    axes[1, 0].bar(['Ações', 'Renda Fixa', 'Crédito Privado', 'Moeda', 'Imobiliário', 'Commodities', 'Outros'], contrib_risco)
    axes[1, 0].set_title('Decomposição do Risco')
    axes[1, 0].set_ylabel('Contribuição (desvio-padrão ajustado)')

    # Scatter Ações vs Portfólio
    axes[1, 1].scatter(np.random.normal(0, vols_horizonte[0], n_sims), pnl/1000, alpha=0.3)
    axes[1, 1].set_title('Scatter Ações vs Portfólio')
    axes[1, 1].set_xlabel('Retorno Ações')
    axes[1, 1].set_ylabel('P&L Portfólio (R$ mil)')

    # VaR Móvel (rolling)
    window = 250
    rolling_var = [np.percentile(pnl[max(0, i-window):i+1], 100*(1-float(conf_map[nivel_conf]))) for i in range(len(pnl))]
    axes[1, 2].plot(rolling_var, color='red')
    axes[1, 2].set_title(f'VaR Móvel (janela={window})')

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("🔎 Execute a simulação para visualizar os gráficos de análise.")

    
# EXPORTAÇÃO DE DADOS
if export_data:
    st.subheader("💾 Exportação de Dados")
    
    if "pnl" in locals() and run_simulation:
        col1, col2 = st.columns(2)

        with col1:
            # CSV com resultados
            results_df = pd.DataFrame({
                'Simulação': range(1, len(pnl)+1),
                'Tipo': all_labels,
                'P&L (R$)': pnl,
                'Retorno Portfolio (%)': port_ret * 100
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Baixar Simulações (CSV)",
                data=csv,
                file_name=f"VaR_Simulations_{datetime.datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON com configurações
            config_json = {
                'projeto': nome_projeto,
                'cnpj': cnpj,
                'responsavel': responsavel,
                'data_analise': datetime.datetime.now().isoformat(),
                'parametros': {
                    'patrimonio_liquido': float(pl),
                    'horizonte_dias': horizonte_dias,
                    'nivel_confianca': nivel_conf,
                    'num_simulacoes': n_sims,
                    'seed': seed
                },
                'resultados': {
                    'var': float(var),
                    'cvar': float(es),
                    'sharpe': float(sharpe),
                    'sortino': float(sortino),
                    'max_perda': float(max_loss),
                    'prob_perda': float(prob_loss)
                }
            }
            
            json_str = json.dumps(config_json, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Baixar Configurações (JSON)",
                data=json_str,
                file_name=f"VaR_Config_{datetime.datetime.now():%Y%m%d_%H%M}.json",
                mime="application/json"
            )
    else:
        st.info("🔎 Execute a simulação para habilitar a exportação de dados.")

# Footer
st.write("---")
st.caption("""
⚠️ **Disclaimer**: Este relatório é fornecido apenas para fins informativos. 
Os resultados são baseados em simulações e premissas que podem não se materializar.
""")
