# app_professional.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime, math
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

# INFORMAÇÕES DO PROJETO
st.subheader("🏢 Informações Institucionais")
col1, col2, col3 = st.columns(3)
with col1:
    nome_projeto = st.text_input("Nome do Projeto/Cliente", value="Análise de Risco - Portfolio", 
                                 help="Nome que aparecerá no relatório")
with col2:
    cnpj = st.text_input("CNPJ/Código", value="", placeholder="00.000.000/0000-00",
                        help="CNPJ ou código identificador")
with col3:
    responsavel = st.text_input("Responsável pela Análise", value="",
                               help="Nome do analista responsável")

# PARÂMETROS PRINCIPAIS
st.subheader("⚙️ Parâmetros da Simulação")
col1, col2 = st.columns(2)

with col1:
with col1:
    pl = st.number_input("Patrimônio Líquido (R$)", 
                         min_value=0.0, 
                         value=10_000_000.0, 
                         step=100_000.0,
                         format="%.2f")
    st.write(f"💰 Patrimônio Líquido Atual: R$ {pl:,.2f}")

    
    # Menu dropdown para horizonte com mais opções
    horizonte_dias = st.selectbox(
        "Horizonte Temporal", 
        options=[1, 5, 10, 15, 21, 42, 63, 126, 252, 504],
        format_func=lambda x: f"{x} dias úteis ({x/21:.1f} meses)" if x > 21 else f"{x} dias úteis",
        index=3,
        help="Período de tempo para cálculo do VaR"
    )
    
    nivel_conf = st.selectbox(
        "Nível de Confiança", 
        ["90%", "95%", "97.5%", "99%", "99.5%"],
        index=1,
        help="Probabilidade de que as perdas não excedam o VaR"
    )
    
with col2:
    conf_map = {"90%": 0.90, "95%": 0.95, "97.5%": 0.975, "99%": 0.99, "99.5%": 0.995}
    alpha = conf_map[nivel_conf]
    
    # Menu dropdown para número de simulações
    n_sims = st.selectbox(
        "Número de Simulações",
        options=[10_000, 50_000, 100_000, 250_000, 500_000],
        format_func=lambda x: f"{x:,} simulações",
        index=1,
        help="Quanto maior o número, mais preciso mas mais lento"
    )
    
    seed = st.number_input("Seed (reprodutibilidade)", 
                          min_value=0, 
                          max_value=1000000, 
                          value=42, 
                          step=1,
                          help="Define reprodutibilidade: mudar o seed altera os cenários sorteados, mas não o risco esperado.")

# ALOCAÇÃO DA CARTEIRA
st.subheader("📈 Composição da Carteira")

col1, col2, col3, col4 = st.columns(4)
with col1:
    acoes = st.slider("(Ibovespa) %", 0, 100, 40)
with col2:
    juros = st.slider("(pré ou pós) %", 0, 100, 30)
with col3:
    dolar = st.slider("Moeda Estrangeira %", 0, 100, 20)
with col4:
    commodities = st.slider("Commodities %", 0, 100, 5)

col5, col6, col7 = st.columns(3)
with col5:
    credito_privado = st.slider("Crédito Privado %", 0, 100, 5)
with col6:
    imobiliario = st.slider("Imobiliário %", 0, 100, 5)
with col7:
    outros = st.slider("Outros %", 0, 100, 0)

total_aloc = acoes + juros + dolar + commodities + credito_privado + imobiliario + outros

# Validação visual da alocação
if total_aloc > 100:
    st.error(f"⚠️ Alocação total: {total_aloc}% excede 100%!")
elif total_aloc == 100:
    st.success(f"✅ Carteira totalmente alocada: {total_aloc}%")
else:
    st.info(f"💰 Alocação: {total_aloc}% | Caixa: {100-total_aloc}%")

# Gráfico de pizza da alocação
fig_aloc = go.Figure(data=[go.Pie(
    labels=['(Ibovespa)', '(pré ou pós)', 'Moeda Estrangeira', 'Commodities',
            'Crédito Privado', 'Imobiliário', 'Outros', 'Caixa'],
    values=[acoes, juros, dolar, commodities, credito_privado, imobiliario, outros,
            max(0, 100-total_aloc)],
    hole=.3
)])

fig_aloc.update_layout(
    title="Alocação da Carteira",
    height=300,
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_aloc, use_container_width=True)

pesos = np.array([acoes, juros, dolar, commodities, credito_privado, imobiliario, outros]) / 100

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
    st.info("💡 Volatilidades anualizadas baseadas em dados históricos ou expectativas futuras")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        vol_acoes = st.number_input("Vol. Ações (%a.a.)", 5.0, 100.0, 25.0, 0.5,
                        
    with col2:
        vol_juros = st.number_input("Vol. Renda Fixa (%a.a.)", 1.0, 50.0, 8.0, 0.5,
                                  
    with col3:
        vol_dolar = st.number_input("Vol. Moeda (%a.a.)", 5.0, 50.0, 15.0, 0.5,
                                 
    with col4:
        vol_commodities = st.number_input("Vol. Commodities (%a.a.)", 10.0, 100.0, 30.0, 0.5,
                                         
    
    vols = np.array([vol_acoes, vol_juros, vol_dolar, vol_commodities]) / 100
    
    # Mostrar volatilidades ajustadas
    vols_horizonte = vols * np.sqrt(horizonte_dias/252)
    
    # Gráfico de barras das volatilidades
    fig_vol = go.Figure(data=[
        go.Bar(name='Anualizada', 
               x=['Ações', 'RF', 'Moeda', 'Commodities'],
               y=vols*100,
               marker_color='lightblue'),
        go.Bar(name=f'{horizonte_dias} dias',
               x=['Ações', 'RF', 'Moeda', 'Commodities'],
               y=vols_horizonte*100,
               marker_color='darkblue')
    ])
    fig_vol.update_layout(
        title=f"Volatilidades: Anualizada vs {horizonte_dias} dias",
        yaxis_title="Volatilidade (%)",
        barmode='group',
        height=300
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with tab2:
    st.write("### Matriz de Correlação entre Ativos")
    
    usar_correlacao = st.checkbox("Ativar correlações entre ativos", value=True,
                                 help="Controla a dependência entre ativos: correlação alta risco alto, correlação baixa risco baixo.")
    
    if usar_correlacao:
        # Templates pré-definidos
        template_corr = st.selectbox(
            "Selecionar cenário de correlação:",
            ["Personalizado", "Mercado Normal", "Crise (Flight to Quality)", 
             "Risk-On", "Stress Sistêmico", "Decorrelação"],
            help="Templates baseados em regimes de mercado históricos"
        )
        
        # Valores default baseados no template
        templates_corr = {
            "Mercado Normal": {
                "acoes_rf": -0.20, "acoes_dolar": 0.30, "acoes_comm": 0.40,
                "rf_dolar": -0.40, "rf_comm": -0.10, "dolar_comm": 0.50
            },
            "Crise (Flight to Quality)": {
                "acoes_rf": 0.40, "acoes_dolar": 0.60, "acoes_comm": 0.70,
                "rf_dolar": -0.20, "rf_comm": 0.10, "dolar_comm": 0.40
            },
            "Risk-On": {
                "acoes_rf": -0.50, "acoes_dolar": -0.40, "acoes_comm": 0.60,
                "rf_dolar": 0.30, "rf_comm": -0.20, "dolar_comm": 0.30
            },
            "Stress Sistêmico": {
                "acoes_rf": 0.80, "acoes_dolar": 0.70, "acoes_comm": 0.85,
                "rf_dolar": 0.60, "rf_comm": 0.70, "dolar_comm": 0.75
            },
            "Decorrelação": {
                "acoes_rf": 0.0, "acoes_dolar": 0.0, "acoes_comm": 0.0,
                "rf_dolar": 0.0, "rf_comm": 0.0, "dolar_comm": 0.0
            },
            "Personalizado": {
                "acoes_rf": -0.20, "acoes_dolar": 0.30, "acoes_comm": 0.40,
                "rf_dolar": -0.40, "rf_comm": -0.10, "dolar_comm": 0.50
            }
        }
        
        selected_template = templates_corr[template_corr]
        
        st.write("#### Definir Correlações")
        col1, col2, col3 = st.columns(3)
        with col1:
            corr_acoes_rf = st.slider("Ações × RF", -1.0, 1.0, 
                                      selected_template["acoes_rf"], 0.05)
            corr_acoes_dolar = st.slider("Ações × Moeda", -1.0, 1.0,
                                         selected_template["acoes_dolar"], 0.05)
        with col2:
            corr_acoes_comm = st.slider("Ações × Commodities", -1.0, 1.0,
                                       selected_template["acoes_comm"], 0.05)
            corr_rf_dolar = st.slider("RF × Moeda", -1.0, 1.0,
                                     selected_template["rf_dolar"], 0.05)
        with col3:
            corr_rf_comm = st.slider("RF × Commodities", -1.0, 1.0,
                                    selected_template["rf_comm"], 0.05)
            corr_dolar_comm = st.slider("Moeda × Commodities", -1.0, 1.0,
                                       selected_template["dolar_comm"], 0.05)
        
        # Construir matriz 4x4
        corr_matrix = np.array([
            [1.0, corr_acoes_rf, corr_acoes_dolar, corr_acoes_comm],
            [corr_acoes_rf, 1.0, corr_rf_dolar, corr_rf_comm],
            [corr_acoes_dolar, corr_rf_dolar, 1.0, corr_dolar_comm],
            [corr_acoes_comm, corr_rf_comm, corr_dolar_comm, 1.0]
        ])
        
        # Verificar se é positiva definida
        eigenvalues = np.linalg.eigvals(corr_matrix)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue <= 0:
            st.warning(f"⚠️ Ajustando matriz para ser positiva definida...")
            # Ajuste automático
            eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
            eigenvalues[eigenvalues < 0.01] = 0.01
            corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Normalizar
            D = np.diag(1.0/np.sqrt(np.diag(corr_matrix)))
            corr_matrix = D @ corr_matrix @ D
            st.success("✅ Matriz ajustada com sucesso")
        
        # Heatmap interativo
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=['Ações', 'RF', 'Moeda', 'Commodities'],
            y=['Ações', 'RF', 'Moeda', 'Commodities'],
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlação"),
            hovertemplate='%{y} × %{x}: %{z:.3f}<extra></extra>'
        ))
        fig_corr.update_layout(
            title="Matriz de Correlação",
            height=400,
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        corr_matrix = np.eye(4)
        st.info("📊 Ativos configurados como independentes")

with tab3:
    st.write("### Distribuições de Probabilidade")
    st.info("💡 Diferentes distribuições capturam características específicas dos retornos")
    
    col1, col2 = st.columns(2)
    with col1:
        dist_acoes = st.selectbox(
            "Distribuição - Ações",
            ["Normal", "t-Student", "Lognormal", "Normal Mixture"],
            help="Distribuição define o formato dos retornos: Normal (otimista), t-Student (crises), Lognormal (positivos), Mixture (volatilidade variável)." )
        if dist_acoes == "t-Student":
            df_acoes = st.slider("Graus de liberdade (Ações)", 3, 30, 5,
                                help="Menor valor = caudas mais pesadas")
        elif dist_acoes == "Normal Mixture":
            mix_prob = st.slider("Probabilidade regime volátil (%)", 5, 30, 10) / 100
            mix_scale = st.slider("Multiplicador de volatilidade", 2.0, 5.0, 3.0)
        
        dist_juros = st.selectbox(
            "Distribuição - Renda Fixa",
            ["Normal", "t-Student", "Lognormal"],
            help="Normal é adequado para títulos de baixo risco"
        )
        if dist_juros == "t-Student":
            df_juros = st.slider("Graus de liberdade (RF)", 3, 30, 10)
    
    with col2:
        dist_dolar = st.selectbox(
            "Distribuição - Moeda",
            ["Normal", "t-Student", "Lognormal"],
            help="t-Student captura saltos cambiais"
        )
        if dist_dolar == "t-Student":
            df_dolar = st.slider("Graus de liberdade (Moeda)", 3, 30, 7)
        
        dist_commodities = st.selectbox(
            "Distribuição - Commodities",
            ["Normal", "t-Student", "Lognormal"],
            help="Lognormal: apenas retornos positivos possíveis"
        )
        if dist_commodities == "t-Student":
            df_commodities = st.slider("Graus de liberdade (Commodities)", 3, 30, 5)

with tab4:
    st.write("### Cenários de Stress Determinísticos")
    
    usar_cenarios = st.checkbox("Incluir cenários de stress históricos", value=True,
                              help="Inclui choques históricos/determinísticos para testar a resiliência da carteira.")
    
    if usar_cenarios:
        # Inicializar cenários
        if 'cenarios' not in st.session_state:
            st.session_state.cenarios = pd.DataFrame({
                'Nome': ['Crise 2008', 'COVID-19', 'Taper Tantrum', 'Brexit', 'Guerra Comercial'],
                'Ações (%)': [-38.5, -33.9, -5.8, -8.5, -15.0],
                'RF (%)': [8.0, -2.0, 12.0, 3.5, 5.0],
                'Moeda (%)': [25.0, 18.0, 15.0, 12.0, 10.0],
                'Commodities (%)': [-45.0, -30.0, -10.0, -5.0, -20.0],
                'Probabilidade (%)': [2.0, 2.0, 5.0, 3.0, 4.0]
            })
        
        # Editor de cenários
        edited_df = st.data_editor(
            st.session_state.cenarios,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Nome": st.column_config.TextColumn("Cenário", width="medium"),
                "Ações (%)": st.column_config.NumberColumn("Ações (%)", format="%.1f"),
                "RF (%)": st.column_config.NumberColumn("RF (%)", format="%.1f"),
                "Moeda (%)": st.column_config.NumberColumn("Moeda (%)", format="%.1f"),
                "Commodities (%)": st.column_config.NumberColumn("Commodities (%)", format="%.1f"),
                "Probabilidade (%)": st.column_config.NumberColumn("Prob (%)", format="%.1f")
            }
        )
        st.session_state.cenarios = edited_df
        
        if len(st.session_state.cenarios) > 0:
            pct_stress = st.slider("Percentual de cenários de stress", 5, 30, 10,
                                 help="Define quanto % das simulações será reservado a eventos extremos: mais peso = cenários mais pessimistas."
)

with tab5:
    st.write("### Configurações de Backtesting")
    st.info("💡 Validação do modelo com dados históricos")
    
    realizar_backtest = st.checkbox("Realizar backtesting histórico", value=False)
    if realizar_backtest:
        col1, col2 = st.columns(2)
        with col1:
            periodo_backtest = st.selectbox(
                "Período de backtesting",
                ["1 ano", "2 anos", "3 anos", "5 anos"],
                help="Período histórico para validação"
            )
        with col2:
            metodo_backtest = st.selectbox(
                "Método de backtesting",
                ["Kupiec (POF)", "Christoffersen", "Ambos"],
                help="Kupiec: calibração (proporção de falhas). Christoffersen: independência (falhas em sequência). Ambos: validação robusta do VaR."
)

# BOTÃO DE SIMULAÇÃO
st.write("---")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    run_simulation = st.button("🚀 Executar Simulação", 
                              type="primary", 
                              use_container_width=True,
                              help="Iniciar simulação Monte Carlo")
with col2:
    generate_report = st.checkbox("📄 Gerar Relatório PDF", value=True,
                                 help="Criar relatório institucional completo")
with col3:
    export_data = st.checkbox("💾 Exportar Dados", value=True,
                            help="Salvar resultados em CSV/JSON")

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
                Z = np.random.normal(size=(n_mc, 4))
                Z_corr = Z @ L.T
            else:
                Z_corr = np.random.normal(size=(n_mc, 4))
            
            R_mc = np.zeros((n_mc, 4))
            
            # Aplicar distribuições
            distributions = [dist_acoes, dist_juros, dist_dolar, dist_commodities]
            for j, (dist, vol) in enumerate(zip(distributions, vols_d)):
                if dist == "Normal":
                    R_mc[:, j] = vol * Z_corr[:, j]
                
                elif dist == "t-Student":
                    df_values = [df_acoes if 'df_acoes' in locals() else 5,
                               df_juros if 'df_juros' in locals() else 10,
                               df_dolar if 'df_dolar' in locals() else 7,
                               df_commodities if 'df_commodities' in locals() else 5]
                    df = df_values[j]
                    t_samples = stats.t.ppf(stats.norm.cdf(Z_corr[:, j]), df)
                    scale_factor = np.sqrt(df / (df - 2)) if df > 2 else 1
                    R_mc[:, j] = vol * t_samples / scale_factor
                
                elif dist == "Lognormal":
                    mu_log = -0.5 * vol**2
                    R_mc[:, j] = np.exp(mu_log + vol * Z_corr[:, j]) - 1
                
                elif dist == "Normal Mixture":
                    is_volatile = np.random.rand(n_mc) < mix_prob
                    normal_part = vol * Z_corr[:, j]
                    volatile_part = vol * mix_scale * Z_corr[:, j]
                    R_mc[:, j] = np.where(is_volatile, volatile_part, normal_part)
            
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
                    row['Moeda (%)'] / 100,
                    row['Commodities (%)'] / 100
                ])
                all_labels.append(row['Nome'])
            
            R_stress = np.array(R_stress)
            all_returns.append(R_stress)
        
        progress_bar.progress(70)
        status_text.text("Calculando métricas de risco...")
        
        # Combinar retornos
        R_total = np.vstack(all_returns) if len(all_returns) > 0 else np.zeros((n_sims, 4))
        
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
        delta_color="inverse",
        help=f"Perda máxima esperada em {100-alpha*100:.0f}% dos casos"
    )
    col2.metric(
        label="CVaR/ES",
        value=f"R$ {es:,.0f}",
        delta=f"{es/pl*100:.2f}% do PL",
        delta_color="inverse",
        help="Perda média quando o VaR é excedido"
    )
    col3.metric(
        label="Probabilidade de Perda",
        value=f"{prob_loss:.1f}%",
        delta=f"{prob_loss-50:.1f}pp vs 50%",
        delta_color="inverse" if prob_loss > 50 else "normal",
        help="Chance de ter retorno negativo"
    )
    col4.metric(
        label="Sharpe Ratio",
        value=f"{sharpe:.2f}",
        delta="Bom" if sharpe > 1 else "Baixo" if sharpe < 0.5 else "Médio",
        help="Retorno ajustado ao risco (>1 é bom)"
    )
    
    # Métricas secundárias
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sortino Ratio", f"{sortino:.2f}",
               help="Similar ao Sharpe mas considera apenas downside")
    col2.metric("Máxima Perda", f"R$ {max_loss:,.0f}",
               f"{max_loss/pl*100:.2f}% do PL")
    col3.metric("Assimetria", f"{skewness:.2f}",
               help="Negativo indica cauda esquerda pesada")
    col4.metric("Curtose", f"{kurtosis_value:.2f}",
               help=">0 indica caudas pesadas")
    
    # VISUALIZAÇÕES INTERATIVAS
    st.subheader("📈 Análise Visual Interativa")
    
    # Criar subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Distribuição de P&L',
            'Q-Q Plot de Normalidade',
            'Distribuição Acumulada (CDF)',
            'Decomposição do Risco',
            'Análise de Correlação dos Retornos',
            'Value at Risk Condicional'
        ),
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'scatter'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # 1. Histograma de P&L
    if n_stress_total > 0:
        fig.add_trace(
            go.Histogram(
                x=pnl[:n_mc]/1000,
                name='Monte Carlo',
                marker_color='blue',
                opacity=0.6,
                nbinsx=50,
                hovertemplate='P&L: R$ %{x:.0f}k<br>Frequência: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=pnl[n_mc:]/1000,
                name='Stress',
                marker_color='red',
                opacity=0.6,
                nbinsx=30,
                hovertemplate='P&L: R$ %{x:.0f}k<br>Frequência: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Histogram(
                x=pnl/1000,
                name='Simulações',
                marker_color='blue',
                opacity=0.7,
                nbinsx=50,
                hovertemplate='P&L: R$ %{x:.0f}k<br>Frequência: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Adicionar linhas de VaR e CVaR
    fig.add_vline(x=-var/1000, line_dash="dash", line_color="red",
                 annotation_text=f"VaR: R$ {var/1000:.0f}k", row=1, col=1)
    fig.add_vline(x=-es/1000, line_dash="dash", line_color="orange",
                 annotation_text=f"CVaR: R$ {es/1000:.0f}k", row=1, col=1)
    
    # 2. Q-Q Plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(port_ret)))
    sample_quantiles = np.sort(port_ret * 100)
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            marker=dict(size=2, color='blue'),
            name='Dados',
            hovertemplate='Teórico: %{x:.2f}<br>Observado: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Linha de referência
    fig.add_trace(
        go.Scatter(
            x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
            y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Normal',
            hoverinfo='skip'
        ),
        row=1, col=2
    )
    
    # 3. CDF
    sorted_pnl = np.sort(pnl)
    cdf = np.arange(1, len(sorted_pnl)+1) / len(sorted_pnl)
    
    fig.add_trace(
        go.Scatter(
            x=sorted_pnl/1000,
            y=cdf*100,
            mode='lines',
            line=dict(color='darkblue', width=2),
            name='CDF',
            fill='tozeroy',
            fillcolor='rgba(0,100,200,0.1)',
            hovertemplate='P&L: R$ %{x:.0f}k<br>Probabilidade: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=3
    )
    
    # Marcar VaR na CDF
    fig.add_vline(x=-var/1000, line_dash="dash", line_color="red", row=1, col=3)
    fig.add_hline(y=(1-alpha)*100, line_dash="dot", line_color="gray", row=1, col=3)
    
    # 4. Decomposição do Risco
    contrib_acoes = np.std(R_total[:, 0] * pesos[0] * pl) / 1000
    contrib_juros = np.std(R_total[:, 1] * pesos[1] * pl) / 1000
    contrib_dolar = np.std(R_total[:, 2] * pesos[2] * pl) / 1000
    contrib_comm = np.std(R_total[:, 3] * pesos[3] * pl) / 1000
    
    fig.add_trace(
        go.Bar(
            x=['Ações', 'RF', 'Moeda', 'Commodities'],
            y=[contrib_acoes, contrib_juros, contrib_dolar, contrib_comm],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
            text=[f'R$ {c:.0f}k' for c in [contrib_acoes, contrib_juros, contrib_dolar, contrib_comm]],
            textposition='outside',
            hovertemplate='%{x}<br>Contribuição: R$ %{y:.0f}k<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 5. Scatter de correlação
    fig.add_trace(
        go.Scatter(
            x=R_total[:min(5000, len(R_total)), 0]*100,
            y=port_ret[:min(5000, len(port_ret))]*100,
            mode='markers',
            marker=dict(
                size=2,
                color=port_ret[:min(5000, len(port_ret))]*100,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Retorno (%)", x=0.63, len=0.35)
            ),
            hovertemplate='Ações: %{x:.1f}%<br>Portfolio: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 6. VaR Condicional (rolling)
    window = max(100, len(pnl) // 50)
    rolling_var = pd.Series(pnl).rolling(window=window).quantile(1-alpha).values * -1
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(rolling_var))),
            y=rolling_var/1000,
            mode='lines',
            line=dict(color='darkred', width=1),
            name='VaR Móvel',
            hovertemplate='Simulação: %{x}<br>VaR: R$ %{y:.0f}k<extra></extra>'
        ),
        row=2, col=3
    )
    
    fig.add_hline(y=var/1000, line_dash="dash", line_color="red",
                 annotation_text=f"VaR Total", row=2, col=3)
    
    # Layout geral
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Dashboard de Risco - {nome_projeto}",
        title_font_size=20,
        hovermode='closest'
    )
    
    # Ajustar eixos
    fig.update_xaxes(title_text="P&L (R$ mil)", row=1, col=1)
    fig.update_xaxes(title_text="Quantis Teóricos", row=1, col=2)
    fig.update_xaxes(title_text="P&L (R$ mil)", row=1, col=3)
    fig.update_xaxes(title_text="Classe de Ativo", row=2, col=1)
    fig.update_xaxes(title_text="Retorno Ações (%)", row=2, col=2)
    fig.update_xaxes(title_text="Número da Simulação", row=2, col=3)
    
    fig.update_yaxes(title_text="Frequência", row=1, col=1)
    fig.update_yaxes(title_text="Quantis Amostrais (%)", row=1, col=2)
    fig.update_yaxes(title_text="Probabilidade (%)", row=1, col=3)
    fig.update_yaxes(title_text="Contribuição (R$ mil)", row=2, col=1)
    fig.update_yaxes(title_text="Retorno Portfolio (%)", row=2, col=2)
    fig.update_yaxes(title_text="VaR (R$ mil)", row=2, col=3)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ANÁLISE DETALHADA DE CENÁRIOS
    if usar_cenarios and n_stress_total > 0:
        st.subheader("🎯 Análise de Cenários de Stress")
        
        # Criar figura para cenários
        fig_stress = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Impacto por Cenário', 'Frequência de Cenários'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Calcular impactos
        cenarios_unicos = st.session_state.cenarios['Nome'].values
        impactos = []
        counts = []
        
        for cenario in cenarios_unicos:
            mask = np.array(all_labels[n_mc:]) == cenario
            if mask.sum() > 0:
                impacto = pnl[n_mc:][mask].mean()
                impactos.append(impacto/1000)
                counts.append(mask.sum())
            else:
                impactos.append(0)
                counts.append(0)
        
        # Gráfico de barras horizontais
        colors = ['red' if x < 0 else 'green' for x in impactos]
        fig_stress.add_trace(
            go.Bar(
                y=cenarios_unicos,
                x=impactos,
                orientation='h',
                marker_color=colors,
                text=[f'R$ {imp:.0f}k (n={c})' for imp, c in zip(impactos, counts)],
                textposition='outside',
                hovertemplate='%{y}<br>Impacto: R$ %{x:.0f}k<br>Frequência: %{text}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Gráfico de pizza
        fig_stress.add_trace(
            go.Pie(
                labels=cenarios_unicos,
                values=counts,
                hole=0.3,
                hovertemplate='%{label}<br>Contagem: %{value}<br>Percentual: %{percent}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig_stress.update_layout(height=400, showlegend=False)
        fig_stress.update_xaxes(title_text="Impacto Médio (R$ mil)", row=1, col=1)
        
        st.plotly_chart(fig_stress, use_container_width=True)
    
    # TABELAS ESTATÍSTICAS
    with st.expander("📊 Estatísticas Detalhadas", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Resumo Estatístico", "Análise de Percentis", "Testes de Validação"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Métricas de Posição e Dispersão")
                stats_df = pd.DataFrame({
                    'Métrica': [
                        'Retorno Esperado (R$)',
                        'Retorno Esperado (%)',
                        'Volatilidade (R$)',
                        'Volatilidade (%)',
                        'Downside Deviation',
                        'Upside Potential'
                    ],
                    'Valor': [
                        f"{mean_return:,.0f}",
                        f"{mean_return/pl*100:.3f}%",
                        f"{std_return:,.0f}",
                        f"{std_return/pl*100:.2f}%",
                        f"{downside_std:,.0f}",
                        f"{pnl[pnl>0].std():,.0f}" if len(pnl[pnl>0]) > 0 else "0"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("#### Métricas de Forma e Risco")
                risk_df = pd.DataFrame({
                    'Métrica': [
                        f'VaR {nivel_conf}',
                        'CVaR/ES',
                        'Máxima Perda',
                        'Máximo Ganho',
                        'Assimetria',
                        'Curtose Excesso'
                    ],
                    'Valor': [
                        f"R$ {var:,.0f}",
                        f"R$ {es:,.0f}",
                        f"R$ {max_loss:,.0f}",
                        f"R$ {max_gain:,.0f}",
                        f"{skewness:.3f}",
                        f"{kurtosis_value:.3f}"
                    ]
                })
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.write("#### Distribuição Completa de Percentis")
            percentiles = [0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 99.9]
            percentile_values = [np.percentile(pnl, p) for p in percentiles]
            
            percentile_df = pd.DataFrame({
                'Percentil': [f"{p:.1f}%" for p in percentiles],
                'P&L (R$)': [f"{val:,.0f}" for val in percentile_values],
                'Retorno (%)': [f"{val/pl*100:.3f}%" for val in percentile_values],
                'Interpretação': [
                    "Cauda extrema inferior" if p <= 1 else
                    "Cauda inferior" if p <= 5 else
                    "Quartil inferior" if p <= 25 else
                    "Mediana" if p == 50 else
                    "Quartil superior" if p <= 75 else
                    "Cauda superior" if p <= 95 else
                    "Cauda extrema superior"
                    for p in percentiles
                ]
            })
            st.dataframe(percentile_df, use_container_width=True, hide_index=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Testes de Normalidade")
                
                # Shapiro-Wilk
                if len(port_ret) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(port_ret)
                else:
                    sample = np.random.choice(port_ret, 5000, replace=False)
                    shapiro_stat, shapiro_p = stats.shapiro(sample)
                
                # Jarque-Bera
                jarque_stat, jarque_p = stats.jarque_bera(port_ret)
                
                # Anderson-Darling
                anderson_result = stats.anderson(port_ret, dist='norm')
                
                normality_df = pd.DataFrame({
                    'Teste': ['Shapiro-Wilk', 'Jarque-Bera', 'Anderson-Darling'],
                    'Estatística': [
                        f"{shapiro_stat:.4f}",
                        f"{jarque_stat:.4f}",
                        f"{anderson_result.statistic:.4f}"
                    ],
                    'p-valor': [
                        f"{shapiro_p:.4f}",
                        f"{jarque_p:.4f}",
                        "Ver níveis críticos"
                    ],
                    'Conclusão': [
                        "✅ Normal" if shapiro_p > 0.05 else "❌ Não-Normal",
                        "✅ Normal" if jarque_p > 0.05 else "❌ Não-Normal",
                        "✅ Normal" if anderson_result.statistic < anderson_result.critical_values[2] else "❌ Não-Normal"
                    ]
                })
                st.dataframe(normality_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("#### Validação do VaR (Backtesting)")
                
                violations = (pnl <= -var).sum()
                expected_violations = n_sims * (1 - alpha)
                violation_rate = violations / n_sims
                
                # Teste de Kupiec
                if violations > 0 and violations < n_sims:
                    lr_uc = -2 * (violations * np.log(1-alpha) + (n_sims-violations) * np.log(alpha) -
                                 violations * np.log(violation_rate) - (n_sims-violations) * np.log(1-violation_rate))
                    kupiec_p = 1 - stats.chi2.cdf(lr_uc, 1)
                else:
                    kupiec_p = 0.0
                
                backtest_df = pd.DataFrame({
                    'Métrica': [
                        'Violações Observadas',
                        'Violações Esperadas',
                        'Taxa de Violação',
                        'Taxa Esperada',
                        'Teste Kupiec (p-valor)',
                        'Conclusão'
                    ],
                    'Valor': [
                        f"{violations} ({violation_rate*100:.2f}%)",
                        f"{expected_violations:.0f} ({(1-alpha)*100:.1f}%)",
                        f"{violation_rate*100:.2f}%",
                        f"{(1-alpha)*100:.1f}%",
                        f"{kupiec_p:.4f}",
                        "✅ VaR Adequado" if 0.05 < kupiec_p < 0.95 else "⚠️ Revisar Modelo"
                    ]
                })
                st.dataframe(backtest_df, use_container_width=True, hide_index=True)
    
    # GERAÇÃO DO RELATÓRIO PDF
    if generate_report:
        st.subheader("📄 Geração de Relatório Institucional")
        
        with st.spinner("Gerando relatório PDF..."):
            pdf_buffer = BytesIO()
            
            with PdfPages(pdf_buffer) as pdf:
                # Página 1: Capa
                fig_capa = plt.figure(figsize=(8.5, 11))
                fig_capa.patch.set_facecolor('white')
                
                # Logo/Título
                plt.text(0.5, 0.85, 'RELATÓRIO DE VALUE AT RISK', 
                        fontsize=24, fontweight='bold', ha='center')
                plt.text(0.5, 0.80, 'Análise de Risco de Mercado', 
                        fontsize=16, ha='center', style='italic')
                
                # Informações do projeto
                plt.text(0.5, 0.65, nome_projeto if nome_projeto else "Análise de Portfolio",
                        fontsize=18, ha='center')
                if cnpj:
                    plt.text(0.5, 0.60, f"CNPJ: {cnpj}",
                            fontsize=12, ha='center')
                
                # Data e responsável
                plt.text(0.5, 0.45, f"Data: {datetime.datetime.now():%d de %B de %Y}",
                        fontsize=12, ha='center')
                if responsavel:
                    plt.text(0.5, 0.40, f"Responsável: {responsavel}",
                            fontsize=12, ha='center')
                
                # Rodapé
                plt.text(0.5, 0.10, "Documento Confidencial",
                        fontsize=10, ha='center', style='italic')
                plt.text(0.5, 0.05, "Sistema VaR Monte Carlo Professional",
                        fontsize=8, ha='center')
                
                plt.axis('off')
                pdf.savefig(fig_capa, bbox_inches='tight')
                plt.close()
                
                # Página 2: Sumário Executivo
                fig_sumario = plt.figure(figsize=(8.5, 11))
                fig_sumario.patch.set_facecolor('white')
                
                sumario_text = f"""
SUMÁRIO EXECUTIVO
{'='*60}

RESULTADOS PRINCIPAIS
{'-'*40}
Value at Risk ({nivel_conf}): R$ {var:,.0f}
  • Representa {var/pl*100:.2f}% do patrimônio líquido
  • Perda máxima esperada em {(alpha)*100:.0f}% dos casos

Conditional VaR (CVaR/ES): R$ {es:,.0f}
  • Perda média quando VaR é excedido
  • Representa {es/pl*100:.2f}% do patrimônio

Probabilidade de Perda: {prob_loss:.1f}%
  • Chance de retorno negativo no horizonte

Indicadores de Performance:
  • Sharpe Ratio: {sharpe:.3f}
  • Sortino Ratio: {sortino:.3f}

PARÂMETROS DA ANÁLISE
{'-'*40}
Patrimônio Líquido: R$ {pl:,.0f}
Horizonte Temporal: {horizonte_dias} dias úteis
Simulações Realizadas: {n_sims:,}
Nível de Confiança: {nivel_conf}

COMPOSIÇÃO DA CARTEIRA
{'-'*40}
Ações: {acoes}%
Renda Fixa: {juros}%
Moeda Estrangeira: {dolar}%
Commodities: {commodities}%
Posição em Caixa: {max(0, 100-total_aloc)}%

CONCLUSÕES
{'-'*40}
{'Risco BAIXO - Carteira conservadora' if var/pl < 0.02 else 
 'Risco MODERADO - Carteira balanceada' if var/pl < 0.05 else
 'Risco ALTO - Carteira agressiva' if var/pl < 0.10 else
 'Risco MUITO ALTO - Revisar alocação'}

Assimetria: {'Negativa (cauda esquerda pesada)' if skewness < -0.5 else
            'Aproximadamente simétrica' if -0.5 <= skewness <= 0.5 else
            'Positiva (cauda direita pesada)'}

Curtose: {'Caudas pesadas (leptocúrtica)' if kurtosis_value > 1 else
         'Aproximadamente normal' if -1 <= kurtosis_value <= 1 else
         'Caudas leves (platicúrtica)'}
                """
                
                plt.text(0.1, 0.95, sumario_text, fontsize=10, 
                        verticalalignment='top', fontfamily='monospace')
                plt.axis('off')
                pdf.savefig(fig_sumario, bbox_inches='tight')
                plt.close()
                
                # Página 3: Gráficos principais
                fig_graficos = plt.figure(figsize=(11, 8.5))
                
                # Distribuição de P&L
                ax1 = plt.subplot(2, 3, 1)
                ax1.hist(pnl/1000, bins=50, alpha=0.7, color='blue', edgecolor='black')
                ax1.axvline(-var/1000, color='red', linestyle='--', linewidth=2, label=f'VaR {nivel_conf}')
                ax1.axvline(-es/1000, color='orange', linestyle='--', linewidth=2, label='CVaR')
                ax1.set_xlabel('P&L (R$ mil)')
                ax1.set_ylabel('Frequência')
                ax1.set_title('Distribuição de Resultados')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Q-Q Plot
                ax2 = plt.subplot(2, 3, 2)
                stats.probplot(port_ret * 100, dist="norm", plot=ax2)
                ax2.set_title('Q-Q Plot (Normalidade)')
                ax2.grid(True, alpha=0.3)
                
                # CDF
                ax3 = plt.subplot(2, 3, 3)
                sorted_pnl = np.sort(pnl)
                cdf = np.arange(1, len(sorted_pnl)+1) / len(sorted_pnl)
                ax3.plot(sorted_pnl/1000, cdf*100, linewidth=2, color='darkblue')
                ax3.axvline(-var/1000, color='red', linestyle='--', label=f'VaR {nivel_conf}')
                ax3.axhline((1-alpha)*100, color='red', linestyle=':', alpha=0.5)
                ax3.fill_betweenx([0, (1-alpha)*100], sorted_pnl.min()/1000, -var/1000,
                                 alpha=0.2, color='red')
                ax3.set_xlabel('P&L (R$ mil)')
                ax3.set_ylabel('Probabilidade Acumulada (%)')
                ax3.set_title('Função de Distribuição')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Decomposição do risco
                ax4 = plt.subplot(2, 3, 4)
                contributions = [contrib_acoes, contrib_juros, contrib_dolar, contrib_comm]
                colors_contrib = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
                bars = ax4.bar(['Ações', 'RF', 'Moeda', 'Comm.'], contributions, 
                              color=colors_contrib, alpha=0.7)
                ax4.set_ylabel('Contribuição (R$ mil)')
                ax4.set_title('Decomposição do Risco')
                ax4.grid(True, alpha=0.3, axis='y')
                
                for bar, val in zip(bars, contributions):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.0f}', ha='center', va='bottom')
                
                # Correlação
                ax5 = plt.subplot(2, 3, 5)
                scatter = ax5.scatter(R_total[:min(1000, len(R_total)), 0]*100,
                                    port_ret[:min(1000, len(port_ret))]*100,
                                    c=port_ret[:min(1000, len(port_ret))]*100,
                                    cmap='RdYlGn', alpha=0.5, s=1)
                ax5.set_xlabel('Retorno Ações (%)')
                ax5.set_ylabel('Retorno Portfolio (%)')
                ax5.set_title('Análise de Correlação')
                ax5.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax5)
                
                # Matriz de correlação
                ax6 = plt.subplot(2, 3, 6)
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                           center=0, square=True, cbar_kws={"shrink": 0.8},
                           xticklabels=['Ações', 'RF', 'Moeda', 'Comm.'],
                           yticklabels=['Ações', 'RF', 'Moeda', 'Comm.'])
                ax6.set_title('Matriz de Correlação')
                
                plt.suptitle(f'Análise de Risco - {nome_projeto}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                pdf.savefig(fig_graficos, bbox_inches='tight')
                plt.close()
                
                # Página 4: Tabelas estatísticas
                fig_tabelas = plt.figure(figsize=(8.5, 11))
                fig_tabelas.patch.set_facecolor('white')
                
                tabelas_text = f"""
ANÁLISE ESTATÍSTICA DETALHADA
{'='*60}

ESTATÍSTICAS DESCRITIVAS
{'-'*40}
Média:                  R$ {mean_return:>15,.0f} ({mean_return/pl*100:>6.3f}%)
Mediana:                R$ {np.median(pnl):>15,.0f} ({np.median(pnl)/pl*100:>6.3f}%)
Desvio Padrão:          R$ {std_return:>15,.0f} ({std_return/pl*100:>6.2f}%)
Assimetria:                {skewness:>15.3f}
Curtose:                   {kurtosis_value:>15.3f}

MÉTRICAS DE RISCO
{'-'*40}
VaR {nivel_conf:>5}:              R$ {var:>15,.0f} ({var/pl*100:>6.2f}%)
CVaR/ES:                R$ {es:>15,.0f} ({es/pl*100:>6.2f}%)
Máxima Perda:           R$ {max_loss:>15,.0f} ({max_loss/pl*100:>6.2f}%)
Máximo Ganho:           R$ {max_gain:>15,.0f} ({max_gain/pl*100:>6.2f}%)

INDICADORES DE PERFORMANCE
{'-'*40}
Sharpe Ratio:                      {sharpe:>10.3f}
Sortino Ratio:                     {sortino:>10.3f}
Probabilidade de Perda:            {prob_loss:>10.1f}%

PERCENTIS DA DISTRIBUIÇÃO
{'-'*40}
Percentil  1%:          R$ {np.percentile(pnl, 1):>15,.0f}
Percentil  5%:          R$ {np.percentile(pnl, 5):>15,.0f}
Percentil 10%:          R$ {np.percentile(pnl, 10):>15,.0f}
Percentil 25%:          R$ {np.percentile(pnl, 25):>15,.0f}
Percentil 50%:          R$ {np.percentile(pnl, 50):>15,.0f}
Percentil 75%:          R$ {np.percentile(pnl, 75):>15,.0f}
Percentil 90%:          R$ {np.percentile(pnl, 90):>15,.0f}
Percentil 95%:          R$ {np.percentile(pnl, 95):>15,.0f}
Percentil 99%:          R$ {np.percentile(pnl, 99):>15,.0f}

VALIDAÇÃO DO MODELO
{'-'*40}
Violações VaR:          {violations:>10} ({violation_rate*100:.2f}%)
Violações Esperadas:    {expected_violations:>10.0f} ({(1-alpha)*100:.1f}%)
Teste Kupiec p-valor:   {kupiec_p:>10.4f}
Status:                 {'APROVADO' if 0.05 < kupiec_p < 0.95 else 'REVISAR'}
                """
                
                plt.text(0.1, 0.95, tabelas_text, fontsize=9,
                        verticalalignment='top', fontfamily='monospace')
                plt.axis('off')
                pdf.savefig(fig_tabelas, bbox_inches='tight')
                plt.close()
                
                # Metadados do PDF
                d = pdf.infodict()
                d['Title'] = f'Relatório VaR - {nome_projeto}'
                d['Author'] = responsavel if responsavel else 'VaR System'
                d['Subject'] = 'Análise de Value at Risk'
                d['Keywords'] = f'VaR, Risk Management, {nivel_conf}'
                d['CreationDate'] = datetime.datetime.now()
            
            pdf_buffer.seek(0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Baixar Relatório PDF",
                data=pdf_buffer,
                file_name=f"VaR_Report_{nome_projeto.replace(' ', '_')}_{datetime.datetime.now():%Y%m%d_%H%M}.pdf",
                mime="application/pdf"
            )
    
    # EXPORTAÇÃO DE DADOS
    if export_data:
        st.subheader("💾 Exportação de Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV com resultados
            results_df = pd.DataFrame({
                'Simulação': range(1, len(pnl)+1),
                'Tipo': all_labels,
                'P&L (R$)': pnl,
                'Retorno Portfolio (%)': port_ret * 100,
                'Retorno Ações (%)': R_total[:, 0] * 100,
                'Retorno RF (%)': R_total[:, 1] * 100,
                'Retorno Moeda (%)': R_total[:, 2] * 100,
                'Retorno Commodities (%)': R_total[:, 3] * 100
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
                'alocacao': {
                    'acoes': acoes,
                    'renda_fixa': juros,
                    'moeda': dolar,
                    'commodities': commodities,
                    'caixa': max(0, 100-total_aloc)
                },
                'volatilidades_anuais': {
                    'acoes': vol_acoes,
                    'renda_fixa': vol_juros,
                    'moeda': vol_dolar,
                    'commodities': vol_commodities
                },
                'correlacoes': corr_matrix.tolist() if usar_correlacao else None,
                'resultados': {
                    'var': float(var),
                    'cvar': float(es),
                    'sharpe': float(sharpe),
                    'sortino': float(sortino),
                    'max_perda': float(max_loss),
                    'prob_perda': float(prob_loss),
                    'assimetria': float(skewness),
                    'curtose': float(kurtosis_value)
                }
            }
            
            json_str = json.dumps(config_json, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Baixar Configurações (JSON)",
                data=json_str,
                file_name=f"VaR_Config_{datetime.datetime.now():%Y%m%d_%H%M}.json",
                mime="application/json"
            )

# SIDEBAR COM INFORMAÇÕES
with st.sidebar:
    st.header("📚 Documentação")
    
    with st.expander("Sobre o VaR", expanded=False):
        st.write("""
        **Value at Risk (VaR)** é uma medida estatística que quantifica 
        o risco de perda de um portfolio em condições normais de mercado.
        
        **Interpretação**: Com X% de confiança, as perdas não excederão 
        o VaR no horizonte especificado.
        """)
    
    with st.expander("Metodologias", expanded=False):
        st.write("""
        **Monte Carlo**: Simula milhares de cenários possíveis baseados 
        em parâmetros estatísticos.
        
        **Vantagens**:
        - Flexibilidade nas distribuições
        - Incorpora não-linearidades
        - Permite cenários complexos
        """)
    
    with st.expander("Métricas", expanded=False):
        st.write("""
        **CVaR/ES**: Perda média além do VaR
        
        **Sharpe**: Retorno/Risco total
        
        **Sortino**: Retorno/Risco negativo
        
        **Assimetria**: Simetria da distribuição
        
        **Curtose**: Peso das caudas
        """)
    
    st.write("---")
    st.write("**Sistema VaR Professional**")
    st.write(f"Versão 2.0 | {datetime.datetime.now():%Y}")
    st.caption("Desenvolvido para análise institucional de risco")

# Footer
st.write("---")
st.caption("""
⚠️ **Disclaimer**: Este relatório é fornecido apenas para fins informativos. 
Os resultados são baseados em simulações e premissas que podem não se materializar. 
Decisões de investimento devem considerar múltiplos fatores e assessoria profissional.
""")
