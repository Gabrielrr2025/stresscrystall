# app_advanced.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime, math
import json
from scipy import stats
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

st.set_page_config(page_title="VaR Monte Carlo Advanced", layout="wide")
st.title("📊 VaR Monte Carlo Avançado - Finhealth")

# Inputs básicos
col1, col2 = st.columns(2)
with col1:
    pl = st.number_input("Patrimônio Líquido (R$)", 0.0, value=1_000_000.0, step=1000.0)
    horizonte_dias = st.selectbox("Horizonte (dias)", [1, 5, 10, 21, 63, 126, 252])
    nivel_conf = st.selectbox("Nível de Confiança", ["90%", "95%", "99%", "99.5%"])
    
with col2:
    conf_map = {"90%": 0.90, "95%": 0.95, "99%": 0.99, "99.5%": 0.995}
    alpha = conf_map[nivel_conf]
    n_sims = st.number_input("Nº simulações Monte Carlo", 1000, 500000, value=50000, step=1000)
    seed = st.number_input("Seed (reprodutibilidade)", 0, 1000000, value=42, step=1)

# Carteira
st.subheader("📈 Alocação da Carteira")
col1, col2, col3 = st.columns(3)
with col1:
    acoes = st.slider("Ações %PL", 0, 100, 40)
with col2:
    juros = st.slider("Renda Fixa %PL", 0, 100, 30)
with col3:
    dolar = st.slider("Dólar %PL", 0, 100, 20)

total_aloc = acoes + juros + dolar
col1, col2 = st.columns(2)
with col1:
    if total_aloc > 100:
        st.error(f"⚠️ Alocação total: {total_aloc}% (máximo 100%)")
    else:
        st.success(f"✅ Alocação total: {total_aloc}%")
with col2:
    if total_aloc < 100:
        st.info(f"💰 Caixa: {100-total_aloc}%")

pesos = np.array([acoes, juros, dolar])/100

# CONFIGURAÇÕES AVANÇADAS
st.subheader("⚙️ Configurações Avançadas do Modelo")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Volatilidades", "🔗 Correlações", "📈 Distribuições", "🎯 Cenários Stress"])

with tab1:
    st.write("### Volatilidades Anualizadas (%)")
    col1, col2, col3 = st.columns(3)
    with col1:
        vol_acoes = st.number_input("Volatilidade Ações", 5.0, 100.0, 25.0, 0.5, format="%.1f")
    with col2:
        vol_juros = st.number_input("Volatilidade Renda Fixa", 1.0, 50.0, 8.0, 0.5, format="%.1f")
    with col3:
        vol_dolar = st.number_input("Volatilidade Dólar", 5.0, 50.0, 15.0, 0.5, format="%.1f")
    
    vols = np.array([vol_acoes, vol_juros, vol_dolar]) / 100
    
    # Mostrar volatilidades ajustadas pelo horizonte
    vols_horizonte = vols * np.sqrt(horizonte_dias/252)
    st.info(f"📏 Volatilidades para {horizonte_dias} dias: Ações={vols_horizonte[0]*100:.1f}%, RF={vols_horizonte[1]*100:.1f}%, Dólar={vols_horizonte[2]*100:.1f}%")

with tab2:
    st.write("### Matriz de Correlação")
    
    # Opção de usar correlações
    usar_correlacao = st.checkbox("Usar correlações entre ativos", value=True)
    
    if usar_correlacao:
        # Templates de correlação
        template_corr = st.selectbox(
            "Usar template de correlação:",
            ["Personalizado", "Mercado Normal", "Crise (Flight to Quality)", "Risk-On", "Decorrelação"]
        )
        
        templates_corr = {
            "Mercado Normal": {
                "acoes_rf": -0.20, "acoes_dolar": 0.30, "rf_dolar": -0.40
            },
            "Crise (Flight to Quality)": {
                "acoes_rf": 0.40, "acoes_dolar": 0.60, "rf_dolar": -0.20
            },
            "Risk-On": {
                "acoes_rf": -0.50, "acoes_dolar": -0.40, "rf_dolar": 0.30
            },
            "Decorrelação": {
                "acoes_rf": 0.0, "acoes_dolar": 0.0, "rf_dolar": 0.0
            },
            "Personalizado": {
                "acoes_rf": -0.20, "acoes_dolar": 0.30, "rf_dolar": -0.40
            }
        }
        
        selected_template = templates_corr[template_corr]
        
        # Inputs de correlação
        col1, col2, col3 = st.columns(3)
        with col1:
            corr_acoes_rf = st.slider("Correlação Ações-RF", -1.0, 1.0, 
                                      selected_template["acoes_rf"], 0.05)
        with col2:
            corr_acoes_dolar = st.slider("Correlação Ações-Dólar", -1.0, 1.0, 
                                         selected_template["acoes_dolar"], 0.05)
        with col3:
            corr_rf_dolar = st.slider("Correlação RF-Dólar", -1.0, 1.0, 
                                      selected_template["rf_dolar"], 0.05)
        
        # Construir matriz de correlação
        corr_matrix = np.array([
            [1.0, corr_acoes_rf, corr_acoes_dolar],
            [corr_acoes_rf, 1.0, corr_rf_dolar],
            [corr_acoes_dolar, corr_rf_dolar, 1.0]
        ])
        
        # Verificar se a matriz é positiva definida
        eigenvalues = np.linalg.eigvals(corr_matrix)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue <= 0:
            st.error(f"⚠️ Matriz não é positiva definida! (menor autovalor: {min_eigenvalue:.4f})")
            st.info("💡 Ajuste as correlações para garantir consistência matemática")
            # Ajustar para a matriz mais próxima positiva definida
            from scipy.linalg import sqrtm
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
            eigenvalues[eigenvalues < 0.001] = 0.001
            corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            corr_matrix = corr_matrix / np.sqrt(np.diag(corr_matrix)[:, None] * np.diag(corr_matrix)[None, :])
            st.success("✅ Matriz ajustada automaticamente")
        else:
            st.success(f"✅ Matriz válida (menor autovalor: {min_eigenvalue:.4f})")
        
        # Visualizar matriz
        fig_corr, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   xticklabels=['Ações', 'RF', 'Dólar'],
                   yticklabels=['Ações', 'RF', 'Dólar'],
                   vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8})
        ax.set_title('Matriz de Correlação')
        st.pyplot(fig_corr)
        
    else:
        # Matriz identidade (sem correlação)
        corr_matrix = np.eye(3)
        st.info("📊 Usando ativos independentes (correlação = 0)")

with tab3:
    st.write("### Distribuições dos Retornos")
    
    # Escolher distribuição para cada ativo
    col1, col2, col3 = st.columns(3)
    with col1:
        dist_acoes = st.selectbox("Distribuição Ações", 
                                  ["Normal", "t-Student", "Lognormal", "Mixture"], 
                                  index=0)
        if dist_acoes == "t-Student":
            df_acoes = st.slider("Graus de liberdade", 3, 30, 5)
        elif dist_acoes == "Mixture":
            mix_prob = st.slider("Prob. cauda pesada (%)", 0, 50, 10) / 100
            mix_scale = st.slider("Escala da cauda", 1.5, 5.0, 2.5)
    
    with col2:
        dist_juros = st.selectbox("Distribuição RF", 
                                  ["Normal", "t-Student", "Lognormal"], 
                                  index=0)
        if dist_juros == "t-Student":
            df_juros = st.slider("Graus de liberdade RF", 3, 30, 10)
    
    with col3:
        dist_dolar = st.selectbox("Distribuição Dólar", 
                                  ["Normal", "t-Student", "Lognormal"], 
                                  index=0)
        if dist_dolar == "t-Student":
            df_dolar = st.slider("Graus de liberdade Dólar", 3, 30, 7)
    
    # Informações sobre as distribuições
    with st.expander("ℹ️ Sobre as distribuições"):
        st.write("""
        - **Normal**: Distribuição padrão, simétrica
        - **t-Student**: Caudas mais pesadas (eventos extremos mais frequentes)
        - **Lognormal**: Assimétrica positiva (crashes mais severos que rallies)
        - **Mixture**: Mistura de normal com distribuição de cauda pesada
        """)

with tab4:
    st.write("### Cenários de Stress Customizados")
    
    usar_cenarios = st.checkbox("Incluir cenários de stress determinísticos", value=False)
    
    if usar_cenarios:
        # Inicializar cenários no session_state
        if 'cenarios' not in st.session_state:
            st.session_state.cenarios = pd.DataFrame({
                'Nome': ['Crise 2008', 'COVID-19', 'Taper Tantrum', 'Brexit'],
                'Ações (%)': [-38.5, -33.9, -5.8, -8.5],
                'RF (%)': [8.0, -2.0, 12.0, 3.5],
                'Dólar (%)': [25.0, 18.0, 15.0, 12.0],
                'Probabilidade (%)': [2.0, 2.0, 5.0, 3.0]
            })
        
        # Editor de cenários
        st.write("#### Editar Cenários")
        edited_df = st.data_editor(
            st.session_state.cenarios,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Nome": st.column_config.TextColumn("Cenário", required=True, width="medium"),
                "Ações (%)": st.column_config.NumberColumn("Ações (%)", format="%.1f", required=True),
                "RF (%)": st.column_config.NumberColumn("RF (%)", format="%.1f", required=True),
                "Dólar (%)": st.column_config.NumberColumn("Dólar (%)", format="%.1f", required=True),
                "Probabilidade (%)": st.column_config.NumberColumn("Prob (%)", min_value=0.0, max_value=100.0, format="%.1f")
            }
        )
        st.session_state.cenarios = edited_df
        
        # Percentual de cenários stress
        if len(st.session_state.cenarios) > 0:
            pct_stress = st.slider("% do total como cenários de stress", 0, 50, 10)
            total_prob = st.session_state.cenarios['Probabilidade (%)'].sum()
            if total_prob > 0:
                st.info(f"📊 Probabilidade total dos cenários: {total_prob:.1f}%")

# SIMULAÇÃO
st.write("---")
st.subheader("🎲 Executar Simulação")

col1, col2 = st.columns([3, 1])
with col1:
    run_simulation = st.button("🚀 Rodar Simulação VaR", type="primary", use_container_width=True)
with col2:
    generate_report = st.checkbox("📄 Gerar PDF", value=True)

if run_simulation:
    with st.spinner("Processando simulação Monte Carlo..."):
        np.random.seed(seed)
        
        # Preparar parâmetros
        vols_d = vols * np.sqrt(horizonte_dias/252)
        
        # Lista para armazenar retornos e labels
        all_returns = []
        all_labels = []
        
        # PARTE 1: SIMULAÇÃO MONTE CARLO
        if usar_cenarios and len(st.session_state.cenarios) > 0:
            n_stress_total = int(n_sims * pct_stress / 100)
            n_mc = n_sims - n_stress_total
        else:
            n_mc = n_sims
            n_stress_total = 0
        
        # Gerar retornos Monte Carlo com correlação e distribuições escolhidas
        if n_mc > 0:
            # Gerar variáveis aleatórias correlacionadas
            if usar_correlacao:
                L = np.linalg.cholesky(corr_matrix)
                Z = np.random.normal(size=(n_mc, 3))
                Z_corr = Z @ L.T
            else:
                Z_corr = np.random.normal(size=(n_mc, 3))
            
            R_mc = np.zeros((n_mc, 3))
            
            # Aplicar distribuição para cada ativo
            for j, (dist, vol) in enumerate(zip([dist_acoes, dist_juros, dist_dolar], vols_d)):
                if dist == "Normal":
                    R_mc[:, j] = vol * Z_corr[:, j]
                
                elif dist == "t-Student":
                    df = [df_acoes, df_juros, df_dolar][j] if 'df_acoes' in locals() else 5
                    # Converter normal para t-student mantendo correlação
                    t_samples = stats.t.ppf(stats.norm.cdf(Z_corr[:, j]), df)
                    # Ajustar escala para manter volatilidade desejada
                    scale_factor = np.sqrt(df / (df - 2)) if df > 2 else 1
                    R_mc[:, j] = vol * t_samples / scale_factor
                
                elif dist == "Lognormal":
                    # Parametrização lognormal
                    mu_log = -0.5 * vol**2
                    R_mc[:, j] = np.exp(mu_log + vol * Z_corr[:, j]) - 1
                
                elif dist == "Mixture":
                    # Mixture: normal + cauda pesada
                    is_tail = np.random.rand(n_mc) < mix_prob
                    normal_part = vol * Z_corr[:, j]
                    tail_part = vol * mix_scale * np.random.standard_t(3, size=n_mc)
                    R_mc[:, j] = np.where(is_tail, tail_part, normal_part)
            
            all_returns.append(R_mc)
            all_labels.extend(['Monte Carlo'] * n_mc)
        
        # PARTE 2: CENÁRIOS DE STRESS
        if usar_cenarios and n_stress_total > 0:
            cenarios_df = st.session_state.cenarios
            probs = cenarios_df['Probabilidade (%)'].values
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)
            
            # Amostrar cenários baseado nas probabilidades
            indices = np.random.choice(len(cenarios_df), size=n_stress_total, p=probs)
            
            R_stress = []
            for idx in indices:
                row = cenarios_df.iloc[idx]
                R_stress.append([
                    row['Ações (%)'] / 100,
                    row['RF (%)'] / 100,
                    row['Dólar (%)'] / 100
                ])
                all_labels.append(row['Nome'])
            
            R_stress = np.array(R_stress)
            all_returns.append(R_stress)
        
        # Combinar todos os retornos
        R_total = np.vstack(all_returns) if len(all_returns) > 0 else np.zeros((n_sims, 3))
        
        # Calcular P&L
        port_ret = R_total @ pesos
        pnl = pl * port_ret
        
        # MÉTRICAS DE RISCO
        var = -np.quantile(pnl, 1-alpha)
        es = -pnl[pnl <= -var].mean() if len(pnl[pnl <= -var]) > 0 else 0
        
        # Métricas adicionais
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252/horizonte_dias) if pnl.std() > 0 else 0
        sortino_denominator = np.sqrt(np.mean(np.minimum(pnl, 0)**2))
        sortino = (pnl.mean() / sortino_denominator) * np.sqrt(252/horizonte_dias) if sortino_denominator > 0 else 0
        max_loss = pnl.min()
        prob_loss = (pnl < 0).mean() * 100
        
    # EXIBIR RESULTADOS
    st.success("✅ Simulação concluída!")
    
    # Métricas principais
    st.subheader("📊 Métricas de Risco")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("VaR " + nivel_conf, f"R$ {var:,.0f}", f"{var/pl*100:.2f}% PL")
    col2.metric("CVaR/ES", f"R$ {es:,.0f}", f"{es/pl*100:.2f}% PL")
    col3.metric("Prob. Perda", f"{prob_loss:.1f}%")
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col5.metric("Pior Cenário", f"R$ {max_loss:,.0f}", f"{max_loss/pl*100:.2f}% PL")
    
    # VISUALIZAÇÕES
    figures = []  # Lista para armazenar figuras para o PDF
    
    # Figura 1: Dashboard principal
    fig1 = plt.figure(figsize=(16, 10))
    
    # 1.1 Histograma com separação MC vs Stress
    ax1 = plt.subplot(2, 3, 1)
    if n_stress_total > 0:
        ax1.hist(pnl[:n_mc], bins=60, alpha=0.6, label=f'Monte Carlo ({n_mc:,})', 
                color='blue', density=True, edgecolor='none')
        ax1.hist(pnl[n_mc:], bins=30, alpha=0.7, label=f'Stress ({n_stress_total:,})', 
                color='red', density=True, edgecolor='none')
    else:
        ax1.hist(pnl, bins=60, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    
    ax1.axvline(-var, color='red', linestyle='--', linewidth=2, label=f'VaR {nivel_conf}')
    ax1.axvline(-es, color='orange', linestyle='--', linewidth=2, label='CVaR')
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel('P&L (R$)')
    ax1.set_ylabel('Densidade')
    ax1.set_title('Distribuição de P&L')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 1.2 Q-Q Plot
    ax2 = plt.subplot(2, 3, 2)
    stats.probplot(port_ret * 100, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normalidade)')
    ax2.set_ylabel('Retornos Observados (%)')
    ax2.grid(True, alpha=0.3)
    
    # 1.3 CDF
    ax3 = plt.subplot(2, 3, 3)
    sorted_pnl = np.sort(pnl)
    cdf = np.arange(1, len(sorted_pnl)+1) / len(sorted_pnl)
    ax3.plot(sorted_pnl/1000, cdf*100, linewidth=2, color='darkblue')
    ax3.axvline(-var/1000, color='red', linestyle='--', label=f'VaR {nivel_conf}')
    ax3.axvline(-es/1000, color='orange', linestyle='--', label='CVaR')
    ax3.axhline((1-alpha)*100, color='red', linestyle=':', alpha=0.5)
    ax3.fill_betweenx([0, (1-alpha)*100], sorted_pnl.min()/1000, -var/1000, 
                      alpha=0.2, color='red', label='Região VaR')
    ax3.set_xlabel('P&L (R$ mil)')
    ax3.set_ylabel('Probabilidade Acumulada (%)')
    ax3.set_title('Função de Distribuição Acumulada')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 1.4 Scatter plot retornos dos ativos
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(R_total[:, 0]*100, R_total[:, 2]*100, 
                         c=port_ret*100, cmap='RdYlGn', alpha=0.5, s=1)
    plt.colorbar(scatter, ax=ax4, label='Retorno Portfolio (%)')
    ax4.set_xlabel('Retorno Ações (%)')
    ax4.set_ylabel('Retorno Dólar (%)')
    ax4.set_title('Mapa de Retornos')
    ax4.grid(True, alpha=0.3)
    
    # 1.5 Decomposição do risco
    ax5 = plt.subplot(2, 3, 5)
    contrib_acoes = (R_total[:, 0] * pesos[0] * pl).std()
    contrib_juros = (R_total[:, 1] * pesos[1] * pl).std()
    contrib_dolar = (R_total[:, 2] * pesos[2] * pl).std()
    
    contributions = [contrib_acoes, contrib_juros, contrib_dolar]
    colors_contrib = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax5.bar(['Ações', 'RF', 'Dólar'], contributions, color=colors_contrib, alpha=0.7)
    ax5.set_ylabel('Contribuição para Volatilidade (R$)')
    ax5.set_title('Decomposição do Risco')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, val in zip(bars, contributions):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'R$ {val:,.0f}', ha='center', va='bottom')
    
    # 1.6 Análise temporal (rolling VaR)
    ax6 = plt.subplot(2, 3, 6)
    window = max(100, len(pnl) // 100)
    rolling_var = pd.Series(pnl).rolling(window=window).quantile(1-alpha).values * -1
    ax6.plot(rolling_var/1000, alpha=0.7, linewidth=1, color='darkred')
    ax6.axhline(var/1000, color='red', linestyle='--', label=f'VaR Total', alpha=0.7)
    ax6.fill_between(range(len(rolling_var)), 0, rolling_var/1000, alpha=0.2, color='red')
    ax6.set_xlabel('Simulação #')
    ax6.set_ylabel('VaR (R$ mil)')
    ax6.set_title(f'VaR Móvel (janela={window})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Dashboard VaR - PL: R$ {pl:,.0f} | Horizonte: {horizonte_dias} dias', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    figures.append(fig1)
    st.pyplot(fig1)
    
    # Figura 2: Análise de Cenários (se aplicável)
    if usar_cenarios and n_stress_total > 0:
        fig2 = plt.figure(figsize=(14, 8))
        
        # 2.1 Impacto por cenário
        ax1 = plt.subplot(2, 2, 1)
        cenarios_unicos = st.session_state.cenarios['Nome'].values
        impactos = []
        counts = []
        
        for cenario in cenarios_unicos:
            mask = np.array(all_labels[n_mc:]) == cenario
            if mask.sum() > 0:
                impacto = pnl[n_mc:][mask].mean()
                impactos.append(impacto)
                counts.append(mask.sum())
            else:
                impactos.append(0)
                counts.append(0)
        
        colors = ['red' if x < 0 else 'green' for x in impactos]
        bars = ax1.barh(cenarios_unicos, np.array(impactos)/1000, color=colors, alpha=0.7)
        ax1.set_xlabel('Impacto Médio (R$ mil)')
        ax1.set_title('Impacto por Cenário de Stress')
        ax1.axvline(0, color='black', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Adicionar contagem
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f' n={count}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)
        
        # 2.2 Comparação de distribuições
        ax2 = plt.subplot(2, 2, 2)
        box_data = []
        labels_box = []
        
        if n_mc > 0:
            box_data.append(pnl[:n_mc]/1000)
            labels_box.append('Monte Carlo')
        
        for cenario in cenarios_unicos:
            mask = np.array(all_labels) == cenario
            if mask.sum() > 5:  # Só mostrar se tiver mais de 5 observações
                box_data.append(pnl[mask]/1000)
                labels_box.append(cenario[:10])  # Truncar nome se muito longo
        
        bp = ax2.boxplot(box_data, labels=labels_box, showmeans=True, patch_artist=True)
        
        # Colorir boxes
        for patch, label in zip(bp['boxes'], labels_box):
            if label == 'Monte Carlo':
                patch.set_facecolor('lightblue')
            else:
                patch.set_facecolor('lightcoral')
        
        ax2.set_ylabel('P&L (R$ mil)')
        ax2.set_title('Comparação de Cenários')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2.3 Matriz de cenários
        ax3 = plt.subplot(2, 2, 3)
        scenario_matrix = st.session_state.cenarios[['Ações (%)', 'RF (%)', 'Dólar (%)']].values
        im = ax3.imshow(scenario_matrix.T, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
        ax3.set_yticks(range(3))
        ax3.set_yticklabels(['Ações', 'RF', 'Dólar'])
        ax3.set_xticks(range(len(cenarios_unicos)))
        ax3.set_xticklabels(cenarios_unicos, rotation=45, ha='right')
        ax3.set_title('Matriz de Cenários (%)')
        plt.colorbar(im, ax=ax3)
        
        # Adicionar valores
        for i in range(3):
            for j in range(len(cenarios_unicos)):
                text = ax3.text(j, i, f'{scenario_matrix[j, i]:.0f}',
                              ha="center", va="center", color="white" if abs(scenario_matrix[j, i]) > 25 else "black",
                              fontsize=8)
        
        # 2.4 Análise de cauda
        ax4 = plt.subplot(2, 2, 4)
        tail_threshold = np.percentile(pnl, 5)
        tail_data = pnl[pnl <= tail_threshold]
        
        # Separar cauda por tipo
        tail_mc = []
        tail_stress = []
        for i, (val, label) in enumerate(zip(pnl, all_labels)):
            if val <= tail_threshold:
                if label == 'Monte Carlo':
                    tail_mc.append(val)
                else:
                    tail_stress.append(val)
        
        if len(tail_mc) > 0:
            ax4.hist(np.array(tail_mc)/1000, bins=20, alpha=0.6, label=f'MC ({len(tail_mc)})', color='blue')
        if len(tail_stress) > 0:
            ax4.hist(np.array(tail_stress)/1000, bins=15, alpha=0.6, label=f'Stress ({len(tail_stress)})', color='red')
        
        ax4.axvline(-var/1000, color='red', linestyle='--', label='VaR')
        ax4.set_xlabel('P&L (R$ mil)')
        ax4.set_ylabel('Frequência')
        ax4.set_title('Análise de Cauda (5% piores)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Análise Detalhada de Cenários de Stress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        figures.append(fig2)
        st.pyplot(fig2)
    
    # TABELAS DE ESTATÍSTICAS
    with st.expander("📈 Estatísticas Detalhadas", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Estatísticas Gerais")
            stats_df = pd.DataFrame({
                'Métrica': [
                    'Retorno Esperado', 'Volatilidade', 'VaR ' + nivel_conf, 'CVaR/ES',
                    'Assimetria', 'Curtose', 'Sharpe Ratio', 'Sortino Ratio',
                    'Máxima Perda', 'Máximo Ganho'
                ],
                'Valor': [
                    f"R$ {pnl.mean():,.0f} ({pnl.mean()/pl*100:.3f}%)",
                    f"R$ {pnl.std():,.0f} ({pnl.std()/pl*100:.2f}%)",
                    f"R$ {var:,.0f} ({var/pl*100:.2f}%)",
                    f"R$ {es:,.0f} ({es/pl*100:.2f}%)",
                    f"{stats.skew(port_ret):.3f}",
                    f"{stats.kurtosis(port_ret):.3f}",
                    f"{sharpe:.3f}",
                    f"{sortino:.3f}",
                    f"R$ {max_loss:,.0f} ({max_loss/pl*100:.2f}%)",
                    f"R$ {pnl.max():,.0f} ({pnl.max()/pl*100:.2f}%)"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("#### Percentis da Distribuição")
            percentiles = [0.5, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5]
            percentile_values = [np.percentile(pnl, p) for p in percentiles]
            
            percentile_df = pd.DataFrame({
                'Percentil': [f"{p}%" for p in percentiles],
                'P&L (R$)': [f"{val:,.0f}" for val in percentile_values],
                'Retorno (%)': [f"{val/pl*100:.2f}%" for val in percentile_values]
            })
            st.dataframe(percentile_df, use_container_width=True, hide_index=True)
    
    # Testes Estatísticos
    with st.expander("🔬 Testes Estatísticos"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("#### Teste de Normalidade")
            shapiro_stat, shapiro_p = stats.shapiro(port_ret[:min(5000, len(port_ret))])
            jarque_stat, jarque_p = stats.jarque_bera(port_ret)
            
            test_df = pd.DataFrame({
                'Teste': ['Shapiro-Wilk', 'Jarque-Bera'],
                'Estatística': [f"{shapiro_stat:.4f}", f"{jarque_stat:.4f}"],
                'p-valor': [f"{shapiro_p:.4f}", f"{jarque_p:.4f}"],
                'Resultado': [
                    "✅ Normal" if shapiro_p > 0.05 else "❌ Não-Normal",
                    "✅ Normal" if jarque_p > 0.05 else "❌ Não-Normal"
                ]
            })
            st.dataframe(test_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("#### Backtesting VaR")
            violations = (pnl <= -var).sum()
            expected_violations = n_sims * (1 - alpha)
            violation_rate = violations / n_sims
            
            # Teste de Kupiec (proporção de violações)
            if violations > 0:
                lr_stat = -2 * (violations * np.log(1-alpha) + (n_sims-violations) * np.log(alpha) -
                               violations * np.log(violation_rate) - (n_sims-violations) * np.log(1-violation_rate))
                kupiec_p = 1 - stats.chi2.cdf(lr_stat, 1)
            else:
                kupiec_p = 0
            
            backtest_df = pd.DataFrame({
                'Métrica': ['Violações Observadas', 'Violações Esperadas', 'Taxa de Violação', 'Teste Kupiec p-valor'],
                'Valor': [
                    f"{violations} ({violation_rate*100:.2f}%)",
                    f"{expected_violations:.1f} ({(1-alpha)*100:.2f}%)",
                    "✅ Adequado" if abs(violation_rate - (1-alpha)) < 0.01 else "⚠️ Revisar",
                    f"{kupiec_p:.4f} " + ("✅" if kupiec_p > 0.05 else "❌")
                ]
            })
            st.dataframe(backtest_df, use_container_width=True, hide_index=True)
        
        with col3:
            st.write("#### Qualidade da Matriz de Correlação")
            if usar_correlacao:
                eigenvalues = np.linalg.eigvals(corr_matrix)
                condition_number = np.max(eigenvalues) / np.min(eigenvalues)
                
                corr_quality_df = pd.DataFrame({
                    'Métrica': ['Menor Autovalor', 'Maior Autovalor', 'Número de Condição', 'Status'],
                    'Valor': [
                        f"{np.min(eigenvalues):.4f}",
                        f"{np.max(eigenvalues):.4f}",
                        f"{condition_number:.2f}",
                        "✅ Estável" if condition_number < 10 else "⚠️ Mal-condicionada" if condition_number < 100 else "❌ Instável"
                    ]
                })
                st.dataframe(corr_quality_df, use_container_width=True, hide_index=True)
    
    # EXPORTAR RESULTADOS
    st.write("---")
    st.subheader("💾 Exportar Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV com simulações
        results_df = pd.DataFrame({
            'Simulação': range(1, len(pnl)+1),
            'Tipo': all_labels,
            'P&L (R$)': pnl,
            'Retorno Portfolio (%)': port_ret * 100,
            'Retorno Ações (%)': R_total[:, 0] * 100,
            'Retorno RF (%)': R_total[:, 1] * 100,
            'Retorno Dólar (%)': R_total[:, 2] * 100
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Baixar Dados (CSV)",
            data=csv,
            file_name=f'var_simulation_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv',
            mime='text/csv'
        )
    
    with col2:
        # JSON com parâmetros
        params = {
            'data': datetime.datetime.now().isoformat(),
            'patrimonio': pl,
            'horizonte_dias': horizonte_dias,
            'nivel_confianca': nivel_conf,
            'num_simulacoes': n_sims,
            'alocacao': {'acoes': acoes, 'rf': juros, 'dolar': dolar},
            'volatilidades': {'acoes': vol_acoes, 'rf': vol_juros, 'dolar': vol_dolar},
            'correlacoes': corr_matrix.tolist() if usar_correlacao else None,
            'distribuicoes': {'acoes': dist_acoes, 'rf': dist_juros, 'dolar': dist_dolar},
            'resultados': {
                'var': var,
                'cvar': es,
                'sharpe': sharpe,
                'sortino': sortino,
                'prob_perda': prob_loss,
                'max_perda': max_loss
            }
        }
        
        json_str = json.dumps(params, indent=2, default=str)
        st.download_button(
            label="📥 Baixar Parâmetros (JSON)",
            data=json_str,
            file_name=f'var_params_{datetime.datetime.now():%Y%m%d_%H%M%S}.json',
            mime='application/json'
        )
    
    with col3:
        # PDF com relatório completo
        if generate_report:
            pdf_buffer = BytesIO()
            
            with PdfPages(pdf_buffer) as pdf:
                # Adicionar cada figura ao PDF
                for fig in figures:
                    pdf.savefig(fig, bbox_inches='tight')
                
                # Criar página de texto com resumo
                fig_text = plt.figure(figsize=(8.5, 11))
                fig_text.text(0.5, 0.95, 'RELATÓRIO DE VALUE AT RISK', 
                            fontsize=16, fontweight='bold', ha='center')
                
                report_text = f"""
                Data: {datetime.datetime.now():%d/%m/%Y %H:%M}
                
                PARÂMETROS DA SIMULAÇÃO
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                Patrimônio Líquido: R$ {pl:,.0f}
                Horizonte: {horizonte_dias} dias úteis
                Nível de Confiança: {nivel_conf}
                Simulações: {n_sims:,} ({n_mc:,} MC + {n_stress_total:,} Stress)
                
                ALOCAÇÃO DA CARTEIRA
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                Ações: {acoes}%
                Renda Fixa: {juros}%
                Dólar: {dolar}%
                Caixa: {100-total_aloc}%
                
                PARÂMETROS DE RISCO
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                Volatilidades (a.a.):
                  • Ações: {vol_acoes:.1f}%
                  • Renda Fixa: {vol_juros:.1f}%
                  • Dólar: {vol_dolar:.1f}%
                
                Correlações:
                  • Ações-RF: {corr_matrix[0,1]:.2f}
                  • Ações-Dólar: {corr_matrix[0,2]:.2f}
                  • RF-Dólar: {corr_matrix[1,2]:.2f}
                
                RESULTADOS PRINCIPAIS
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                VaR {nivel_conf}: R$ {var:,.0f} ({var/pl*100:.2f}% do PL)
                CVaR/ES: R$ {es:,.0f} ({es/pl*100:.2f}% do PL)
                Probabilidade de Perda: {prob_loss:.1f}%
                Sharpe Ratio: {sharpe:.3f}
                Sortino Ratio: {sortino:.3f}
                Máxima Perda: R$ {max_loss:,.0f}
                
                ANÁLISE DE CAUDA
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                Assimetria: {stats.skew(port_ret):.3f}
                Curtose: {stats.kurtosis(port_ret):.3f}
                Percentil 1%: R$ {np.percentile(pnl, 1):,.0f}
                Percentil 0.5%: R$ {np.percentile(pnl, 0.5):,.0f}
                """
                
                fig_text.text(0.1, 0.85, report_text, fontsize=10, 
                            verticalalignment='top', fontfamily='monospace')
                fig_text.axis('off')
                pdf.savefig(fig_text, bbox_inches='tight')
                
                # Metadados do PDF
                d = pdf.infodict()
                d['Title'] = 'Relatório VaR Monte Carlo'
                d['Author'] = 'Finhealth Risk Analytics'
                d['Subject'] = 'Análise de Value at Risk'
                d['Keywords'] = f'VaR, Monte Carlo, {nivel_conf}'
                d['CreationDate'] = datetime.datetime.now()
            
            pdf_buffer.seek(0)
            
            st.download_button(
                label="📥 Baixar Relatório (PDF)",
                data=pdf_buffer,
                file_name=f'var_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf',
                mime='application/pdf'
            )

# Rodapé com informações
st.write("---")
st.caption("""
💡 **Dicas:**
- Use correlações negativas entre ações e renda fixa para diversificação
- Distribuições t-Student capturam melhor eventos extremos
- Cenários de stress complementam a análise probabilística
- CVaR/ES fornece informação sobre perdas além do VaR
""")

# Informações técnicas
with st.sidebar:
    st.header("ℹ️ Informações")
    st.write("""
    ### Sobre o Modelo
    
    **VaR (Value at Risk)**: Perda máxima esperada com determinado nível de confiança.
    
    **CVaR/ES**: Perda média quando o VaR é excedido.
    
    **Distribuições**:
    - Normal: Modelo padrão
    - t-Student: Caudas pesadas
    - Lognormal: Assimetria
    - Mixture: Eventos extremos
    
    **Correlações**: Captura dependências entre ativos.
    
    **Cenários de Stress**: Eventos determinísticos baseados em crises históricas.
    """)
    
    st.write("---")
    st.write("🏢 **Finhealth Risk Analytics**")
    st.write("📅 " + datetime.datetime.now().strftime("%d/%m/%Y"))