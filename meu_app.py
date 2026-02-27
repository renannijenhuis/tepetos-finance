import streamlit as st
import google.generativeai as genai
import pandas as pd
import io
import os
import requests
import datetime
import math
import re  
import plotly.express as px
import sqlite3
from contextlib import closing
from PIL import Image 

# --- TENTA IMPORTAR O YFINANCE ---
try:
    import yfinance as yf
    YFINANCE_INSTALADO = True
except ImportError:
    YFINANCE_INSTALADO = False

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Tepetos' Finance V22.2", page_icon="🐥", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f4fbf7; }
    h1, h2, h3 { color: #006437 !important; font-family: 'Arial', sans-serif; }
    .stButton>button { background-color: #006437; color: white; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { background-color: #004d2a; color: #FFD700; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: transparent; border-radius: 4px 4px 0px 0px; padding-top: 10px; color: #006437; font-weight: bold;}
    .stTabs [aria-selected="true"] { background-color: #e2f0e9; border-bottom: 3px solid #006437;}
    .kinvo-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; border-top: 4px solid #006437; }
    .kinvo-card h4 { color: #555 !important; font-size: 16px; margin-bottom: 10px;}
    .kinvo-card h2 { color: #222 !important; font-size: 28px; margin: 0;}
    .pos-val { color: green; font-weight: bold; }
    .neg-val { color: red; font-weight: bold; }
    .fire-card { background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom: 15px;}
    .fire-card h4 { color: #2E7D32; margin:0 0 5px 0; }
    .status-card { background-color: white; padding: 15px; border-radius: 8px; border-left: 4px solid #006437; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 15px;}
    .metric-value { font-size: 22px; font-weight: bold; color: #222; margin: 0;}
    .metric-ideal { font-size: 13px; color: #006437; margin: 0; font-weight: bold;}
    .metric-desc { font-size: 11px; color: #666; margin-top: -5px; line-height: 1.2;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# ⚡ CONEXÃO SEGURA E PERFORMANCE DE BANCO 
# ==========================================
DB_NAME = "financas.db"

def get_conn():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False, timeout=20.0)
    conn.execute("PRAGMA journal_mode=WAL;") 
    return conn

def iniciar_banco():
    with closing(get_conn()) as conn:
        with conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS orcamento (conta TEXT, "Ativo?" INTEGER, "Titulo da Tabela" TEXT, "Tipo de Gasto" TEXT, "Categoria" TEXT, "Sub-Categoria" TEXT, "Meta Mensal (R$)" REAL)""")
            
            conn.execute("""CREATE TABLE IF NOT EXISTS fluxo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conta TEXT, "Data" TEXT, "Cartão/Pix" TEXT, "Descrição cobrança" TEXT, 
                "Natureza" TEXT, "Finalidade" TEXT, "Tipo de gasto" TEXT, "Categoria" TEXT, 
                "Sub-Categoria" TEXT, "Valor (R$)" REAL, "Origem" TEXT)""")
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fluxo_conta ON fluxo(conta)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fluxo_data ON fluxo(Data)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fluxo_origem ON fluxo(Origem)")
            
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(fluxo)")
            cols_f = [col[1] for col in cursor.fetchall()]
            if 'Natureza' not in cols_f: conn.execute("ALTER TABLE fluxo ADD COLUMN Natureza TEXT DEFAULT 'Despesa'")
            if 'Origem' not in cols_f: conn.execute("ALTER TABLE fluxo ADD COLUMN Origem TEXT DEFAULT 'Manual'")

            conn.execute("""CREATE TABLE IF NOT EXISTS carteira (conta TEXT, ticker TEXT, classe_ativo TEXT, quantidade REAL, preco_medio REAL, valor_aplicado REAL, saldo_bruto REAL)""")
            cursor.execute("PRAGMA table_info(carteira)")
            cols_c = [col[1] for col in cursor.fetchall()]
            if 'classe_ativo' not in cols_c:
                conn.execute("ALTER TABLE carteira ADD COLUMN classe_ativo TEXT DEFAULT 'Indefinido'")
                conn.execute("ALTER TABLE carteira ADD COLUMN valor_aplicado REAL DEFAULT 0.0")
                conn.execute("ALTER TABLE carteira ADD COLUMN saldo_bruto REAL DEFAULT 0.0")

iniciar_banco()

# ==========================================
# 🛡️ MOTOR ANTI-DUPLICIDADE E ANTI-TIMESTAMP
# ==========================================
def salvar_lote_seguro(conta, df_novo, nome_lote):
    df_proc = df_novo.copy()
    df_proc['conta'] = conta
    df_proc['Origem'] = nome_lote
    
    dt_parsed = pd.to_datetime(df_proc['Data'], errors='coerce', dayfirst=True)
    mask_nat = dt_parsed.isna() & df_proc['Data'].notna()
    if mask_nat.any():
        dt_parsed[mask_nat] = pd.to_datetime(df_proc.loc[mask_nat, 'Data'], errors='coerce')
    df_proc['Data'] = dt_parsed.dt.strftime('%d/%m/%Y')
    
    df_proc['Valor (R$)'] = converter_para_float(df_proc['Valor (R$)']).round(2)
    df_proc = df_proc.dropna(subset=['Data', 'Valor (R$)'])
    
    cols_oficiais = ["conta", "Data", "Cartão/Pix", "Descrição cobrança", "Natureza", "Finalidade", "Tipo de gasto", "Categoria", "Sub-Categoria", "Valor (R$)", "Origem"]
    for c in cols_oficiais:
        if c not in df_proc.columns:
            df_proc[c] = ""
    df_final = df_proc[cols_oficiais].copy()
    
    with closing(get_conn()) as conn:
        try: 
            df_banco = pd.read_sql("SELECT * FROM fluxo WHERE conta=?", conn, params=(conta,))
        except: 
            df_banco = pd.DataFrame()
            
        if not df_banco.empty:
            df_banco_outros = df_banco[df_banco['Origem'] != nome_lote].copy()
            if not df_banco_outros.empty:
                df_banco_outros['d_comp'] = df_banco_outros['Data'].astype(str).str.strip()
                df_banco_outros['desc_comp'] = df_banco_outros['Descrição cobrança'].astype(str).str.strip().str.upper()
                df_banco_outros['v_comp'] = pd.to_numeric(df_banco_outros['Valor (R$)'], errors='coerce').round(2)
                
                df_final['d_comp'] = df_final['Data'].astype(str).str.strip()
                df_final['desc_comp'] = df_final['Descrição cobrança'].astype(str).str.strip().str.upper()
                df_final['v_comp'] = df_final['Valor (R$)'].copy()
                
                merged = df_final.merge(df_banco_outros[['d_comp', 'desc_comp', 'v_comp']].drop_duplicates(), 
                                        on=['d_comp', 'desc_comp', 'v_comp'], 
                                        how='left', indicator=True)
                
                df_final = merged[merged['_merge'] == 'left_only'].drop(columns=['d_comp', 'desc_comp', 'v_comp', '_merge'])
            
        with conn:
            conn.execute("DELETE FROM fluxo WHERE conta=? AND Origem=?", (conta, nome_lote))
            if not df_final.empty:
                df_final.to_sql('fluxo', conn, if_exists='append', index=False)

# ==========================================
# ⚡ FUNÇÕES GERAIS FINANCEIRAS E MATEMÁTICAS
# ==========================================
@st.cache_resource
def carregar_excel_em_cache(file_bytes): return pd.ExcelFile(io.BytesIO(file_bytes))

def s_float(val):
    try: return float(val) if val is not None else 0.0
    except: return 0.0

@st.cache_data(ttl=3600)
def buscar_dados_ativo(ticker):
    if not YFINANCE_INSTALADO: return None, 1.0
    try:
        t_obj = yf.Ticker(ticker)
        hist = t_obj.history(period="1d")
        preco = hist['Close'].iloc[-1] if not hist.empty else None
        beta = t_obj.info.get('beta', 1.0) if t_obj.info else 1.0
        return preco, beta
    except: return None, 1.0

# 🔥 V22.2: Adicionadas Commodities do Agronegócio e Cobre 🔥
@st.cache_data(ttl=3600)
def buscar_indicadores_macro():
    indicadores = {
        "USD": {"v": 0.0, "d": 0.0}, "EUR": {"v": 0.0, "d": 0.0}, "GBP": {"v": 0.0, "d": 0.0}, 
        "SELIC": {"v": 0.0, "d": 0.0}, "IPCA": {"v": 0.0, "d": 0.0}, 
        "IBOV": {"v": 0.0, "d": 0.0}, "SP500": {"v": 0.0, "d": 0.0}, "NASDAQ": {"v": 0.0, "d": 0.0},
        "NIKKEI": {"v": 0.0, "d": 0.0}, "HANGSENG": {"v": 0.0, "d": 0.0},
        "STOXX600": {"v": 0.0, "d": 0.0}, "FTSE": {"v": 0.0, "d": 0.0},
        "BRENT": {"v": 0.0, "d": 0.0}, "GOLD": {"v": 0.0, "d": 0.0}, "BITCOIN": {"v": 0.0, "d": 0.0},
        "SOYBEAN": {"v": 0.0, "d": 0.0}, "CORN": {"v": 0.0, "d": 0.0}, "SUGAR": {"v": 0.0, "d": 0.0}, 
        "COFFEE": {"v": 0.0, "d": 0.0}, "COPPER": {"v": 0.0, "d": 0.0}
    }
    try:
        res = requests.get("https://economia.awesomeapi.com.br/last/USD-BRL,EUR-BRL,GBP-BRL", timeout=3)
        if res.status_code == 200:
            data = res.json()
            if "USDBRL" in data: indicadores["USD"] = {"v": float(data["USDBRL"]["bid"]), "d": float(data["USDBRL"]["pctChange"])}
            if "EURBRL" in data: indicadores["EUR"] = {"v": float(data["EURBRL"]["bid"]), "d": float(data["EURBRL"]["pctChange"])}
            if "GBPBRL" in data: indicadores["GBP"] = {"v": float(data["GBPBRL"]["bid"]), "d": float(data["GBPBRL"]["pctChange"])}
    except: pass
    try:
        res_selic = requests.get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json", timeout=3)
        if res_selic.status_code == 200: indicadores["SELIC"]["v"] = float(res_selic.json()[0]["valor"])
    except: pass
    try:
        res_ipca = requests.get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados/ultimos/1?formato=json", timeout=3)
        if res_ipca.status_code == 200: indicadores["IPCA"]["v"] = float(res_ipca.json()[0]["valor"])
    except: pass
    
    if YFINANCE_INSTALADO:
        bolsas_comodities = {
            "^BVSP": "IBOV", "^GSPC": "SP500", "^IXIC": "NASDAQ", 
            "^N225": "NIKKEI", "^HSI": "HANGSENG", "^STOXX": "STOXX600", "^FTSE": "FTSE",
            "BZ=F": "BRENT", "GC=F": "GOLD", "BTC-USD": "BITCOIN",
            "ZS=F": "SOYBEAN", "ZC=F": "CORN", "SB=F": "SUGAR", "KC=F": "COFFEE", "HG=F": "COPPER"
        }
        for tick, chave in bolsas_comodities.items():
            try:
                hist = yf.Ticker(tick).history(period="5d")
                if len(hist) >= 2:
                    p = hist['Close'].iloc[-1]
                    d = ((p - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                    indicadores[chave] = {"v": p, "d": d}
            except: pass
    return indicadores

def calcular_tepetos_score(pl, roe, div_patr, dy, cagr_lucro=0):
    score = 0
    if pl <= 0: score += 0 
    elif 0 < pl < 10: score += 25
    elif 10 <= pl <= 15: score += 18
    elif 15 < pl <= 25: score += 10
    else: score += 5
    if roe > 0.20: score += 25
    elif roe > 0.15: score += 18
    elif roe > 0.10: score += 10
    elif roe > 0: score += 5
    if 0 <= div_patr < 50: score += 20
    elif div_patr < 100: score += 15
    elif div_patr >= 200: score += 0
    else: score += 10 
    if dy > 0.06: score += 15
    elif dy > 0.04: score += 10
    else: score += 5
    if cagr_lucro > 0.10: score += 15
    else: score += 5
    return score

def classificar_ativo(pl, roe, dy):
    if roe > 0.15 and (pl > 15 or pl <= 0): return "🚀 Crescimento"
    if dy > 0.05: return "🐄 Vaca Leiteira"
    if 0 < pl < 12 and roe > 0.10: return "💎 Valor Descontado"
    return "⚖️ Misto"

def simular_juros_compostos(patrimonio_inicial, aporte_mensal, taxa_anual, anos):
    taxa_mensal = (1 + taxa_anual/100) ** (1/12) - 1
    meses = int(anos * 12)
    evolucao = []
    saldo = patrimonio_inicial
    total_investido = patrimonio_inicial
    for mes in range(1, meses + 1):
        saldo = saldo * (1 + taxa_mensal) + aporte_mensal
        total_investido += aporte_mensal
        if mes % 12 == 0 or mes == meses:
            evolucao.append({"Ano": mes/12, "Total Investido": total_investido, "Patrimônio Total": saldo})
    return pd.DataFrame(evolucao)

def converter_para_float(coluna):
    if pd.api.types.is_numeric_dtype(coluna): return coluna
    def limpa_valor(val):
        if pd.isna(val): return 0.0
        val = str(val).replace('R$', '').strip()
        if val in ['nan', 'None', '', 'NaN']: return 0.0
        val = re.sub(r'\.(?=\d{3})', '', val) 
        val = val.replace(',', '.')
        try: return float(val)
        except: return 0.0
    return coluna.apply(limpa_valor)

def formata_moeda(valor):
    try: return f"R$ {float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except: return "R$ 0,00"

def encontrar_modelo_flash():
    try:
        modelos = genai.list_models()
        validos = []
        for m in modelos:
            metodos = getattr(m, 'supported_generation_methods', [])
            if metodos is not None and 'generateContent' in metodos:
                validos.append(m.name)
        flash = [m for m in validos if 'flash' in m.lower()]
        return flash[0] if flash else (validos[0] if validos else "gemini-1.5-pro-latest")
    except: 
        return "gemini-1.5-flash"

def calcular_valuation_dcf_matematico(ticker, is_fii=False):
    try:
        tick_yf = f"{ticker}.SA" if not ticker.endswith(".SA") and len(ticker) <= 6 else ticker
        tk = yf.Ticker(tick_yf)
        info = tk.info
        
        if not info or ('currentPrice' not in info and 'previousClose' not in info):
            return None, 0, "Dados de cotação não encontrados na API."
            
        preco_atual = float(info.get('currentPrice', info.get('previousClose', 0.0)))
        if preco_atual <= 0: return None, 0, "Preço inválido."
        
        is_br = tick_yf.endswith('.SA')
        
        if is_fii:
            div_rate = float(info.get('trailingAnnualDividendRate', 0))
            if div_rate <= 0:
                dy_raw = float(info.get('dividendYield', info.get('trailingAnnualDividendYield', 0)))
                div_rate = preco_atual * dy_raw if dy_raw > 0 else preco_atual * 0.08
            
            wacc_base = 0.11 
            g_base = 0.015   
            
            cenarios = [
                {"Nome": "🔴 Pessimista", "Retorno Exigido": wacc_base + 0.02, "G": g_base - 0.01},
                {"Nome": "🟡 Base", "Retorno Exigido": wacc_base, "G": g_base},
                {"Nome": "🟢 Otimista", "Retorno Exigido": wacc_base - 0.02, "G": g_base + 0.01}
            ]
            
            resultados = []
            for c in cenarios:
                w = c["Retorno Exigido"]
                g = c["G"]
                pj = (div_rate * (1 + g)) / (w - g) if w > g else preco_atual
                if pj <= 0: pj = preco_atual * 0.5
                upside = ((pj / preco_atual) - 1) * 100
                resultados.append({
                    "Cenário": c["Nome"], "Retorno Exigido": f"{w*100:.1f}%", "Cresc. Dividendo (g)": f"{g*100:.1f}%",
                    "Preço Justo": f"R$ {pj:.2f}", "Upside / Margem": f"{upside:.1f}%"
                })
            return pd.DataFrame(resultados), preco_atual, ""
            
        else:
            lpa = float(info.get('trailingEps', 0))
            vpa = float(info.get('bookValue', 0))
            
            if lpa <= 0: 
                cfps = vpa * 0.06 if vpa > 0 else preco_atual * 0.05
            else: 
                cfps = lpa
                
            wacc_base = 0.13 if is_br else 0.09
            g_base = 0.04 if is_br else 0.025
            
            cr_est = float(info.get('revenueGrowth', 0.05))
            if pd.isna(cr_est) or cr_est <= 0 or cr_est > 0.20: cr_est = 0.06
            
            cenarios = [
                {"Nome": "🔴 Pessimista", "WACC": wacc_base + 0.02, "G": g_base - 0.01, "Cresc": cr_est - 0.03},
                {"Nome": "🟡 Base", "WACC": wacc_base, "G": g_base, "Cresc": cr_est},
                {"Nome": "🟢 Otimista", "WACC": wacc_base - 0.02, "G": g_base + 0.01, "Cresc": cr_est + 0.03}
            ]
            
            resultados = []
            for c in cenarios:
                w = c["WACC"]
                g = c["G"]
                cr = c["Cresc"]
                fluxo = cfps
                vp_f = 0
                for i in range(1, 6):
                    fluxo *= (1 + cr)
                    vp_f += fluxo / ((1 + w)**i)
                vt = (fluxo * (1 + g)) / (w - g) if w > g else (fluxo * 10)
                vp_vt = vt / ((1 + w)**5)
                pj = vp_f + vp_vt
                
                if pj <= 0: pj = preco_atual * 0.5
                upside = ((pj / preco_atual) - 1) * 100
                resultados.append({
                    "Cenário": c["Nome"], "WACC (%)": f"{w*100:.1f}%", "Cresc. Perpétuo (g)": f"{g*100:.1f}%",
                    "Preço Justo": f"R$ {pj:.2f}" if is_br else f"$ {pj:.2f}", "Upside / Margem": f"{upside:.1f}%"
                })
            return pd.DataFrame(resultados), preco_atual, ""
    except Exception as e:
        return None, 0, str(e)

def obter_orcamento_padrao():
    dados = [
        [True, "Receitas", "Receitas", "Receitas", "Salário Líquido", 24410.65],
        [True, "Receitas", "Receitas", "Receitas", "13º salário", 791.67],
        [True, "Receitas", "Receitas", "Receitas", "Férias", 263.89],
        [True, "Investimentos", "Investimentos de Longo Prazo", "Investimentos", "Tesouro Direto", 600.0],
        [True, "Investimentos", "Investimentos de Longo Prazo", "Investimentos", "Renda Variavel", 1000.0],
        [True, "Investimentos", "Investimentos de Longo Prazo", "Investimentos", "Caixinha Casa/Carro", 500.0],
        [True, "Investimentos", "Investimentos de Longo Prazo", "Investimentos", "Caixinha Viagem", 1000.0],
        [True, "Despesas Fixas (Manutenção Mensal)", "Fixas", "Habitação", "Prestação da casa", 6330.0],
        [True, "Despesas Fixas (Manutenção Mensal)", "Fixas", "Habitação", "Diarista", 940.0],
        [True, "Despesas Fixas (Manutenção Mensal)", "Fixas", "Transporte", "Seguro do carro", 375.53],
        [True, "Despesas Variáveis", "Variáveis", "Habitação", "Luz", 320.0],
        [True, "Despesas Variáveis", "Variáveis", "Habitação", "Água", 330.0],
        [True, "Despesas Variáveis", "Variáveis", "Alimentação", "Supermercado", 1800.0],
        [True, "Despesas Extras", "Extras", "Saúde", "Médico", 425.0],
        [True, "Despesas Extras", "Extras", "Bento", "Fraldas", 800.0],
        [True, "Gastos Livres", "Adicionais", "Lazer", "Viagem anual", 1000.0],
        [True, "Gastos Livres", "Adicionais", "Lazer", "Restaurantes/IFOOD", 1000.0],
        [True, "GASTOS NÃO IDENTIFICADOS", "GASTOS NÃO IDENTIFICADOS", "Gastos não identificados", "????", 0.0]
    ]
    return pd.DataFrame(dados, columns=["Ativo?", "Titulo da Tabela", "Tipo de Gasto", "Categoria", "Sub-Categoria", "Meta Mensal (R$)"])

# ==========================================
# 👤 GESTÃO DE MÚLTIPLAS CONTAS 
# ==========================================
st.sidebar.header("👤 Perfil da Conta")
with closing(get_conn()) as conn:
    try: perfis_salvos = pd.read_sql("SELECT DISTINCT conta FROM orcamento", conn)['conta'].tolist()
    except: perfis_salvos = []

if "Tepetos" not in perfis_salvos: perfis_salvos.insert(0, "Tepetos")
perfis_salvos.append("+ Adicionar Nova Conta...")
conta_selecionada = st.sidebar.selectbox("Selecione quem está usando:", perfis_salvos)

if conta_selecionada == "+ Adicionar Nova Conta...":
    nova_conta = st.sidebar.text_input("Nome do Novo Perfil:")
    if st.sidebar.button("Criar Conta"):
        if nova_conta:
            df_padrao = obter_orcamento_padrao()
            df_padrao['conta'] = nova_conta
            with closing(get_conn()) as conn:
                with conn: df_padrao.to_sql('orcamento', conn, if_exists='append', index=False)
            st.rerun() 
    st.stop() 

st.sidebar.success(f"🟢 Logado como: {conta_selecionada}")

caminho_secrets = ".streamlit/secrets.toml"
chave_salva = ""
try:
    if os.path.exists(caminho_secrets):
        with open(caminho_secrets, "r") as f: chave_salva = f.read().replace('GOOGLE_API_KEY="', '').replace('"', '').strip()
except: pass

with st.sidebar:
    st.divider()
    st.header("⚙️ Configuração IA")
    api_key = st.text_input("🔑 API Key do Google:", type="password", value=chave_salva)
    if api_key and api_key != chave_salva:
        if not os.path.exists(".streamlit"): os.makedirs(".streamlit")
        with open(caminho_secrets, "w") as f: f.write(f'GOOGLE_API_KEY="{api_key}"')
        
    st.divider()
    st.markdown("### 🧹 Gestão do Cofre (Lotes)")
    
    with closing(get_conn()) as conn:
        try: origens_db = pd.read_sql("SELECT DISTINCT Origem FROM fluxo WHERE conta=?", conn, params=(conta_selecionada,))['Origem'].tolist()
        except: origens_db = []
        
    if origens_db:
        lote_excluir = st.selectbox("Excluir Lote Específico:", ["Selecione..."] + origens_db)
        if lote_excluir != "Selecione...":
            if st.button(f"🗑️ Apagar Lote"):
                with closing(get_conn()) as conn:
                    with conn: conn.execute("DELETE FROM fluxo WHERE conta=? AND Origem=?", (conta_selecionada, lote_excluir))
                st.success("Lote removido do banco de dados!")
                st.rerun()
                
    st.write("")
    if st.button("🚨 APAGAR TODO O FLUXO", type="primary"):
        with closing(get_conn()) as conn:
            with conn: conn.execute("DELETE FROM fluxo WHERE conta=?", (conta_selecionada,))
        st.success("Cofre de fluxo zerado! Base limpa.")
        st.rerun()

# ==========================================
# HEADER PRINCIPAL 
# ==========================================
st.title("🐥 Tepetos' Finance 22.2 🐷💚")
st.markdown(f"**Sistema ERP Institucional & Terminal Status Invest - Ativo: {conta_selecionada}**")

st.sidebar.divider()
st.sidebar.markdown("### 🏢 Módulos do Sistema")
modulo_selecionado = st.sidebar.radio("Navegue por área:", ["💳 Tesouraria & Despesas", "💼 Wealth Management"])

# =========================================================================================
# MÓDULO 1: TESOURARIA & DESPESAS
# =========================================================================================
if modulo_selecionado == "💳 Tesouraria & Despesas":
    
    tab_orcamento, tab_dashboard, tab_import, tab_cupom = st.tabs([
        "🎯 Meu Orçamento", "📈 Dashboard do Dono", "🚀 Importar Extratos e Planilhas", "🧾 Leitor de Cupons"
    ])

    with tab_orcamento:
        st.header("🎯 Definição de Metas de Gastos e Receitas")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1: meta_fixos = st.number_input("🔴 Teto: Despesas Fixas (%)", value=60)
        with col_r2: meta_livres = st.number_input("🟡 Teto: Estilo de Vida (%)", value=20)
        with col_r3: meta_invest = st.number_input("🟢 Piso: Investimentos (%)", value=20)
        st.divider()

        with closing(get_conn()) as conn:
            try:
                df_orcamento = pd.read_sql("SELECT * FROM orcamento WHERE conta=?", conn, params=(conta_selecionada,))
                df_orcamento = df_orcamento.drop(columns=['conta'])
                if df_orcamento.empty: df_orcamento = obter_orcamento_padrao()
                df_orcamento['Ativo?'] = df_orcamento['Ativo?'].astype(bool)
            except: df_orcamento = obter_orcamento_padrao()

        tabelas_editadas = []
        ordem_titulos = ["Receitas", "Investimentos", "Despesas Fixas (Manutenção Mensal)", "Despesas Variáveis", "Despesas Extras", "Gastos Livres", "GASTOS NÃO IDENTIFICADOS"]

        for titulo in ordem_titulos:
            df_tipo = df_orcamento[df_orcamento["Titulo da Tabela"] == titulo].copy()
            if not df_tipo.empty:
                st.markdown(f"### 📋 {titulo}")
                df_view = df_tipo.drop(columns=["Titulo da Tabela"])
                editado = st.data_editor(df_view, num_rows="dynamic", use_container_width=True, key=f"tbl_{titulo}_{conta_selecionada}", column_config={"Meta Mensal (R$)": st.column_config.NumberColumn("Meta Mensal (R$)", format="R$ %.2f")})
                editado["Titulo da Tabela"] = titulo 
                editado = editado[["Ativo?", "Titulo da Tabela", "Tipo de Gasto", "Categoria", "Sub-Categoria", "Meta Mensal (R$)"]]
                tabelas_editadas.append(editado)
                
        titulos_existentes = df_orcamento["Titulo da Tabela"].dropna().unique()
        for titulo in titulos_existentes:
            if titulo not in ordem_titulos:
                df_tipo = df_orcamento[df_orcamento["Titulo da Tabela"] == titulo].copy()
                st.markdown(f"### 📋 {titulo}")
                df_view = df_tipo.drop(columns=["Titulo da Tabela"])
                editado = st.data_editor(df_view, num_rows="dynamic", use_container_width=True, key=f"tbl_extra_{titulo}_{conta_selecionada}", column_config={"Meta Mensal (R$)": st.column_config.NumberColumn("Meta Mensal (R$)", format="R$ %.2f")})
                editado["Titulo da Tabela"] = titulo 
                editado = editado[["Ativo?", "Titulo da Tabela", "Tipo de Gasto", "Categoria", "Sub-Categoria", "Meta Mensal (R$)"]]
                tabelas_editadas.append(editado)

        st.write("")
        if st.button(f"💾 Salvar Orçamento Seguro", type="primary"):
            df_salvar = pd.concat(tabelas_editadas, ignore_index=True)
            df_salvar = df_salvar[df_salvar["Ativo?"] == True].copy()
            df_salvar['conta'] = conta_selecionada
            with closing(get_conn()) as conn:
                with conn:
                    conn.execute("DELETE FROM orcamento WHERE conta=?", (conta_selecionada,))
                    df_salvar.to_sql('orcamento', conn, if_exists='append', index=False)
            st.success("Orçamento salvo com segurança de alto nível no Cofre Digital!")

        st.markdown("---")
        st.subheader("📊 Resultado do Orçamento Previsto")
        df_ao_vivo = pd.concat(tabelas_editadas, ignore_index=True)
        df_ao_vivo = df_ao_vivo[df_ao_vivo["Ativo?"] == True]
        df_rec = df_ao_vivo[df_ao_vivo["Titulo da Tabela"] == "Receitas"]
        df_sai = df_ao_vivo[df_ao_vivo["Titulo da Tabela"] != "Receitas"]
        tot_receita = pd.to_numeric(df_rec["Meta Mensal (R$)"], errors='coerce').sum()
        tot_saida = pd.to_numeric(df_sai["Meta Mensal (R$)"], errors='coerce').sum()
        saldo_livre = tot_receita - tot_saida
        
        col_resumo1, col_resumo2 = st.columns([1, 1])
        with col_resumo1:
            st.markdown(f"**💰 Receita:** {formata_moeda(tot_receita)}")
            st.markdown(f"**💳 Despesas/Invest:** {formata_moeda(tot_saida)}")
            st.markdown(f"**✅ Saldo:** <span style='color:{'green' if saldo_livre >= 0 else 'red'};'>{formata_moeda(saldo_livre)}</span>", unsafe_allow_html=True)
            
            if tot_receita > 0 and not df_sai.empty:
                df_sai_calc = df_sai.copy()
                df_sai_calc["Meta Mensal (R$)"] = pd.to_numeric(df_sai_calc["Meta Mensal (R$)"], errors='coerce').fillna(0)
                resumo_estatico = df_sai_calc.groupby("Titulo da Tabela")["Meta Mensal (R$)"].sum().reset_index()
                resumo_estatico["% da Receita"] = (resumo_estatico["Meta Mensal (R$)"] / tot_receita) * 100
                resumo_estatico["% da Receita"] = resumo_estatico["% da Receita"].map("{:.1f}%".format)
                resumo_estatico["Meta Mensal (R$)"] = resumo_estatico["Meta Mensal (R$)"].apply(formata_moeda)
                st.dataframe(resumo_estatico, use_container_width=True, hide_index=True)
                
        with col_resumo2:
            if not df_sai.empty and tot_receita > 0:
                resumo_estatico_graf = df_sai.groupby("Titulo da Tabela")["Meta Mensal (R$)"].sum().reset_index()
                fig_orc = px.pie(resumo_estatico_graf, values='Meta Mensal (R$)', names='Titulo da Tabela', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_orc, use_container_width=True)

    with tab_dashboard:
        st.header("📈 Dashboard Executivo (O Painel do Dono)")
        
        df_consolidado = pd.DataFrame()
        with closing(get_conn()) as conn:
            try: df_consolidado = pd.read_sql("SELECT * FROM fluxo WHERE conta=?", conn, params=(conta_selecionada,))
            except: pass

        if not df_consolidado.empty:
            df_consolidado['valor_calc'] = converter_para_float(df_consolidado["Valor (R$)"])
            df_consolidado['data_calc'] = pd.to_datetime(df_consolidado["Data"], dayfirst=True, errors='coerce')
            df_consolidado = df_consolidado.dropna(subset=['valor_calc', 'data_calc'])
            
            if 'Natureza' not in df_consolidado.columns: df_consolidado['Natureza'] = 'Despesa'
            if 'Origem' not in df_consolidado.columns: df_consolidado['Origem'] = 'Legado Antigo'
            df_consolidado['Natureza'] = df_consolidado['Natureza'].fillna('Despesa')
            
            origens_disponiveis = df_consolidado['Origem'].unique().tolist()
            with st.expander("📂 Controle de Lotes (Filtre suas importações)", expanded=False):
                origens_selecionadas = st.multiselect("Mostrando dados de:", origens_disponiveis, default=origens_disponiveis)
            
            df_filtrado_origem = df_consolidado[df_consolidado['Origem'].isin(origens_selecionadas)].copy()
            
            if not df_filtrado_origem.empty:
                df_filtrado_origem['Mes_Ano'] = df_filtrado_origem['data_calc'].dt.strftime('%m/%Y')
                anos = sorted(df_filtrado_origem['data_calc'].dt.year.dropna().astype(int).unique().tolist(), reverse=True)
                
                col_filtro1, col_filtro2 = st.columns(2)
                with col_filtro1: ano_escolhido = st.selectbox("📅 Ano de Análise:", ["Todos os Anos"] + anos)
                
                if ano_escolhido != "Todos os Anos":
                    df_ano = df_filtrado_origem[df_filtrado_origem['data_calc'].dt.year == ano_escolhido]
                    meses_str = [f"{m:02d}" for m in sorted(df_ano['data_calc'].dt.month.dropna().astype(int).unique().tolist())]
                    with col_filtro2: mes_escolhido = st.selectbox("📅 Mês de Análise:", ["Ano Inteiro"] + meses_str)
                    if mes_escolhido != "Ano Inteiro":
                        df_filtrado = df_ano[df_ano['data_calc'].dt.month == int(mes_escolhido)]
                        fator_meta = 1
                    else:
                        df_filtrado, fator_meta = df_ano, 12
                else:
                    with col_filtro2: st.info("Mostrando a vida inteira.")
                    df_filtrado, fator_meta = df_filtrado_origem, None
                
                tot_rec = df_filtrado[df_filtrado['Natureza'].str.contains('Receita', case=False, na=False)]['valor_calc'].sum()
                tot_desp = df_filtrado[df_filtrado['Natureza'].str.contains('Despesa', case=False, na=False)]['valor_calc'].sum()
                tot_invest = df_filtrado[df_filtrado['Natureza'].str.contains('Investimento', case=False, na=False)]['valor_calc'].sum()
                
                res_operacional = tot_rec - tot_desp
                taxa_poupanca = (tot_invest / tot_rec * 100) if tot_rec > 0 else 0
                
                st.markdown("### 🏛️ O Painel do Dono")
                col_r, col_s, col_p = st.columns(3)
                col_r.metric("💰 Receitas Totais", formata_moeda(tot_rec))
                col_s.metric("🔴 Custo de Vida (Despesas)", formata_moeda(tot_desp))
                col_p.metric("🚀 Investimentos", formata_moeda(tot_invest))
                
                st.markdown(f"**🟢 Resultado Operacional (Livre p/ Investir/Gastar):** <span style='color:{'green' if res_operacional>=0 else 'red'}; font-size:18px; font-weight:bold;'>{formata_moeda(res_operacional)}</span>", unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 🎯 Termômetro 60/20/20 (Saúde do Orçamento)")
                
                df_despesas = df_filtrado[df_filtrado['Natureza'].str.contains('Despesa', case=False, na=False)]
                mask_essencial = df_despesas['Categoria'].astype(str).str.contains('Habitação|Alimentação|Saúde|Transporte|Fixo', case=False) | df_despesas['Tipo de gasto'].astype(str).str.contains('Fixa|Variável', case=False)
                gasto_essencial = df_despesas[mask_essencial]['valor_calc'].sum()
                mask_desejo = ~mask_essencial 
                gasto_desejo = df_despesas[mask_desejo]['valor_calc'].sum()
                
                pct_essencial = (gasto_essencial / tot_rec * 100) if tot_rec > 0 else 0
                pct_desejo = (gasto_desejo / tot_rec * 100) if tot_rec > 0 else 0
                pct_invest = taxa_poupanca
                
                c_ess, c_des, c_inv = st.columns(3)
                with c_ess:
                    cor = "green" if pct_essencial <= meta_fixos else ("orange" if pct_essencial <= (meta_fixos+10) else "red")
                    st.markdown(f"<div class='fire-card'><h4>🏠 Custos Essenciais</h4><h2 style='color:{cor}; margin:0;'>{pct_essencial:.1f}%</h2><p style='margin:0; font-size:12px;'>Meta: Até {meta_fixos}% ({formata_moeda(gasto_essencial)})</p></div>", unsafe_allow_html=True)
                with c_des:
                    cor = "green" if pct_desejo <= meta_livres else ("orange" if pct_desejo <= (meta_livres+10) else "red")
                    st.markdown(f"<div class='fire-card'><h4>🎮 Estilo de Vida</h4><h2 style='color:{cor}; margin:0;'>{pct_desejo:.1f}%</h2><p style='margin:0; font-size:12px;'>Meta: Até {meta_livres}% ({formata_moeda(gasto_desejo)})</p></div>", unsafe_allow_html=True)
                with c_inv:
                    cor = "green" if pct_invest >= meta_invest else ("orange" if pct_invest >= (meta_invest-10) else "red")
                    st.markdown(f"<div class='fire-card'><h4>💎 Investimentos</h4><h2 style='color:{cor}; margin:0;'>{pct_invest:.1f}%</h2><p style='margin:0; font-size:12px;'>Meta: Mínimo {meta_invest}% ({formata_moeda(tot_invest)})</p></div>", unsafe_allow_html=True)

                st.markdown("---")
                col_graf_pizza, col_graf_barra = st.columns(2)
                realizado = df_despesas.groupby("Categoria")['valor_calc'].sum().reset_index().sort_values(by='valor_calc', ascending=False)
                
                with col_graf_pizza:
                    if not realizado.empty: st.plotly_chart(px.pie(realizado, values='valor_calc', names='Categoria', hole=0.4, title="Distribuição do Custo de Vida", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
                with col_graf_barra:
                    if not realizado.empty: st.plotly_chart(px.bar(realizado.head(10).sort_values(by='valor_calc', ascending=True), x='valor_calc', y='Categoria', orientation='h', title="Os 10 Maiores Ladrões do Orçamento", color_discrete_sequence=['#006437']), use_container_width=True)

                with closing(get_conn()) as conn:
                    try:
                        df_orc_db = pd.read_sql("SELECT * FROM orcamento WHERE conta=?", conn, params=(conta_selecionada,))
                        if df_orc_db.empty:
                            df_orc_db = obter_orcamento_padrao()
                            
                        if fator_meta is None:
                            st.warning("⚠️ Para visualizar a Batalha do Orçamento e falar com o CFO, selecione um Ano e um Mês específico no topo do Dashboard.")
                        elif not df_orc_db.empty:
                            st.markdown("---")
                            st.subheader(f"🎯 A Batalha Geral: Orçado vs Realizado (360º)")
                            st.caption("Visão unificada das suas Receitas, Investimentos e Despesas.")
                            
                            real_geral = df_filtrado.groupby("Categoria")['valor_calc'].sum().reset_index()
                            real_geral.rename(columns={'valor_calc': 'Realizado (R$)'}, inplace=True)
                            
                            df_orc_db["Ativo?"] = df_orc_db["Ativo?"].apply(lambda x: True if str(x).lower() in ['1', 'true', 'sim', 'yes'] else False)
                            metas_ativas = df_orc_db[df_orc_db["Ativo?"] == True].copy()
                            
                            metas_ativas["Meta Mensal (R$)"] = pd.to_numeric(metas_ativas["Meta Mensal (R$)"], errors='coerce').fillna(0)
                            metas_grp = metas_ativas.groupby(["Tipo de Gasto", "Categoria"])["Meta Mensal (R$)"].sum().reset_index()
                            metas_grp["Meta Calculada (R$)"] = metas_grp["Meta Mensal (R$)"] * fator_meta
                            
                            df_comp = pd.merge(metas_grp, real_geral, on="Categoria", how="left").fillna(0)
                            
                            def calc_folga(row):
                                tg = str(row.get('Tipo de Gasto', '')).lower()
                                m = float(row.get('Meta Calculada (R$)', 0))
                                r = float(row.get('Realizado (R$)', 0))
                                if "receita" in tg or "invest" in tg: return r - m
                                else: return m - r
                            
                            def calc_uso(row):
                                m = float(row.get('Meta Calculada (R$)', 0))
                                r = float(row.get('Realizado (R$)', 0))
                                if m == 0: return 0.0
                                return (r / m) * 100.0
                            
                            df_comp['Saldo (Folga/Falta)'] = df_comp.apply(calc_folga, axis=1)
                            df_comp['% Atingimento'] = df_comp.apply(calc_uso, axis=1)
                            
                            st.dataframe(
                                df_comp[["Tipo de Gasto", "Categoria", "Meta Calculada (R$)", "Realizado (R$)", "Saldo (Folga/Falta)", "% Atingimento"]].sort_values(by=["Tipo de Gasto", "Categoria"]), 
                                use_container_width=True, hide_index=True,
                                column_config={
                                    "Meta Calculada (R$)": st.column_config.NumberColumn(format="R$ %.2f"), 
                                    "Realizado (R$)": st.column_config.NumberColumn(format="R$ %.2f"), 
                                    "Saldo (Folga/Falta)": st.column_config.NumberColumn(format="R$ %.2f"), 
                                    "% Atingimento": st.column_config.ProgressColumn("Atingimento", format="%.1f%%", min_value=0, max_value=100)
                                }
                            )
                            
                            st.markdown("### 🤖 O Veredito do Seu CFO (Inteligência Artificial)")
                            st.caption("O robô avaliará seu comportamento cruzando suas metas de receitas, aportes e despesas.")
                            
                            if st.button("🧠 Gerar Parecer do Orçamento", type="primary", key="btn_cfo"):
                                if not api_key: 
                                    st.error("Insira a API Key do Google na barra lateral!")
                                else:
                                    with st.spinner("O CFO está analisando o seu fluxo de caixa..."):
                                        try:
                                            texto_orc = df_comp.to_csv(index=False, sep="|")
                                            genai.configure(api_key=api_key)
                                            modelo_ia = encontrar_modelo_flash()
                                            
                                            prompt_cfo = f"""
                                            Atue como meu CFO (Diretor Financeiro Sênior) e Conselheiro Pessoal.
                                            Abaixo está a tabela exata com as minhas metas orçamentárias (Receitas, Investimentos e Despesas) versus o que eu realmente realizei/gastei neste mês/ano:
                                            
                                            {texto_orc}
                                            
                                            Escreva um veredito direto, elegante e altamente motivacional avaliando meu desempenho:
                                            1) Elogie com entusiasmo se eu bati as metas de Receita e Investimentos e se economizei nas despesas essenciais.
                                            2) Puxe a orelha (com carinho e dicas práticas) naquelas despesas onde eu estourei a meta de forma preocupante.
                                            3) Traga uma reflexão final inspiradora sobre como essa disciplina constrói a minha independência financeira e meus maiores sonhos.
                                            
                                            Use emojis corporativos. Assine como: 'Seu CFO de Confiança - Tepeto'.
                                            """
                                            res = genai.GenerativeModel(modelo_ia).generate_content(prompt_cfo)
                                            st.markdown(f"<div style='background-color: white; padding: 25px; border-left: 5px solid #006437; border-radius: 8px;'>{res.text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                                        except Exception as e:
                                            st.error(f"Erro ao gerar parecer com a IA: {e}")
                    except Exception as e: 
                        st.error(f"Erro ao processar a tabela orçamentária: {e}")

                st.markdown("---")
                st.markdown("### 🕵️ Auditoria de Lançamentos (Deletar ou Editar)")
                st.info("💡 Marque a caixinha 'Excluir?' nas linhas que deseja apagar e clique no botão vermelho abaixo.")
                
                if "id" in df_filtrado.columns:
                    df_auditoria = df_filtrado[["id", "Data", "Descrição cobrança", "Natureza", "Categoria", "Valor (R$)", "Origem"]].copy()
                    df_auditoria.insert(0, "Excluir?", False)
                    
                    df_editado_auditoria = st.data_editor(
                        df_auditoria, 
                        use_container_width=True, 
                        hide_index=True,
                        disabled=["id", "Origem"], 
                        column_config={
                            "id": None, 
                            "Valor (R$)": st.column_config.NumberColumn(format="R$ %.2f")
                        }
                    )
                    
                    col_btn_1, col_btn_2 = st.columns(2)
                    with col_btn_1:
                        linhas_para_excluir = df_editado_auditoria[df_editado_auditoria["Excluir?"] == True]["id"].tolist()
                        if st.button(f"🗑️ Excluir Selecionados ({len(linhas_para_excluir)})", type="primary") and len(linhas_para_excluir) > 0:
                            with closing(get_conn()) as conn:
                                with conn:
                                    placeholders = ','.join('?' for _ in linhas_para_excluir)
                                    conn.execute(f"DELETE FROM fluxo WHERE id IN ({placeholders})", tuple(linhas_para_excluir))
                            st.success(f"{len(linhas_para_excluir)} transações removidas para sempre!")
                            st.rerun()
                            
                    with col_btn_2:
                        if st.button("💾 Salvar Edições nos Nomes/Valores"):
                            linhas_para_salvar = df_editado_auditoria[df_editado_auditoria["Excluir?"] == False]
                            with closing(get_conn()) as conn:
                                with conn:
                                    for _, row in linhas_para_salvar.iterrows():
                                        conn.execute("""
                                            UPDATE fluxo 
                                            SET "Data"=?, "Descrição cobrança"=?, "Natureza"=?, "Categoria"=?, "Valor (R$)"=? 
                                            WHERE id=?
                                        """, (row["Data"], row["Descrição cobrança"], row["Natureza"], row["Categoria"], row["Valor (R$)"], row["id"]))
                            st.success("Edições salvas!")
                            st.rerun()
                else:
                    df_auditoria = df_filtrado[["Data", "Descrição cobrança", "Natureza", "Categoria", "Valor (R$)", "Origem"]].copy()
                    st.warning("⚠️ Seu Banco de Dados é de uma versão anterior. Para habilitar a edição e exclusão individual de linhas, zere o cofre em '🚨 APAGAR TODO O FLUXO' (barra lateral) e importe as planilhas novamente.")
                    df_auditoria = df_auditoria.sort_values(by="Data", ascending=False)
                    st.dataframe(df_auditoria, use_container_width=True, hide_index=True)

            else: st.warning("As caixinhas de Lote selecionadas não possuem dados.")
        else:
            st.info("Nenhuma transação encontrada no seu Cofre Digital.")

    with tab_import:
        st.markdown("### 📥 1. Importar Excel (Planilha Base/Legado)")
        with st.expander("Clique aqui para subir sua planilha", expanded=False):
            nome_lote_legado = st.text_input("Nome deste Lote:", "Planilha Base", key="lote_leg")
            arquivo_legado = st.file_uploader("Upload Planilha Excel", type=['xlsx', 'xls'])
            
            if arquivo_legado:
                xls = carregar_excel_em_cache(arquivo_legado.getvalue())
                todas_abas = xls.sheet_names
                sugestao_abas = [s for s in todas_abas if "fluxo" in s.lower()]
                if not sugestao_abas: sugestao_abas = [todas_abas[0]]
                
                abas_selecionadas = st.multiselect("Selecione as abas que deseja importar:", todas_abas, default=sugestao_abas)
                
                if st.button("Processar Abas Selecionadas"):
                    try:
                        df_legado_final = pd.DataFrame()
                        for aba in abas_selecionadas:
                            df_excel = pd.read_excel(xls, sheet_name=aba)
                            col_data = next((c for c in df_excel.columns if 'data' in str(c).lower()), None)
                            col_desc = next((c for c in df_excel.columns if any(x in str(c).lower() for x in ['descri', 'histórico'])), None)
                            col_tipo = next((c for c in df_excel.columns if any(x in str(c).lower() for x in ['tipo', 'natureza'])), None)
                            col_cat = next((c for c in df_excel.columns if 'categoria' in str(c).lower() and 'sub' not in str(c).lower()), None)
                            col_sub = next((c for c in df_excel.columns if 'sub' in str(c).lower()), None)
                            col_val = next((c for c in df_excel.columns if c.strip().lower() == 'valor (r$)'), None)
                            if not col_val: col_val = next((c for c in df_excel.columns if 'valor' in str(c).lower() and 'mensal' not in str(c).lower() and 'anual' not in str(c).lower()), None)
                            
                            if col_data and col_val:
                                df_aba = df_excel[[col_data, col_desc, col_tipo, col_cat, col_sub, col_val]].copy()
                                df_aba.columns = ["Data", "Descrição cobrança", "Tipo de gasto", "Categoria", "Sub-Categoria", "Valor (R$)"]
                                df_aba['Natureza'] = df_aba.apply(lambda row: "Receita" if "receita" in str(row.get('Tipo de gasto', '')).lower() or "receita" in str(row.get('Categoria', '')).lower() else ("Investimento" if "investimento" in str(row.get('Tipo de gasto', '')).lower() else "Despesa"), axis=1)
                                df_aba['Finalidade'] = 'Legado Excel'
                                df_aba['Cartão/Pix'] = 'Desconhecido'
                                df_legado_final = pd.concat([df_legado_final, df_aba], ignore_index=True)
                        
                        if not df_legado_final.empty:
                            salvar_lote_seguro(conta_selecionada, df_legado_final, nome_lote_legado)
                            st.success(f"Lote '{nome_lote_legado}' processado! Abas salvas.")
                            st.rerun()
                        else: st.error("Não encontrei colunas de Data e Valor nas abas.")
                    except Exception as e: st.error(f"Erro no legado: {e}")

        st.markdown("---")
        st.markdown("### 🤖 2. Auditor Inteligente de Faturas (IA)")
        nome_lote_ia = st.text_input("Nome deste Lote de Faturas:", f"Faturas IA - {datetime.date.today().strftime('%b/%y')}")
        
        arquivos_ia = st.file_uploader("Faturas em PDF/CSV/OFX/Excel para a IA Ler", type=['pdf', 'csv', 'ofx', 'ofc', 'xlsx', 'xls'], accept_multiple_files=True, key="ia_files_center")
        
        if st.button("🚀 Iniciar Leitura da IA", type="primary"):
            if not api_key: st.error("Faltou a API Key.")
            elif not arquivos_ia: st.warning("Faça o upload dos arquivos logo acima.")
            else:
                with st.spinner("A IA está lendo linha por linha..."):
                    try:
                        genai.configure(api_key=api_key)
                        modelo_escolhido = encontrar_modelo_flash()
                        PROMPT_MESTRE = """Atue como Auditor Financeiro. Extraia gastos e receitas.
                        REGRA 1: Retorne APENAS o CSV. Sem saudações ou explicações.
                        REGRA 2: Use PONTO E VÍRGULA (;) como separador. Decimal com VÍRGULA.
                        REGRA 3: O CSV DEVE ter EXATAMENTE o cabeçalho abaixo na primeira linha:
                        Data;Cartão/Pix;Descrição cobrança;Natureza;Finalidade;Tipo de gasto;Categoria;Sub-Categoria;Valor (R$)
                        - Natureza DEVE ser: 'Receita', 'Despesa', 'Investimento' ou 'Transferência'.
                        - Valores SEMPRE positivos. Formato Data: DD/MM/AAAA.
                        """
                        conteudo_envio = [PROMPT_MESTRE]
                        for arq in arquivos_ia:
                            if arq.name.endswith(('.xlsx', '.xls')):
                                try:
                                    df_temp = pd.read_excel(arq)
                                    conteudo_envio.append(df_temp.to_csv(index=False, sep=';'))
                                except: pass
                            elif arq.name.endswith(('.ofx', '.ofc', '.csv')): 
                                conteudo_envio.append(arq.getvalue().decode('utf-8', errors='ignore'))
                            else: 
                                conteudo_envio.append({"mime_type": "application/pdf", "data": arq.getvalue()})

                        response = genai.GenerativeModel(modelo_escolhido).generate_content(conteudo_envio)
                        texto_csv = response.text.replace("```csv", "").replace("```", "").strip()
                        if "Data;" in texto_csv: texto_csv = texto_csv[texto_csv.index("Data;"):]
                        
                        df_novos = pd.read_csv(io.StringIO(texto_csv), sep=';', on_bad_lines='skip')
                        if len(df_novos.columns) != 9:
                            st.error("⚠️ Erro: A IA não formatou as colunas perfeitamente. Tente novamente.")
                        else:
                            df_novos.columns = ["Data", "Cartão/Pix", "Descrição cobrança", "Natureza", "Finalidade", "Tipo de gasto", "Categoria", "Sub-Categoria", "Valor (R$)"]
                            st.session_state['dados_revisao'] = df_novos
                            st.success("✅ Leitura concluída! Por favor, revise os dados abaixo.")
                            st.rerun() 
                    except Exception as e: st.error(f"Erro na IA: {e}")

        if 'dados_revisao' in st.session_state and not st.session_state['dados_revisao'].empty:
            st.markdown("---")
            st.subheader("🕵️ Tela de Revisão (Edite o que a IA errou)")
            df_revisao = st.data_editor(st.session_state['dados_revisao'], num_rows="dynamic", use_container_width=True)
            if st.button("💾 Confirmar e Salvar no Cofre Definitivo", type="primary"):
                with st.spinner("Acionando o Motor Anti-Duplicidade..."):
                    salvar_lote_seguro(conta_selecionada, df_revisao, nome_lote_ia)
                    st.session_state.pop('dados_revisao') 
                    st.success("🎉 Faturas importadas com sucesso!")
                    st.rerun()

            if st.button("❌ Cancelar / Apagar Leitura"):
                st.session_state.pop('dados_revisao')
                st.rerun()

    with tab_cupom:
        st.markdown("### Desconstrução de Cupom Fiscal")
        foto_cupom = st.file_uploader("📸 Foto do Cupom", type=['jpg', 'jpeg', 'png'])
        if foto_cupom and st.button("🧾 Esmiuçar Cupom"):
            with st.spinner("Lendo a notinha..."):
                genai.configure(api_key=api_key)
                response = genai.GenerativeModel(encontrar_modelo_flash()).generate_content([Image.open(foto_cupom), "Extraia itens: Item; Qtde; Categoria; Valor Total(R$). CSV com ponto e vírgula."])
                try: st.dataframe(pd.read_csv(io.StringIO(response.text.replace("```csv", "").replace("```", "").strip()), sep=';'), use_container_width=True)
                except: st.text(response.text)

# =========================================================================================
# MÓDULO 2: WEALTH MANAGEMENT 
# =========================================================================================
elif modulo_selecionado == "💼 Wealth Management":
    
    tab_carteira, tab_plan, tab_terminal, tab_radar, tab_garimpo, tab_analise, tab_macro = st.tabs([
        "💼 Carteira", "🔥 FIRE", "💹 Terminal", "📡 Radar", "🔬 Garimpo", "📊 Análise", "🌍 Cenário"
    ])

    with tab_carteira:
        st.header("💼 Carteira Universal")
        
        col_add, col_imp = st.columns(2)
        with col_add:
            with st.expander("➕ Gestão Manual e Exclusão", expanded=False):
                with st.form("form_add_manual"):
                    t_add = st.text_input("Ticker / Nome do Produto").upper()
                    c_add = st.selectbox("Classe do Ativo", ["Ação", "Fundo Imobiliário", "Renda Fixa", "Fundo de Investimento", "BDR", "Cripto", "Outros"])
                    col_q, col_v = st.columns(2)
                    with col_q: q_add = st.number_input("Quantidade", min_value=0.0001, value=1.0)
                    with col_v: va_add = st.number_input("Valor Aplicado (R$)", min_value=0.01, value=100.0)
                    if st.form_submit_button("Salvar Ativo") and t_add:
                        t_add = t_add.strip()
                        if c_add in ["Ação", "Fundo Imobiliário", "BDR"] and t_add[-1].isdigit() and "." not in t_add: t_add = f"{t_add}.SA"
                        preco_medio = va_add / q_add if q_add > 0 else 0
                        with closing(get_conn()) as conn:
                            with conn:
                                cursor = conn.cursor()
                                cursor.execute("SELECT quantidade FROM carteira WHERE conta=? AND ticker=?", (conta_selecionada, t_add))
                                if cursor.fetchone(): cursor.execute("UPDATE carteira SET classe_ativo=?, quantidade=?, preco_medio=?, valor_aplicado=?, saldo_bruto=? WHERE conta=? AND ticker=?", (c_add, q_add, preco_medio, va_add, va_add, conta_selecionada, t_add))
                                else: cursor.execute("INSERT INTO carteira (conta, ticker, classe_ativo, quantidade, preco_medio, valor_aplicado, saldo_bruto) VALUES (?, ?, ?, ?, ?, ?, ?)", (conta_selecionada, t_add, c_add, q_add, preco_medio, va_add, va_add))
                        st.success(f"{t_add} salvo!")
                        st.rerun()

                with closing(get_conn()) as conn:
                    try:
                        df_excluir = pd.read_sql("SELECT ticker FROM carteira WHERE conta=?", conn, params=(conta_selecionada,))
                        if not df_excluir.empty:
                            with st.form("form_excluir"):
                                t_del = st.selectbox("Selecione o ativo:", df_excluir['ticker'].tolist())
                                col_btn1, col_btn2 = st.columns(2)
                                with col_btn1:
                                    btn_excluir_um = st.form_submit_button("❌ Excluir Ativo")
                                with col_btn2:
                                    btn_zerar_tudo = st.form_submit_button("🚨 Zerar Toda a Carteira")
                                
                                if btn_excluir_um:
                                    with closing(get_conn()) as conn_del:
                                        with conn_del: conn_del.execute("DELETE FROM carteira WHERE conta=? AND ticker=?", (conta_selecionada, t_del))
                                    st.success(f"{t_del} removido!")
                                    st.rerun()
                                    
                                if btn_zerar_tudo:
                                    with closing(get_conn()) as conn_del:
                                        with conn_del: conn_del.execute("DELETE FROM carteira WHERE conta=?", (conta_selecionada,))
                                    st.success("Sua carteira inteira foi zerada com sucesso!")
                                    st.rerun()
                    except: pass

        with col_imp:
            with st.expander("📥 Importar Planilhas Kinvo (Posição / Extrato)", expanded=False):
                st.info("Dica: Toda vez que você clicar em 'Processar Posições', o aplicativo **substitui** a sua carteira antiga pela nova automaticamente.")
                planilha_posicao = st.file_uploader("1. Arquivo de Posições Kinvo", type=['xlsx', 'csv'])
                if planilha_posicao and st.button("Processar Posições"):
                    try:
                        if planilha_posicao.name.endswith('.csv'): df_bruto = pd.read_csv(planilha_posicao, sep=None, engine='python')
                        else: df_bruto = pd.read_excel(planilha_posicao)
                        header_idx = 0
                        for i, row in df_bruto.iterrows():
                            if 'produto' in str(row.values).lower() and ('classe' in str(row.values).lower() or 'valor aplicado' in str(row.values).lower()): header_idx = i + 1; break
                        if planilha_posicao.name.endswith('.csv'): df_import = pd.read_csv(planilha_posicao, sep=None, engine='python', skiprows=header_idx)
                        else: df_import = pd.read_excel(planilha_posicao, skiprows=header_idx)
                        
                        col_t = next((c for c in df_import.columns if any(x in str(c).lower() for x in ['produto', 'ativo', 'ticker'])), None)
                        col_c = next((c for c in df_import.columns if 'classe' in str(c).lower()), None)
                        col_q = next((c for c in df_import.columns if any(x in str(c).lower() for x in ['quantidade', 'quant', 'qtde'])), None)
                        col_va = next((c for c in df_import.columns if 'valor aplicado' in str(c).lower()), None)
                        col_sb = next((c for c in df_import.columns if 'saldo bruto' in str(c).lower()), None)
                        
                        if col_t and col_va and col_sb:
                            df_import = df_import.dropna(subset=[col_t, col_va])
                            
                            df_import[col_va] = converter_para_float(df_import[col_va])
                            df_import[col_sb] = converter_para_float(df_import[col_sb])
                            
                            with closing(get_conn()) as conn:
                                with conn:
                                    conn.execute("DELETE FROM carteira WHERE conta=?", (conta_selecionada,)) 
                                    for _, row in df_import.iterrows():
                                        t_raw = str(row[col_t]).upper().strip()
                                        if t_raw == 'NAN' or t_raw == 'NONE' or len(t_raw) < 2: continue
                                        t = t_raw.split(' - ')[0].strip()
                                        c = str(row[col_c]).title() if col_c else "Outros"
                                        
                                        va = float(row[col_va]) if pd.notnull(row[col_va]) else 0.0
                                        sb = float(row[col_sb]) if pd.notnull(row[col_sb]) else 0.0
                                        
                                        try:
                                            q_str = str(row[col_q]).replace('.', '').replace(',', '.')
                                            q = float(q_str) if col_q and pd.notnull(row[col_q]) else 1.0
                                        except: q = 1.0
                                        
                                        if c in ["Ação", "Fundo Imobiliário", "Bdr"]:
                                            if t[-1].isdigit() and "." not in t: t = f"{t}.SA"
                                        conn.execute("INSERT INTO carteira (conta, ticker, classe_ativo, quantidade, preco_medio, valor_aplicado, saldo_bruto) VALUES (?, ?, ?, ?, ?, ?, ?)", (conta_selecionada, t, c, q, (va/q if q>0 else 0), va, sb))
                            st.success("Carteira Sincronizada com Sucesso!")
                            st.rerun()
                    except Exception as e: st.error(f"Erro na leitura: {e}")

                st.markdown("---")
                nome_lote_kinvo = st.text_input("Nome Lote Kinvo (Para Extrato):", f"Kinvo Extrato - {datetime.date.today().strftime('%b/%y')}")
                planilha_extrato = st.file_uploader("2. Extrato Mensal Kinvo", type=['xlsx', 'csv'], key="up_ex")
                if planilha_extrato and st.button("Gerar Fluxo Automático"):
                    try:
                        if planilha_extrato.name.endswith('.csv'): df_ex_bruto = pd.read_csv(planilha_extrato, sep=None, engine='python')
                        else: df_ex_bruto = pd.read_excel(planilha_extrato)
                        header_idx = 0
                        for i, row in df_ex_bruto.iterrows():
                            if 'Data' in str(row.values).title() and 'Produto' in str(row.values).title(): header_idx = i + 1; break
                        if planilha_extrato.name.endswith('.csv'): df_ex = pd.read_csv(planilha_extrato, sep=None, engine='python', skiprows=header_idx)
                        else: df_ex = pd.read_excel(planilha_extrato, skiprows=header_idx)
                        
                        col_data = next((c for c in df_ex.columns if 'data' in str(c).lower()), None)
                        col_ticker = next((c for c in df_ex.columns if any(x in str(c).lower() for x in ['ticker', 'produto', 'ativo'])), None)
                        col_movimento = next((c for c in df_ex.columns if any(x in str(c).lower() for x in ['descrição', 'movimentação', 'operação', 'tipo'])), None)
                        col_valor = next((c for c in df_ex.columns if 'valor' in str(c).lower()), None)
                        
                        if col_valor:
                            df_ex[col_valor] = converter_para_float(df_ex[col_valor])
                            
                        novos_fluxos = []
                        for _, row in df_ex.iterrows():
                            data = row.get(col_data)
                            ticker = str(row.get(col_ticker, 'N/A')).upper().split(' - ')[0].strip()
                            tipo_str = str(row.get(col_movimento, '')).lower()
                            
                            valor = abs(float(row.get(col_valor, 0.0))) if col_valor and pd.notnull(row.get(col_valor)) else 0.0
                            
                            if valor == 0 or pd.isnull(data): continue
                            if any(x in tipo_str for x in ['dividendo', 'juros sobre', 'rendimento', 'isento', 'jcp']): n, t_gasto, cat = "Receita", "Receitas", "Rendimentos / Dividendos"
                            elif any(x in tipo_str for x in ['aplicação', 'compra', 'investimento']): n, t_gasto, cat = "Investimento", "Investimentos", "Aportes"
                            elif any(x in tipo_str for x in ['resgate', 'venda']): n, t_gasto, cat = "Receita", "Receitas", "Resgate de Investimento"
                            else: continue 
                            novos_fluxos.append({"Data": data, "Cartão/Pix": "Kinvo", "Descrição cobrança": f"{tipo_str.title()} - {ticker}", "Natureza": n, "Finalidade": "Investimento", "Tipo de gasto": t_gasto, "Categoria": cat, "Sub-Categoria": str(ticker), "Valor (R$)": valor})
                            
                        if novos_fluxos:
                            df_nf = pd.DataFrame(novos_fluxos)
                            salvar_lote_seguro(conta_selecionada, df_nf, nome_lote_kinvo)
                            st.success(f"🎉 {len(novos_fluxos)} transações injetadas no Dashboard sem duplicatas!")
                            st.rerun()
                    except Exception as e: st.error(f"Erro: {e}")

        st.markdown("---")
        
        with closing(get_conn()) as conn:
            df_cart = pd.read_sql("SELECT ticker, classe_ativo, quantidade, preco_medio, valor_aplicado, saldo_bruto FROM carteira WHERE conta=?", conn, params=(conta_selecionada,))
        
        if not df_cart.empty:
            try:
                with closing(get_conn()) as conn:
                    df_fluxo = pd.read_sql("SELECT Data, \"Valor (R$)\", Natureza FROM fluxo WHERE conta=?", conn, params=(conta_selecionada,))
                df_fluxo_desp = df_fluxo[df_fluxo['Natureza'] == 'Despesa'].copy()
                df_fluxo_desp['v'] = converter_para_float(df_fluxo_desp['Valor (R$)'])
                df_fluxo_desp['data_calc'] = pd.to_datetime(df_fluxo_desp['Data'], dayfirst=True, errors='coerce')
                df_fluxo_desp = df_fluxo_desp.dropna(subset=['data_calc'])
                df_fluxo_desp['mes_ano'] = df_fluxo_desp['data_calc'].dt.to_period('M')
                gastos_mensais = df_fluxo_desp.groupby('mes_ano')['v'].sum()
                despesa_media_mensal = gastos_mensais.mean() if not gastos_mensais.empty else 5000.0
            except: despesa_media_mensal = 5000.0
            
            reserva_ideal = despesa_media_mensal * 6
            reserva_atual = df_cart[df_cart['classe_ativo'].str.contains('Renda Fixa', case=False, na=False)]['saldo_bruto'].sum()
            pct_reserva = (reserva_atual / reserva_ideal * 100) if reserva_ideal > 0 else 0
            
            with st.expander("💧 Saúde da Reserva de Emergência (Alvo: 6 meses de Custo de Vida)", expanded=True):
                cor_res = "green" if pct_reserva >= 100 else ("orange" if pct_reserva >= 50 else "red")
                st.markdown(f"**Custo de Vida Médio Estimado:** {formata_moeda(despesa_media_mensal)}/mês")
                st.markdown(f"**Reserva Ideal Alvo (6x):** {formata_moeda(reserva_ideal)}")
                st.markdown(f"**Reserva Atual (Sua Renda Fixa):** {formata_moeda(reserva_atual)}")
                st.progress(min(pct_reserva / 100, 1.0))
                st.markdown(f"<span style='color:{cor_res}; font-weight:bold;'>Progresso: {pct_reserva:.1f}%</span>", unsafe_allow_html=True)
            
            st.markdown("---")

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                classes = ["Todas as Classes"] + list(df_cart['classe_ativo'].unique())
                filtro_classe = st.selectbox("🔎 Filtrar por Classe:", classes)
            with col_f2:
                busca_nome = st.text_input("🔍 Buscar Ativo/Produto:")
                
            df_exibir = df_cart.copy()
            if filtro_classe != "Todas as Classes": df_exibir = df_exibir[df_exibir['classe_ativo'] == filtro_classe]
            if busca_nome: df_exibir = df_exibir[df_exibir['ticker'].str.contains(busca_nome.upper(), na=False)]

            if not df_exibir.empty:
                df_calc = df_exibir.copy()
                total_investido = df_calc['valor_aplicado'].sum()
                total_atual = df_calc['saldo_bruto'].sum()
                lucro_global_rs = total_atual - total_investido
                lucro_global_pct = (lucro_global_rs / total_investido * 100) if total_investido > 0 else 0
                
                renda_variavel = df_calc[df_calc['classe_ativo'].isin(["Ação", "Fundo Imobiliário", "Bdr", "Cripto"])]['saldo_bruto'].sum()
                pct_rv = (renda_variavel / total_atual * 100) if total_atual > 0 else 0
                nivel_risco = "Agressivo" if pct_rv > 70 else ("Moderado" if pct_rv > 30 else "Conservador")
                
                st.markdown("<br>", unsafe_allow_html=True)
                col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                with col_k1: st.markdown(f"""<div class="kinvo-card"><h4>💼 PATRIMÔNIO</h4><h2>{formata_moeda(total_atual)}</h2><p style="color: #888; font-size: 12px;">Total Inves.: {formata_moeda(total_investido)}</p></div>""", unsafe_allow_html=True)
                cor_l = "pos-val" if lucro_global_rs >= 0 else "neg-val"
                s_l = "+" if lucro_global_rs >= 0 else ""
                with col_k2: st.markdown(f"""<div class="kinvo-card"><h4>📈 RENTABILIDADE (Total Acumulada)</h4><h2 class="{cor_l}">{s_l}{lucro_global_pct:.2f}%</h2><p class="{cor_l}" style="font-size: 12px;">{s_l}{formata_moeda(lucro_global_rs)}</p></div>""", unsafe_allow_html=True)
                with col_k3: st.markdown(f"""<div class="kinvo-card"><h4>🌪️ PERFIL RISCO</h4><h2>{nivel_risco}</h2><p style="color: #888; font-size: 12px;">RV: {pct_rv:.1f}% / RF: {(100-pct_rv):.1f}%</p></div>""", unsafe_allow_html=True)
                
                with col_k4:
                    df_calc['Rentabilidade (%)'] = ((df_calc['saldo_bruto'] - df_calc['valor_aplicado']) / df_calc['valor_aplicado'] * 100).fillna(0)
                    if len(df_calc) > 1:
                        melhor = df_calc.loc[df_calc['Rentabilidade (%)'].idxmax()]
                        st.markdown(f"""<div class="kinvo-card"><h4>🏆 TOP ATIVO</h4><p style="margin: 0; font-size: 15px; font-weight: bold; color: green;">{melhor['ticker']}</p><p style="margin: 0; font-size: 14px; color: green;">+{melhor['Rentabilidade (%)']:.1f}%</p></div>""", unsafe_allow_html=True)
                    else: st.markdown("""<div class="kinvo-card"><h4>🏆 TOP ATIVO</h4><p>-</p></div>""", unsafe_allow_html=True)

                st.write("")
                col_graf1, col_graf2 = st.columns(2)
                with col_graf1:
                    st.markdown("### 🥧 Divisão por Classe")
                    resumo_classes = df_calc.groupby('classe_ativo')['saldo_bruto'].sum().reset_index()
                    if not resumo_classes.empty:
                        fig_pie = px.pie(resumo_classes, values='saldo_bruto', names='classe_ativo', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig_pie.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_graf2:
                    st.markdown("### 🗺️ Mapa de Calor (Treemap)")
                    df_tree = df_calc[df_calc['saldo_bruto'] > 0].copy()
                    if not df_tree.empty:
                        fig_tree = px.treemap(df_tree, path=[px.Constant("Minha Carteira"), 'classe_ativo', 'ticker'], values='saldo_bruto', color='Rentabilidade (%)', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
                        fig_tree.update_traces(textinfo="label+value+percent parent")
                        fig_tree.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                        st.plotly_chart(fig_tree, use_container_width=True)

                st.markdown("### 📋 Posições Consolidadas")
                df_formatado = df_calc[['ticker', 'classe_ativo', 'valor_aplicado', 'saldo_bruto', 'Rentabilidade (%)']].copy()
                df_formatado.columns = ["Produto", "Classe", "Valor Aplicado", "Saldo Atual", "Rent. (%)"]
                df_formatado["Valor Aplicado"] = df_formatado["Valor Aplicado"].apply(lambda x: f"R$ {x:,.2f}")
                df_formatado["Saldo Atual"] = df_formatado["Saldo Atual"].apply(lambda x: f"R$ {x:,.2f}")
                st.dataframe(df_formatado, use_container_width=True, hide_index=True, column_config={"Rent. (%)": st.column_config.NumberColumn("Rent. (%)", format="%.2f%%")})
        else:
            st.info("Sua carteira está vazia. Adicione ativos ou importe a planilha do Kinvo.")

    with tab_plan:
        st.header("🔥 O Motor F.I.R.E (Independência Financeira)")
        with closing(get_conn()) as conn:
            try:
                saldo_base = pd.read_sql("SELECT SUM(saldo_bruto) as s FROM carteira WHERE conta=?", conn, params=(conta_selecionada,))['s'].iloc[0]
                saldo_base = float(saldo_base) if pd.notnull(saldo_base) else 0.0
            except: saldo_base = 0.0

        st.markdown("#### 1. Fase de Acumulação (O Caminho)")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1: meta_patrimonio = st.number_input("🎯 Meta Desejada (R$)", value=1000000.0, step=10000.0)
        with col_p2: aporte_proj = st.number_input("💸 Aporte Mensal (R$)", value=2000.0, step=100.0)
        with col_p3: taxa_proj = st.number_input("📈 Taxa Anual Esperada (% a.a.)", value=12.0, step=1.0)
        
        anos_proj = st.slider("📅 Simule o Tempo de Investimento (Anos)", min_value=1, max_value=40, value=15)
        
        df_simulacao = simular_juros_compostos(saldo_base, aporte_proj, taxa_proj, anos_proj)
        saldo_final = df_simulacao['Patrimônio Total'].iloc[-1]
        
        st.markdown("---")
        
        st.markdown("#### 2. Fase de Fruição (A Recompensa Real)")
        st.markdown("Cuidado com a **Ilusão Nominal**! Entenda a diferença entre o Rendimento do Banco e o Saque Seguro contra a inflação.")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1: taxa_aposentadoria = st.number_input("🛡️ Rendimento Nominal no Futuro (% a.a.)", value=8.0, step=0.5)
        
        taxa_mensal_aposentadoria = (1 + taxa_aposentadoria/100) ** (1/12) - 1
        rendimento_mensal = meta_patrimonio * taxa_mensal_aposentadoria
        renda_regra_4 = (meta_patrimonio * 0.04) / 12
        
        with col_r2: saque_mensal = st.number_input("💸 Saque Mensal Desejado (R$)", value=float(round(rendimento_mensal, 2)), step=500.0)
        with col_r3: st.metric("💰 Juros Nominais (Mês)", formata_moeda(rendimento_mensal))
        
        if saque_mensal >= rendimento_mensal:
            try:
                n_meses = -math.log(1 - (meta_patrimonio * taxa_mensal_aposentadoria) / saque_mensal) / math.log(1 + taxa_mensal_aposentadoria)
                anos_esgotamento = int(n_meses // 12)
                meses_esgotamento = int(n_meses % 12)
                st.error(f"🚨 **Risco Extremo:** Você está sacando mais (ou tudo) que o rendimento! Sacando {formata_moeda(saque_mensal)}, seu saldo **ZERA em {anos_esgotamento} anos e {meses_esgotamento} meses**.")
            except:
                st.error(f"🚨 **Risco Extremo:** Você está sacando tudo! O seu dinheiro vai esgotar rapidamente e a inflação vai te devorar antes disso.")
        elif saque_mensal > renda_regra_4:
            st.warning(f"⚠️ **A Ilusão Nominal (Risco de Inflação):** Sacando {formata_moeda(saque_mensal)}, o número no banco não cai, mas **você ficará mais pobre a cada ano que passar**. A inflação fará com que esses {formata_moeda(saque_mensal)} não comprem quase nada no futuro. A Regra dos 4% manda você sacar no máximo **{formata_moeda(renda_regra_4)}** e deixar a diferença rendendo no banco para corrigir a inflação!")
        else:
            st.success(f"🟢 **Blindagem Total (Trinity Study):** Sacando {formata_moeda(saque_mensal)}, você está respeitando a Regra dos 4%. O que sobra do rendimento vai repor a inflação automaticamente e seu patrimônio nunca esgotará na vida real!")

        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### 🏁 Projeção em {anos_proj} anos: **<span style='color:green;'>{formata_moeda(saldo_final)}</span>**", unsafe_allow_html=True)
            
            if saldo_base > 0 or aporte_proj > 0:
                s = saldo_base
                t_m = (1 + taxa_proj/100)**(1/12) - 1
                meses_calc = 0
                while s < meta_patrimonio and meses_calc < 600:
                    s = s * (1 + t_m) + aporte_proj
                    meses_calc += 1
                
                if meses_calc < 600:
                    st.success(f"⏳ Mantendo esse aporte, você atingirá a Independência Financeira em **{meses_calc//12} anos e {meses_calc%12} meses**.")
                else:
                    st.warning("Com esses valores, demorará mais de 50 anos para atingir a meta.")
                    
        with c2:
            st.plotly_chart(px.area(df_simulacao, x="Ano", y=["Total Investido", "Patrimônio Total"], title="A Bola de Neve", color_discrete_sequence=['#A9DFBF', '#006437']), use_container_width=True)

    with tab_terminal:
        st.header("💹 Terminal Fundamentalista (Modo Profissional)")
        if not YFINANCE_INSTALADO: st.error("Módulo yfinance não instalado!")
        else:
            col_busca, col_tempo = st.columns([3, 1])
            with col_busca: ticker_input = st.text_input("🔍 Código do Ativo:", "VALE3")
            with col_tempo: periodo = st.selectbox("📅 Período do Gráfico:", ["1mo", "6mo", "1y", "5y", "max"], index=2)

            if ticker_input:
                ticker_buscado = ticker_input.upper().strip()
                if ticker_buscado[-1].isdigit() and "." not in ticker_buscado and "-" not in ticker_buscado and "^" not in ticker_buscado: ticker_buscado = f"{ticker_buscado}.SA"
                
                with st.spinner(f"Extraindo dados profundos de {ticker_buscado}..."):
                    try:
                        ativo = yf.Ticker(ticker_buscado)
                        hist = ativo.history(period=periodo)
                        
                        if not hist.empty:
                            info = ativo.info
                            nome = info.get("shortName", ticker_buscado)
                            simb = "R$" if info.get("currency", "BRL") == "BRL" else "$"
                            
                            def s_float(val):
                                try: return float(val) if val is not None else 0.0
                                except: return 0.0

                            preco_atual = s_float(hist['Close'].iloc[-1])
                            min_52 = s_float(info.get('fiftyTwoWeekLow'))
                            max_52 = s_float(info.get('fiftyTwoWeekHigh'))
                            
                            div_rate = s_float(info.get('trailingAnnualDividendRate'))
                            if div_rate > 0 and preco_atual > 0:
                                dy = (div_rate / preco_atual) * 100
                            else:
                                dy_raw = s_float(info.get('dividendYield', 0))
                                if dy_raw == 0:
                                    dy_raw = s_float(info.get('trailingAnnualDividendYield', 0))
                                dy = dy_raw if dy_raw > 0.3 else dy_raw * 100
                            
                            p_l = s_float(info.get('trailingPE'))
                            p_vp = s_float(info.get('priceToBook'))
                            ev_ebitda = s_float(info.get('enterpriseToEbitda'))
                            vpa = s_float(info.get('bookValue'))
                            lpa = s_float(info.get('trailingEps'))
                            
                            roe = s_float(info.get('returnOnEquity')) * 100
                            roa = s_float(info.get('returnOnAssets')) * 100
                            margem_liq = s_float(info.get('profitMargins')) * 100
                            margem_bruta = s_float(info.get('grossMargins')) * 100
                            
                            div_pl = s_float(info.get('debtToEquity')) / 100 
                            liq_corr = s_float(info.get('currentRatio'))
                            
                            col_h1, col_h2 = st.columns([2, 1])
                            with col_h1: st.subheader(f"🏢 {nome} ({ticker_buscado})")
                            with col_h2: 
                                score = calcular_tepetos_score(p_l, roe/100, div_pl*100, dy/100, info.get('revenueGrowth', 0) or 0)
                                st.metric("🥇 TEPETOS SCORE", f"{score} / 100")
                                st.caption("Nota de 0 a 100 baseada na solidez, preço, dívida e dividendos do ativo.")
                                
                            with st.expander("📚 Como o TEPETOS SCORE é calculado?"):
                                st.markdown("""
                                O **Tepetos Score** é um algoritmo exclusivo inspirado nas metodologias de Benjamin Graham e Décio Bazin. Ele avalia a ação com uma nota de 0 a 100 baseada em 5 pilares:
                                
                                1. **Valuation (P/L) - Até 25 pts:** P/L entre 0 e 10 (25 pts) | 10 a 15 (18 pts) | 15 a 25 (10 pts).
                                2. **Rentabilidade (ROE) - Até 25 pts:** ROE acima de 20% (25 pts) | Acima de 15% (18 pts) | Acima de 10% (10 pts).
                                3. **Endividamento (Dívida/PL) - Até 20 pts:** Abaixo de 0.5 (20 pts) | Abaixo de 1.0 (15 pts) | Acima de 2.0 zera.
                                4. **Proventos (Dividend Yield) - Até 15 pts:** Acima de 6% (15 pts) | Acima de 4% (10 pts).
                                5. **Crescimento (Receita) - Até 15 pts:** Acima de 10% de crescimento (15 pts).
                                """)
                            
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Cotação Atual", f"{simb} {preco_atual:.2f}")
                            c2.metric("Min (52 semanas)", f"{simb} {min_52:.2f}")
                            c3.metric("Max (52 semanas)", f"{simb} {max_52:.2f}")
                            c4.metric("Dividend Yield (12m)", f"{dy:.2f}%")
                            
                            st.markdown("---")
                            benchmarks = st.multiselect("📈 Adicionar Comparativos ao Gráfico:", ["IBOVESPA", "S&P 500", "NASDAQ"], default=["IBOVESPA"])
                            
                            try:
                                df_grafico = pd.DataFrame()
                                df_grafico[ticker_buscado] = (hist['Close'] / hist['Close'].iloc[0]) * 100
                                
                                if "IBOVESPA" in benchmarks:
                                    ibov_hist = yf.Ticker("^BVSP").history(period=periodo)
                                    if not ibov_hist.empty: df_grafico['IBOVESPA'] = (ibov_hist['Close'] / ibov_hist['Close'].iloc[0]) * 100
                                if "S&P 500" in benchmarks:
                                    sp_hist = yf.Ticker("^GSPC").history(period=periodo)
                                    if not sp_hist.empty: df_grafico['S&P 500'] = (sp_hist['Close'] / sp_hist['Close'].iloc[0]) * 100
                                if "NASDAQ" in benchmarks:
                                    nas_hist = yf.Ticker("^IXIC").history(period=periodo)
                                    if not nas_hist.empty: df_grafico['NASDAQ'] = (nas_hist['Close'] / nas_hist['Close'].iloc[0]) * 100
                                    
                                st.plotly_chart(px.line(df_grafico, x=df_grafico.index, y=df_grafico.columns, title="Desempenho Normalizado (Base 100)"), use_container_width=True)
                            except: st.plotly_chart(px.line(hist, x=hist.index, y='Close', title="Evolução de Preço"), use_container_width=True)
                            
                            st.markdown("---")
                            st.markdown("### 📊 Raio-X Fundamentalista")
                            f1, f2, f3 = st.columns(3)
                            
                            with f1:
                                st.markdown("<div class='status-card'>", unsafe_allow_html=True)
                                st.markdown("<h4 style='color:#006437; margin-bottom:15px;'>💰 Valuation (Preço)</h4>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{p_l:.2f}</p><p class='metric-ideal'>**P/L** (Ideal: 5 a 15)</p><p class='metric-desc'>Anos para o lucro pagar a ação.</p><hr style='margin:10px 0;'>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{p_vp:.2f}</p><p class='metric-ideal'>**P/VP** (Ideal: 0.5 a 2.0)</p><p class='metric-desc'>Preço que o mercado paga pelo patrimônio.</p><hr style='margin:10px 0;'>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{ev_ebitda:.2f}</p><p class='metric-ideal'>**EV/EBITDA** (Ideal: < 10)</p><p class='metric-desc'>Anos para a operação pagar a empresa.</p><hr style='margin:10px 0;'>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{simb} {vpa:.2f}</p><p class='metric-ideal'>**VPA**</p><p class='metric-desc'>Valor real contábil de 1 ação.</p><hr style='margin:10px 0;'>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{simb} {lpa:.2f}</p><p class='metric-ideal'>**LPA**</p><p class='metric-desc'>Lucro líquido gerado por 1 ação.</p>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                            with f2:
                                st.markdown("<div class='status-card'>", unsafe_allow_html=True)
                                st.markdown("<h4 style='color:#006437; margin-bottom:15px;'>🚀 Rentabilidade</h4>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{roe:.2f}%</p><p class='metric-ideal'>**ROE** (Ideal: > 10%)</p><p class='metric-desc'>Retorno sobre o patrimônio dos sócios.</p><hr style='margin:10px 0;'>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{roa:.2f}%</p><p class='metric-ideal'>**ROA** (Ideal: > 5%)</p><p class='metric-desc'>Retorno sobre todos os ativos.</p><hr style='margin:10px 0;'>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{margem_liq:.2f}%</p><p class='metric-ideal'>**Margem Líquida** (Ideal: > 10%)</p><p class='metric-desc'>% da receita que vira lucro limpo.</p><hr style='margin:10px 0;'>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{margem_bruta:.2f}%</p><p class='metric-ideal'>**Margem Bruta**</p><p class='metric-desc'>Lucro da operação base antes dos impostos.</p>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                            with f3:
                                st.markdown("<div class='status-card'>", unsafe_allow_html=True)
                                st.markdown("<h4 style='color:#006437; margin-bottom:15px;'>⚖️ Dívida & Proventos</h4>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{div_pl:.2f}</p><p class='metric-ideal'>**Dívida L. / Patrimônio** (Ideal: < 1.0)</p><p class='metric-desc'>Se for > 1, tem mais dívida que patrimônio.</p><hr style='margin:10px 0;'>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{liq_corr:.2f}</p><p class='metric-ideal'>**Liquidez Corrente** (Ideal: > 1.5)</p><p class='metric-desc'>Capacidade de pagar contas de curto prazo.</p><hr style='margin:10px 0;'>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value'>{dy:.2f}%</p><p class='metric-ideal'>**Dividend Yield** (Ideal: > 6%)</p><p class='metric-desc'>Rendimento em dividendos (12m).</p><hr style='margin:10px 0;'>", unsafe_allow_html=True)
                                
                                st.markdown(f"<p class='metric-value' style='font-size:16px;'>{classificar_ativo(p_l, roe/100, dy/100)}</p><p class='metric-ideal'>**Classificação do Ativo**</p><p class='metric-desc'>Resumo do algoritmo sobre o papel.</p>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                    except Exception as e: 
                        st.error(f"Ativo não encontrado ou falha técnica na API do Yahoo Finance. Detalhe do Erro: {e}")

    with tab_radar:
        st.header("📡 Radar Tepetos (Monitor de Oportunidades)")
        st.markdown("Esta inteligência varre automaticamente uma lista selecionada de grandes empresas sólidas (Blue Chips, Pagadoras de Dividendos e Tecnologia) e calcula o **Tepetos Score** em tempo real para encontrar as melhores oportunidades de aporte hoje.")
        
        tickers_radar = [
            "VALE3.SA", "PETR4.SA", "WEGE3.SA", "ITUB4.SA", "BBAS3.SA", 
            "EGIE3.SA", "TAEE11.SA", "BBDC4.SA", "KLBN11.SA", "SAPR11.SA",
            "CSAN3.SA", "VIVT3.SA", "AAPL", "MSFT", "NVDA"
        ]
        
        if st.button("🔄 Escanear Mercado Agora", type="primary"):
            with st.spinner("Analisando balanços e cotações da Bolsa. Isso pode levar alguns segundos..."):
                dados_radar = []
                
                def r_float(val):
                    try: return float(val) if val is not None else 0.0
                    except: return 0.0
                
                for t in tickers_radar:
                    try:
                        ativo_radar = yf.Ticker(t)
                        info = ativo_radar.info
                        if info:
                            preco_atual_radar = r_float(info.get('currentPrice', info.get('previousClose')))
                            
                            p_l = r_float(info.get('trailingPE'))
                            roe = r_float(info.get('returnOnEquity'))
                            div_pl = r_float(info.get('debtToEquity')) / 100
                            
                            div_rate = r_float(info.get('trailingAnnualDividendRate'))
                            if div_rate > 0 and preco_atual_radar > 0:
                                dy = (div_rate / preco_atual_radar) * 100
                            else:
                                dy_raw = r_float(info.get('dividendYield', 0))
                                if dy_raw == 0:
                                    dy_raw = r_float(info.get('trailingAnnualDividendYield', 0))
                                dy = dy_raw if dy_raw > 0.3 else dy_raw * 100
                            
                            cagr = r_float(info.get('revenueGrowth'))
                            
                            score_calculado = calcular_tepetos_score(p_l, roe, div_pl*100, dy/100, cagr)
                            
                            dados_radar.append({
                                "Ativo": t.replace(".SA", ""),
                                "Setor": info.get('sector', 'N/A'),
                                "Cotação Atual": f"R$ {preco_atual_radar:.2f}",
                                "P/L": round(p_l, 2),
                                "ROE (%)": round(roe * 100, 2),
                                "DY (%)": round(dy, 2),
                                "Tepetos Score": score_calculado
                            })
                    except:
                        pass
                
                if dados_radar:
                    df_radar = pd.DataFrame(dados_radar).sort_values(by="Tepetos Score", ascending=False)
                    st.success("Varredura concluída! Veja o ranking das melhores ações do momento:")
                    st.dataframe(
                        df_radar, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Tepetos Score": st.column_config.ProgressColumn("Tepetos Score", format="%d", min_value=0, max_value=100)
                        }
                    )
        else:
            st.info("Clique no botão acima para acionar o robô de varredura do mercado.")

    with tab_garimpo:
        st.header("🔬 Garimpo Tepetos (Stock Screener)")
        st.markdown("Faça o upload da planilha exportada do **Status Invest** ou **Investidor10** (Busca Avançada) e use os filtros dinâmicos abaixo para garimpar as melhores oportunidades.")

        arquivo_garimpo = st.file_uploader("📥 Suba o arquivo CSV/Excel:", type=['csv', 'xlsx'])

        st.markdown("### 🎛️ Filtros Fundamentalistas Dinâmicos")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            filtro_pl = st.slider("P/L (Min/Max)", min_value=-50.0, max_value=150.0, value=(5.0, 15.0), step=0.5, help="Brasil: 8 a 12 | EUA: 18 a 25")
            filtro_roe = st.slider("ROE (%) (Min/Max)", min_value=-50.0, max_value=150.0, value=(15.0, 60.0), step=1.0, help="Brasil: > 15% | EUA: > 18% (Teto máximo evita distorções contábeis)")
        with col_f2:
            filtro_roic = st.slider("ROIC (%) (Min/Max)", min_value=-50.0, max_value=150.0, value=(12.0, 60.0), step=1.0, help="Brasil: > 12% | EUA: > 15% (Teto máximo evita distorções)")
            filtro_div_ebitda = st.number_input("Dív. Líq/EBITDA Máx", value=2.5, step=0.1, help="Brasil: < 2.5 | EUA: < 2.0 (quando houver)")
        with col_f3:
            filtro_cagr_lucro = st.number_input("CAGR Lucro Mín (%)", value=10.0, step=1.0, help="Brasil: >= 10% | EUA: > 15% (quando houver)")
            filtro_cagr_rec = st.number_input("CAGR Receita Mín (%)", value=15.0, step=1.0, help="Brasil: >= 15% | EUA: > 15% (quando houver)")
        with col_f4:
            filtro_valor_mercado = st.number_input("Valor Mercado Mín (Milhões)", value=0.0, step=100.0, help="Ex: 1000 para empresas > 1 Bilhão")

        if arquivo_garimpo:
            try:
                if arquivo_garimpo.name.endswith('.csv'):
                    df_bruto = pd.read_csv(arquivo_garimpo, sep=';', decimal=',', encoding='utf-8-sig', on_bad_lines='skip')
                    if len(df_bruto.columns) < 3: 
                        arquivo_garimpo.seek(0)
                        df_bruto = pd.read_csv(arquivo_garimpo, sep=',', on_bad_lines='skip')
                else:
                    df_bruto = pd.read_excel(arquivo_garimpo)

                def limpa_si(val):
                    if pd.isna(val) or str(val).strip() == '-' or str(val).strip() == '': return None
                    v_str = str(val).replace('%', '').strip()
                    if ',' in v_str and '.' in v_str:
                        v_str = v_str.replace('.', '').replace(',', '.')
                    elif ',' in v_str:
                        v_str = v_str.replace(',', '.')
                    try: return float(v_str)
                    except: return None

                cols = [str(c).upper().strip() for c in df_bruto.columns]
                df_bruto.columns = cols

                col_ticker = next((c for c in cols if 'TICKER' in c or 'ATIVO' in c or 'PAPEL' in c), None)
                col_preco = next((c for c in cols if 'PRECO' in c or 'PREÇO' in c or 'COTAÇÃO' in c), None)
                col_pl = next((c for c in cols if c == 'P/L' or 'P/L' in c), None)
                col_roe = next((c for c in cols if 'ROE' in c), None)
                col_roic = next((c for c in cols if 'ROIC' in c), None)
                col_div = next((c for c in cols if 'EBITDA' in c and ('DIV' in c or 'DÍV' in c)), next((c for c in cols if 'EBIT' in c and 'DIV' in c), None))
                col_cagr_r = next((c for c in cols if 'CAGR REC' in c), None)
                col_cagr_l = next((c for c in cols if 'CAGR LUC' in c), None)
                col_dy = next((c for c in cols if 'DY' in c or 'DIVIDEND' in c), None)
                col_div_patr = next((c for c in cols if 'PATRI' in c and 'DIV' in c), None)
                col_val_mercado = next((c for c in cols if 'VALOR DE MERCADO' in c or 'MERCADO' in c), None)

                if col_ticker and col_pl:
                    df_limpo = df_bruto.copy()
                    
                    df_limpo['P/L_calc'] = df_limpo[col_pl].apply(limpa_si)
                    df_limpo['ROE_calc'] = df_limpo[col_roe].apply(limpa_si) if col_roe else None
                    df_limpo['ROIC_calc'] = df_limpo[col_roic].apply(limpa_si) if col_roic else None
                    df_limpo['Div_calc'] = df_limpo[col_div].apply(limpa_si) if col_div else None
                    df_limpo['CagrR_calc'] = df_limpo[col_cagr_r].apply(limpa_si) if col_cagr_r else None
                    df_limpo['CagrL_calc'] = df_limpo[col_cagr_l].apply(limpa_si) if col_cagr_l else None
                    df_limpo['ValMercado_calc'] = df_limpo[col_val_mercado].apply(limpa_si) if col_val_mercado else None
                    
                    df_limpo = df_limpo.dropna(subset=['P/L_calc', 'ROE_calc', 'ROIC_calc', 'Div_calc', 'CagrR_calc', 'CagrL_calc', 'ValMercado_calc'])
                    
                    mask_pl = (df_limpo['P/L_calc'] >= filtro_pl[0]) & (df_limpo['P/L_calc'] <= filtro_pl[1])
                    mask_roe = (df_limpo['ROE_calc'] >= filtro_roe[0]) & (df_limpo['ROE_calc'] <= filtro_roe[1])
                    mask_roic = (df_limpo['ROIC_calc'] >= filtro_roic[0]) & (df_limpo['ROIC_calc'] <= filtro_roic[1])
                    mask_div = (df_limpo['Div_calc'] <= filtro_div_ebitda)
                    mask_cagrr = (df_limpo['CagrR_calc'] >= filtro_cagr_rec)
                    mask_cagrl = (df_limpo['CagrL_calc'] >= filtro_cagr_lucro)
                    mask_mercado = ((df_limpo['ValMercado_calc'] / 1000000) >= filtro_valor_mercado)

                    df_filtrado = df_limpo[mask_pl & mask_roe & mask_roic & mask_div & mask_cagrr & mask_cagrl & mask_mercado].copy()

                    if not df_filtrado.empty:
                        scores = []
                        for _, row in df_filtrado.iterrows():
                            dy_val = limpa_si(row[col_dy]) if col_dy else 0.0
                            div_patr_val = limpa_si(row[col_div_patr]) if col_div_patr else 0.0
                            pl_val = row['P/L_calc'] or 0.0
                            roe_val = (row['ROE_calc'] or 0.0) / 100.0
                            cagr_l_val = (row['CagrL_calc'] or 0.0) / 100.0
                            
                            dy_calc = (dy_val / 100.0) if dy_val else 0.0
                            div_patr_calc = (div_patr_val * 100) if div_patr_val else 0.0

                            s = calcular_tepetos_score(pl_val, roe_val, div_patr_calc, dy_calc, cagr_l_val)
                            scores.append(s)

                        df_filtrado['Tepetos Score'] = scores
                        df_filtrado = df_filtrado.sort_values(by='Tepetos Score', ascending=False)

                        cols_view = [col_ticker]
                        if col_preco: cols_view.append(col_preco)
                        cols_view.extend([col_pl, col_roe, col_roic, col_div, col_cagr_r, col_cagr_l, col_val_mercado, col_dy, 'Tepetos Score'])
                        
                        cols_view_final = []
                        for c in cols_view:
                            if c is not None and c not in cols_view_final:
                                cols_view_final.append(c)

                        st.success(f"🎉 O Filtro Implacável encontrou exatamente {len(df_filtrado)} diamantes!")
                        st.dataframe(df_filtrado[cols_view_final], use_container_width=True, hide_index=True, column_config={"Tepetos Score": st.column_config.ProgressColumn("Tepetos Score", min_value=0, max_value=100)})
                        
                    else:
                        st.warning("Nenhum ativo passou por esse filtro rigoroso! Tente afrouxar um pouco as métricas nas caixinhas acima.")
                else:
                    st.error("Não consegui identificar as colunas 'TICKER' ou 'P/L' no seu arquivo. Tem certeza de que é a exportação correta?")
            except Exception as e:
                st.error(f"Erro ao processar a planilha: {e}")

    with tab_analise:
        st.header("📊 Análise Profunda da Carteira")
        st.markdown("Acompanhe os fundamentos dos ativos que você possui, rebalanceie setores e execute o seu *Deep Dive Completo* sob demanda.")

        sub_br, sub_us, sub_fii = st.tabs(["🇧🇷 Ações Brasileiras", "🇺🇸 Stocks Americanas", "🏢 FIIs / REITs"])

        with sub_br:
            st.subheader("🇧🇷 Ações Brasileiras na sua Carteira")
            
            with closing(get_conn()) as conn:
                df_br = pd.read_sql("SELECT ticker, valor_aplicado, saldo_bruto FROM carteira WHERE conta=? AND classe_ativo='Ação' AND ticker LIKE '%.SA'", conn, params=(conta_selecionada,))
            
            if df_br.empty:
                st.info("Você não possui Ações Brasileiras cadastradas na sua carteira (ativos terminados em .SA).")
            else:
                tot_aplicado_br = df_br['valor_aplicado'].sum()
                tot_atual_br = df_br['saldo_bruto'].sum()
                rent_br = ((tot_atual_br / tot_aplicado_br) - 1) * 100 if tot_aplicado_br > 0 else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Valor Aplicado (BR)", formata_moeda(tot_aplicado_br))
                c2.metric("Saldo Atual (BR)", formata_moeda(tot_atual_br))
                c3.metric("Rentabilidade (Total Acumulada)", f"{rent_br:.2f}%")
                
                st.markdown("---")
                
                st.markdown("### 📋 Tabela Fundamentalista e Setorial")
                
                if st.button("🔄 Atualizar Fundamentos (Ações BR)"):
                    with st.spinner("Buscando dados no Yahoo Finance..."):
                        fund_br_list = []
                        for t in df_br['ticker'].unique():
                            try:
                                ativo = yf.Ticker(t)
                                info = ativo.info
                                p = s_float(info.get('currentPrice', info.get('previousClose', 0)))
                                pl = info.get('trailingPE', 0)
                                pvp = info.get('priceToBook', 0)
                                evebitda = info.get('enterpriseToEbitda', 0)
                                roe = (info.get('returnOnEquity', 0) or 0) * 100
                                roa = (info.get('returnOnAssets', 0) or 0) * 100
                                m_liq = (info.get('profitMargins', 0) or 0) * 100
                                
                                div_rate = s_float(info.get('trailingAnnualDividendRate'))
                                if div_rate > 0 and p > 0:
                                    dy = (div_rate / p) * 100
                                else:
                                    dy_raw = s_float(info.get('dividendYield', info.get('trailingAnnualDividendYield', 0)))
                                    dy = dy_raw if dy_raw > 0.3 else dy_raw * 100
                                
                                setor = str(info.get('sector', 'Desconhecido'))
                                aplicado_no_app = float(df_br[df_br['ticker'] == t]['valor_aplicado'].sum())
                                saldo_no_app = float(df_br[df_br['ticker'] == t]['saldo_bruto'].sum())
                                rent_ativo = ((saldo_no_app / aplicado_no_app) - 1) * 100 if aplicado_no_app > 0 else 0
                                
                                fund_br_list.append({
                                    "Ticker": t.replace('.SA', ''),
                                    "Setor": setor,
                                    "Preço Atual": f"R$ {p:.2f}",
                                    "Rentabilidade (%)": f"{rent_ativo:.2f}%",
                                    "P/L": round(pl, 2) if pl else "-",
                                    "P/VP": round(pvp, 2) if pvp else "-",
                                    "EV/EBITDA": round(evebitda, 2) if evebitda else "-",
                                    "ROE (%)": round(roe, 2) if roe else "-",
                                    "ROA (%)": round(roa, 2) if roa else "-",
                                    "Margem Líq (%)": round(m_liq, 2) if m_liq else "-",
                                    "Div. Yield (%)": round(dy, 2) if dy else "-",
                                    "Saldo Bruto": saldo_no_app
                                })
                            except: pass
                        if fund_br_list:
                            st.session_state[f'fund_br_cache_{conta_selecionada}'] = pd.DataFrame(fund_br_list)
                
                if f'fund_br_cache_{conta_selecionada}' in st.session_state:
                    df_fund_br = st.session_state[f'fund_br_cache_{conta_selecionada}']
                    st.dataframe(df_fund_br.drop(columns=["Saldo Bruto"]), use_container_width=True, hide_index=True)
                    
                    st.write("")
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_setor = px.pie(df_fund_br, values='Saldo Bruto', names='Setor', hole=0.4, title="Exposição por Setor (R$)", color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_setor, use_container_width=True)
                    with col_g2:
                        fig_bar = px.bar(df_fund_br.sort_values(by='Saldo Bruto', ascending=True), x='Saldo Bruto', y='Ticker', orientation='h', title="Peso por Ativo (R$)", color_discrete_sequence=['#006437'])
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("### 👔 Consultoria Especializada (Economista Sênior)")
                    
                    if st.button("🧠 Consultoria: Analisar Minha Carteira BR", type="primary", key="consultoria_br"):
                        if not api_key: st.error("Insira a API Key do Google no menu lateral.")
                        else:
                            with st.spinner("Analisando ciclo econômico, Selic e fundamentos do seu portfólio..."):
                                try:
                                    ind = buscar_indicadores_macro()
                                    texto_tabela = df_fund_br.drop(columns=["Saldo Bruto"]).to_csv(index=False, sep="|")
                                    
                                    genai.configure(api_key=api_key)
                                    modelo_ia = encontrar_modelo_flash()
                                    
                                    prompt_gestor = f"""
                                    Você é o Economista-Chefe e Gestor Sênior do Family Office Tepetos.
                                    Faça uma análise crítica, direta e institucional da seguinte Carteira de Ações Brasileiras do seu cliente.
                                    
                                    CENÁRIO MACRO DE HOJE (ANCORAGEM):
                                    - Taxa Selic: {ind['SELIC']['v']}%
                                    - Inflação (IPCA): {ind['IPCA']['v']}%
                                    
                                    TABELA DO PORTFÓLIO (Com Rentabilidade Real do Cliente e Fundamentos Atuais):
                                    {texto_tabela}
                                    
                                    Sua análise deve conter:
                                    1) Avaliação de Distribuição Setorial: A carteira está bem diversificada para o atual ciclo da Selic? Falta proteção em algum setor? (Cite os setores atuais).
                                    2) Sugestão de Realização de Lucros (Take Profit): Verifique se há ativos com P/L muito alto E Rentabilidade Positiva alta na carteira, que justificariam uma realização de lucros.
                                    3) Alertas de Risco: Identifique ativos que pioraram de fundamento ou estão com margens perigosas.
                                    4) Recomendações de Estudo: Baseado no cenário macro atual, sugira 2 ou 3 ações (ou setores) da B3 que NÃO ESTÃO na carteira, mas que se beneficiariam do ciclo econômico atual.
                                    
                                    Escreva em formato Markdown profissional.
                                    Assine como 'Economista Sênior - Tepetos' Finance'.
                                    """
                                    res = genai.GenerativeModel(modelo_ia).generate_content(prompt_gestor)
                                    st.markdown(f"<div style='background-color: white; padding: 30px; border-left: 5px solid #006437; border-radius: 8px;'>{res.text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                                except Exception as e: st.error(f"Erro na consultoria: {e}")

                st.markdown("---")
                st.markdown("### 🧠 Deep Dive Masterclass ALL-IN-ONE (Ações BR)")
                t_deep_br_input = st.text_input("Digite o Ticker para o Raio-X Completo (ex: B3SA3):", key="deep_br").upper().strip()
                
                if st.button("🔎 Gerar Raio-X Integrado", key="btn_deep_br") and t_deep_br_input:
                    if not api_key: st.error("Insira a API Key do Google no Menu Lateral.")
                    else:
                        with st.spinner("Extraindo Históricos da Bolsa, calculando DCF e desenhando Gráficos..."):
                            t_deep_br = f"{t_deep_br_input}.SA" if not t_deep_br_input.endswith(".SA") else t_deep_br_input
                            
                            try:
                                tk = yf.Ticker(t_deep_br)
                                
                                hist_5y = tk.history(period="5y")
                                if not hist_5y.empty:
                                    st.markdown(f"#### 📈 Evolução da Cotação (Últimos 5 Anos): {t_deep_br_input}")
                                    fig_price = px.line(hist_5y.reset_index(), x='Date', y='Close', labels={'Date': 'Data', 'Close': 'Cotação (R$)'}, color_discrete_sequence=['#006437'])
                                    st.plotly_chart(fig_price, use_container_width=True)
                                    
                                divs = tk.dividends
                                if not divs.empty:
                                    divs_df = pd.DataFrame(divs).reset_index()
                                    divs_df.columns = ['Date', 'Dividends']
                                    divs_df['Year'] = divs_df['Date'].dt.year
                                    divs_yearly = divs_df.groupby('Year')['Dividends'].sum().reset_index()
                                    ano_atual = datetime.date.today().year
                                    divs_yearly = divs_yearly[divs_yearly['Year'] >= (ano_atual - 5)]
                                    if not divs_yearly.empty:
                                        st.markdown(f"#### 💰 Histórico de Pagamento de Dividendos: {t_deep_br_input}")
                                        fig_divs = px.bar(divs_yearly, x='Year', y='Dividends', labels={'Year': 'Ano', 'Dividends': 'Total Pago (R$)'}, color_discrete_sequence=['#FFD700'])
                                        fig_divs.update_layout(xaxis=dict(tickmode='linear', dtick=1))
                                        st.plotly_chart(fig_divs, use_container_width=True)
                            except: pass

                            txt_historico = "Dados trimestrais não encontrados."
                            try:
                                qf = tk.quarterly_financials
                                if qf is not None and not qf.empty:
                                    qf_t = qf.T 
                                    plot_cols = []
                                    if 'Total Revenue' in qf_t.columns: plot_cols.append('Total Revenue')
                                    if 'Net Income' in qf_t.columns: plot_cols.append('Net Income')
                                    
                                    if plot_cols:
                                        qf_plot = qf_t[plot_cols].head(4).sort_index(ascending=True) 
                                        
                                        st.markdown(f"#### 📊 Evolução Trimestral (DRE): {t_deep_br_input}")
                                        fig_hist = px.bar(qf_plot.reset_index(), x='index', y=plot_cols, barmode='group', 
                                                          title="Receita vs Lucro Líquido (Últimos 4 Trimestres)",
                                                          labels={'index': 'Trimestre', 'value': 'Valor (R$)'},
                                                          color_discrete_sequence=['#006437', '#FFD700'])
                                        st.plotly_chart(fig_hist, use_container_width=True)
                                        
                                        txt_historico = qf_plot.to_csv(sep="|")
                            except Exception:
                                pass 
                            
                            try:
                                info = tk.info
                                p_hoje = s_float(info.get('currentPrice', info.get('previousClose', 0)))
                                ind = buscar_indicadores_macro()
                                
                                pl = info.get('trailingPE', 0)
                                pvp = info.get('priceToBook', 0)
                                evebitda = info.get('enterpriseToEbitda', 0)
                                roe = (info.get('returnOnEquity', 0) or 0) * 100
                                
                                div_rate = s_float(info.get('trailingAnnualDividendRate'))
                                if div_rate > 0 and p_hoje > 0:
                                    dy = (div_rate / p_hoje) * 100
                                else:
                                    dy_raw = s_float(info.get('dividendYield', info.get('trailingAnnualDividendYield', 0)))
                                    dy = dy_raw if dy_raw > 0.3 else dy_raw * 100
                                
                                fundamentos_texto = f"""
                                - Cotação Atual: R$ {p_hoje:.2f}
                                - P/L: {round(pl, 2)}
                                - P/VP: {round(pvp, 2)}
                                - EV/EBITDA: {round(evebitda, 2)}
                                - ROE: {round(roe, 2)}%
                                - Dividend Yield: {round(dy, 2)}%
                                """
                            except: 
                                p_hoje = 0.0
                                fundamentos_texto = "Dados fundamentalistas básicos indisponíveis."

                            texto_dcf_para_ia = ""
                            df_dcf, _, erro_dcf = calcular_valuation_dcf_matematico(t_deep_br_input, is_fii=False)
                            if df_dcf is not None and not df_dcf.empty:
                                texto_dcf_para_ia = df_dcf.to_csv(index=False, sep="|")

                            data_atual = datetime.date.today().strftime('%d/%m/%Y')
                            genai.configure(api_key=api_key)
                            
                            with st.spinner("O Economista Chefe está redigindo o Dossiê Final..."):
                                try:
                                    modelo_ia = encontrar_modelo_flash()
                                    prompt_deepdive = f"""
                                    Hoje é {data_atual}. Atue como Analista de Equity Sênior e Head de Valuation Institucional.
                                    Faça o Deep Dive Supremo da ação {t_deep_br_input} (Preço: R$ {p_hoje:.2f}, Selic: {ind['SELIC']['v']}%.

                                    DADOS FORNECIDOS PELO SISTEMA QUANTITATIVO:
                                    1. Fundamentos: {fundamentos_texto}
                                    2. Histórico Trimestral DRE: {txt_historico}
                                    3. Resultados do Valuation Matemático: {texto_dcf_para_ia}

                                    Escreva um Relatório (Dossiê) completo e elegante em Markdown com a seguinte estrutura EXATA:
                                    
                                    ### 1. Visão Geral da Empresa e Mercado
                                    (Explique o que a empresa faz, como ela ganha dinheiro, o mercado em que atua e a situação macro/microeconômica atual desse setor).

                                    ### 2. Análise Competitiva e Concorrentes
                                    (Quais são os principais concorrentes diretos e indiretos? Qual é o tamanho da sua vantagem competitiva - Fosso Econômico/Moat?)

                                    ### 3. Matriz SWOT (Forças, Fraquezas, Oportunidades e Riscos)
                                    (Aponte de forma clara e objetiva os pontos fortes, pontos fracos, oportunidades de crescimento e as ameaças/riscos que o investidor precisa monitorar).

                                    ### 4. Governança Corporativa e Gestão
                                    (Avalie a diretoria, o alinhamento de interesses com o acionista minoritário, estrutura de capital e nível de governança corporativa).

                                    ### 5. Tabela de Fundamentos Básicos
                                    (Crie a tabela resumo de forma limpa e bonita com as métricas informadas acima).

                                    ### 6. Evolução Histórica (Receita, Lucro e Dividendos)
                                    (Comente a evolução dos lucros trimestrais que te passei e as perspectivas. Avalie a consistência na distribuição de dividendos. Aponte SINAIS DE ALERTA 🚨).

                                    ### 7. Valuation e Preço Justo
                                    (Apresente a tabela de Valuation com os dados exatos calculados pelo Python acima: Cenário | Retorno Exigido | Cresc. Perpétuo (g) | Preço Justo | Upside / Margem). 
                                    (Se for Banco, use DDM e ignore os números do Python se estiverem distorcidos).
                                    Comente a Margem de Segurança do cenário base.

                                    ### 8. Parecer Final do Analista
                                    (Sua conclusão de ouro: onde esta empresa estará em 5 anos? Vale o risco frente ao cenário macro?).

                                    Assine EXATAMENTE: 'Com carinho, Tepeto'
                                    """
                                    res = genai.GenerativeModel(modelo_ia).generate_content(prompt_deepdive)
                                    st.success("Dossiê Deep Dive Gerado com Sucesso!")
                                    st.markdown(f"<div style='background-color: white; padding: 30px; border-left: 5px solid #006437; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>{res.text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                                except Exception as e: st.error(f"Erro no Deep Dive: {e}")

        # ==========================================
        # SUB-ABA 2: STOCKS AMERICANAS
        # ==========================================
        with sub_us:
            st.subheader("🇺🇸 Stocks e BDRs na sua Carteira")
            
            with closing(get_conn()) as conn:
                df_us = pd.read_sql("SELECT ticker, valor_aplicado, saldo_bruto FROM carteira WHERE conta=? AND (classe_ativo IN ('Ação', 'Bdr', 'Outros')) AND ticker NOT LIKE '%.SA'", conn, params=(conta_selecionada,))
            
            if df_us.empty:
                st.info("Você não possui Stocks Americanas cadastradas na sua carteira (ativos sem o .SA).")
            else:
                tot_aplicado_us = df_us['valor_aplicado'].sum()
                tot_atual_us = df_us['saldo_bruto'].sum()
                rent_us = ((tot_atual_us / tot_aplicado_us) - 1) * 100 if tot_aplicado_us > 0 else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Valor Aplicado (US)", f"$ {tot_aplicado_us:,.2f}")
                c2.metric("Saldo Atual (US)", f"$ {tot_atual_us:,.2f}")
                c3.metric("Rentabilidade (Total Acumulada)", f"{rent_us:.2f}%")
                
                st.markdown("---")
                st.markdown("### 📋 Tabela Fundamentalista e Setorial")
                if st.button("🔄 Atualizar Fundamentos (Stocks)"):
                    with st.spinner("Buscando dados no Yahoo Finance..."):
                        fund_us_list = []
                        for t in df_us['ticker'].unique():
                            try:
                                info = yf.Ticker(t).info
                                p = s_float(info.get('currentPrice', info.get('previousClose', 0)))
                                pl = info.get('trailingPE', 0)
                                pvp = info.get('priceToBook', 0)
                                roe = (info.get('returnOnEquity', 0) or 0) * 100
                                
                                div_rate = s_float(info.get('trailingAnnualDividendRate'))
                                if div_rate > 0 and p > 0:
                                    dy = (div_rate / p) * 100
                                else:
                                    dy_raw = s_float(info.get('dividendYield', info.get('trailingAnnualDividendYield', 0)))
                                    dy = dy_raw if dy_raw > 0.3 else dy_raw * 100
                                    
                                gross_m = (info.get('grossMargins', 0) or 0) * 100
                                op_m = (info.get('operatingMargins', 0) or 0) * 100
                                setor = str(info.get('sector', 'Desconhecido'))
                                
                                aplicado_no_app = float(df_us[df_us['ticker'] == t]['valor_aplicado'].sum())
                                saldo_no_app = float(df_us[df_us['ticker'] == t]['saldo_bruto'].sum())
                                rent_ativo = ((saldo_no_app / aplicado_no_app) - 1) * 100 if aplicado_no_app > 0 else 0
                                
                                fund_us_list.append({
                                    "Ticker": t,
                                    "Setor": setor,
                                    "Preço Atual": f"$ {p:.2f}",
                                    "Rentabilidade (%)": f"{rent_ativo:.2f}%",
                                    "P/L": round(pl, 2) if pl else "-",
                                    "P/VP": round(pvp, 2) if pvp else "-",
                                    "ROE (%)": round(roe, 2) if roe else "-",
                                    "Gross Margin (%)": round(gross_m, 2) if gross_m else "-",
                                    "Operating Margin (%)": round(op_m, 2) if op_m else "-",
                                    "Div. Yield (%)": round(dy, 2) if dy else "-",
                                    "Saldo Bruto": saldo_no_app
                                })
                            except: pass
                        
                        if fund_us_list:
                            st.session_state[f'fund_us_cache_{conta_selecionada}'] = pd.DataFrame(fund_us_list)
                            
                if f'fund_us_cache_{conta_selecionada}' in st.session_state:
                    df_fund_us = st.session_state[f'fund_us_cache_{conta_selecionada}']
                    st.dataframe(df_fund_us.drop(columns=["Saldo Bruto"]), use_container_width=True, hide_index=True)
                    
                    st.write("")
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_setor = px.pie(df_fund_us, values='Saldo Bruto', names='Setor', hole=0.4, title="Exposição por Setor ($)", color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_setor, use_container_width=True)
                    with col_g2:
                        fig_bar = px.bar(df_fund_us.sort_values(by='Saldo Bruto', ascending=True), x='Saldo Bruto', y='Ticker', orientation='h', title="Peso por Ativo ($)", color_discrete_sequence=['#006437'])
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                    st.markdown("---")
                    st.markdown("### 👔 Consultoria Especializada (Economista Internacional)")
                    
                    if st.button("🧠 Consultoria: Analisar Minha Carteira Global", type="primary", key="consultoria_us"):
                        if not api_key: st.error("Insira a API Key do Google.")
                        else:
                            with st.spinner("Analisando taxa de juros do Fed e fundamentos de Wall Street..."):
                                try:
                                    ind = buscar_indicadores_macro()
                                    texto_tabela = df_fund_us.drop(columns=["Saldo Bruto"]).to_csv(index=False, sep="|")
                                    
                                    genai.configure(api_key=api_key)
                                    modelo_ia = encontrar_modelo_flash()
                                    
                                    prompt_gestor = f"""
                                    Você é o Gestor de Equities Globais do Family Office Tepetos.
                                    Analise criticamente a Carteira de Stocks Americanas do seu cliente.
                                    
                                    TABELA DO PORTFÓLIO (Rentabilidade e Fundamentos):
                                    {texto_tabela}
                                    
                                    Sua análise deve conter:
                                    1) Diversificação e Risco Setorial: Estamos muito concentrados em Tech? Há necessidade de diversificação defensiva?
                                    2) Take Profit e Avaliação de Múltiplos: Aponte as ações que esticaram o P/L e já deram muito lucro. Recomende se é hora de manter ou realizar parte do lucro.
                                    3) Sugestões de Mercado (Stock Picking): Com o cenário de juros americanos atual, quais 2 ou 3 ações fora da carteira seriam excelentes barganhas para adicionar ao radar?
                                    
                                    Escreva em formato Markdown.
                                    Assine como 'Gestor Internacional - Tepetos' Finance'.
                                    """
                                    res = genai.GenerativeModel(modelo_ia).generate_content(prompt_gestor)
                                    st.markdown(f"<div style='background-color: white; padding: 30px; border-left: 5px solid #006437; border-radius: 8px;'>{res.text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                                except Exception as e: st.error(f"Erro na consultoria: {e}")
                
                st.markdown("---")
                
                st.markdown("### 🧠 Deep Dive Masterclass ALL-IN-ONE (Stocks EUA)")
                st.caption("Gera Gráficos Históricos, DRE, Tabela de Fundamentos, Análise Qualitativa SWOT e Valuation Matemático.")
                t_deep_us_input = st.text_input("Digite o Ticker para o Raio-X Completo (ex: AAPL, EQX):", key="deep_us_input").upper().strip()
                
                if st.button("🔎 Gerar Raio-X Integrado", key="btn_deep_us") and t_deep_us_input:
                    if not api_key: st.error("Insira a API Key do Google.")
                    else:
                        with st.spinner("Extraindo balanços globais, desenhando Gráficos e projetando Valuation..."):
                            
                            try:
                                tk = yf.Ticker(t_deep_us_input)
                                
                                hist_5y = tk.history(period="5y")
                                if not hist_5y.empty:
                                    st.markdown(f"#### 📈 Evolução da Cotação (Últimos 5 Anos): {t_deep_us_input}")
                                    fig_price = px.line(hist_5y.reset_index(), x='Date', y='Close', labels={'Date': 'Data', 'Close': 'Cotação (USD $)'}, color_discrete_sequence=['#006437'])
                                    st.plotly_chart(fig_price, use_container_width=True)
                                    
                                divs = tk.dividends
                                if not divs.empty:
                                    divs_df = pd.DataFrame(divs).reset_index()
                                    divs_df.columns = ['Date', 'Dividends']
                                    divs_df['Year'] = divs_df['Date'].dt.year
                                    divs_yearly = divs_df.groupby('Year')['Dividends'].sum().reset_index()
                                    ano_atual = datetime.date.today().year
                                    divs_yearly = divs_yearly[divs_yearly['Year'] >= (ano_atual - 5)]
                                    if not divs_yearly.empty:
                                        st.markdown(f"#### 💰 Histórico de Pagamento de Dividendos: {t_deep_us_input}")
                                        fig_divs = px.bar(divs_yearly, x='Year', y='Dividends', labels={'Year': 'Ano', 'Dividends': 'Total Pago (USD $)'}, color_discrete_sequence=['#FFD700'])
                                        fig_divs.update_layout(xaxis=dict(tickmode='linear', dtick=1))
                                        st.plotly_chart(fig_divs, use_container_width=True)
                                
                                info = tk.info
                                p_hoje = s_float(info.get('currentPrice', info.get('previousClose', 0)))
                                ind = buscar_indicadores_macro()
                                
                                pl = info.get('trailingPE', 0)
                                pvp = info.get('priceToBook', 0)
                                roe = (info.get('returnOnEquity', 0) or 0) * 100
                                
                                div_rate = s_float(info.get('trailingAnnualDividendRate'))
                                if div_rate > 0 and p_hoje > 0:
                                    dy = (div_rate / p_hoje) * 100
                                else:
                                    dy_raw = s_float(info.get('dividendYield', info.get('trailingAnnualDividendYield', 0)))
                                    dy = dy_raw if dy_raw > 0.3 else dy_raw * 100
                                
                                fundamentos_texto = f"""
                                - Cotação Atual: $ {p_hoje:.2f}
                                - P/L: {round(pl, 2)}
                                - P/VP: {round(pvp, 2)}
                                - ROE: {round(roe, 2)}%
                                - Dividend Yield: {round(dy, 2)}%
                                """
                            except: 
                                p_hoje = 0.0
                                fundamentos_texto = "Dados fundamentalistas indisponíveis."

                            txt_historico = "Dados trimestrais não encontrados."
                            try:
                                qf = tk.quarterly_financials
                                if qf is not None and not qf.empty:
                                    qf_t = qf.T
                                    plot_cols = []
                                    if 'Total Revenue' in qf_t.columns: plot_cols.append('Total Revenue')
                                    if 'Net Income' in qf_t.columns: plot_cols.append('Net Income')
                                    
                                    if plot_cols:
                                        qf_plot = qf_t[plot_cols].head(4).sort_index(ascending=True) 
                                        
                                        st.markdown(f"#### 📊 Evolução Trimestral: {t_deep_us_input}")
                                        fig_hist = px.bar(qf_plot.reset_index(), x='index', y=plot_cols, barmode='group', 
                                                          title="Revenue vs Net Income (Last 4 Quarters)",
                                                          labels={'index': 'Quarter', 'value': 'USD ($)'},
                                                          color_discrete_sequence=['#006437', '#FFD700'])
                                        st.plotly_chart(fig_hist, use_container_width=True)
                                        txt_historico = qf_plot.to_csv(sep="|")
                            except Exception: pass

                            texto_dcf_para_ia = ""
                            df_dcf, _, _ = calcular_valuation_dcf_matematico(t_deep_us_input, is_fii=False)
                            if df_dcf is not None and not df_dcf.empty:
                                texto_dcf_para_ia = df_dcf.to_csv(index=False, sep="|")
                                
                            data_atual = datetime.date.today().strftime('%d/%m/%Y')
                            genai.configure(api_key=api_key)
                            
                            with st.spinner("O Analista Global está desenhando o Dossiê Final..."):
                                try:
                                    modelo_ia = encontrar_modelo_flash()
                                    prompt_deepdive = f"""
                                    Hoje é {data_atual}. Atue como Analista de Equity Global e Head de Valuation Institucional.
                                    Faça o Deep Dive Supremo da Stock americana {t_deep_us_input} (Preço: $ {p_hoje:.2f}).

                                    DADOS QUANTITATIVOS COLETADOS:
                                    1. Fundamentos: {fundamentos_texto}
                                    2. Histórico Trimestral SEC: {txt_historico}
                                    3. DCF Matemático Calculado: {texto_dcf_para_ia}

                                    Escreva o relatório OBRIGATORIAMENTE em Markdown com a estrutura exata:
                                    
                                    ### 1. Visão Geral da Empresa e Mercado Global
                                    (Explique o que a empresa faz, como ganha dinheiro, tamanho do mercado de atuação).

                                    ### 2. Análise Competitiva e Concorrentes
                                    (Quais os principais players do setor e qual o Fosso Econômico/Moat desta Stock?)

                                    ### 3. Matriz SWOT Global
                                    (Aponte de forma direta: Forças, Fraquezas, Oportunidades Tecnológicas/Mercadológicas e Ameaças/Riscos macro).

                                    ### 4. Governança e Estrutura
                                    (Avalie a qualidade da gestão e proteção ao acionista).

                                    ### 5. Tabela de Fundamentos Básicos
                                    (Crie a tabela resumo de forma limpa com P/L, ROE, DY, etc).

                                    ### 6. Evolução Histórica (Receita, Lucro e Dividendos)
                                    (Comente o histórico DRE que te passei e o histórico de dividendos que o cliente vê nos gráficos. Aponte SINAIS DE ALERTA 🚨).

                                    ### 7. Valuation e Preço Justo
                                    (Use a tabela fornecida pelo DCF Matemático acima com as colunas: Cenário | Retorno Exigido | Cresc. Perpétuo (g) | Preço Justo | Upside / Margem). 
                                    Comente a Margem de Segurança do cenário base.

                                    ### 8. Parecer Final do Analista
                                    (A conclusão de ouro: vale a pena ser sócio para os próximos 5 anos?).

                                    Assine EXATAMENTE: 'Com carinho, Tepeto'
                                    """
                                    res = genai.GenerativeModel(modelo_ia).generate_content(prompt_deepdive)
                                    st.success("Dossiê Deep Dive Gerado com Sucesso!")
                                    st.markdown(f"<div style='background-color: white; padding: 30px; border-left: 5px solid #006437; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>{res.text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                                except Exception as e: st.error(f"Erro no Deep Dive: {e}")

        # ==========================================
        # SUB-ABA 3: FIIs E REITs
        # ==========================================
        with sub_fii:
            st.subheader("🏢 Fundos Imobiliários e REITs")
            
            with closing(get_conn()) as conn:
                df_fii = pd.read_sql("SELECT ticker, valor_aplicado, saldo_bruto FROM carteira WHERE conta=? AND classe_ativo='Fundo Imobiliário'", conn, params=(conta_selecionada,))
            
            if df_fii.empty:
                st.info("Você não possui FIIs cadastrados na sua carteira.")
            else:
                tot_aplicado_fii = df_fii['valor_aplicado'].sum()
                tot_atual_fii = df_fii['saldo_bruto'].sum()
                rent_fii = ((tot_atual_fii / tot_aplicado_fii) - 1) * 100 if tot_aplicado_fii > 0 else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Valor Aplicado (FIIs)", formata_moeda(tot_aplicado_fii))
                c2.metric("Saldo Atual (FIIs)", formata_moeda(tot_atual_fii))
                c3.metric("Rentabilidade (Total Acumulada)", f"{rent_fii:.2f}%")
                
                st.markdown("---")
                st.markdown("### 📋 Tabela Fundamentalista e Setorial")
                st.caption("*Nota: APIs públicas limitam dados profundos de FIIs (como Vacância e FFO). O Deep Dive cobrirá esses pontos qualitativamente.*")
                
                if st.button("🔄 Atualizar P/VP e Dividendos (FIIs)"):
                    with st.spinner("Buscando dados no Yahoo Finance..."):
                        fund_fii_list = []
                        for t in df_fii['ticker'].unique():
                            try:
                                info = yf.Ticker(t).info
                                p = s_float(info.get('currentPrice', info.get('previousClose', 0)))
                                pvp = info.get('priceToBook', 0)
                                
                                div_rate = s_float(info.get('trailingAnnualDividendRate'))
                                if div_rate > 0 and p > 0:
                                    dy = (div_rate / p) * 100
                                else:
                                    dy_raw = s_float(info.get('dividendYield', info.get('trailingAnnualDividendYield', 0)))
                                    dy = dy_raw if dy_raw > 0.3 else dy_raw * 100
                                    
                                setor = str(info.get('sector', info.get('industry', 'Fundo Imobiliário')))
                                
                                aplicado_no_app = float(df_fii[df_fii['ticker'] == t]['valor_aplicado'].sum())
                                saldo_no_app = float(df_fii[df_fii['ticker'] == t]['saldo_bruto'].sum())
                                rent_ativo = ((saldo_no_app / aplicado_no_app) - 1) * 100 if aplicado_no_app > 0 else 0
                                
                                fund_fii_list.append({
                                    "Ticker": t.replace('.SA', ''),
                                    "Setor/Segmento": setor,
                                    "Preço Atual": f"R$ {p:.2f}",
                                    "Rentabilidade (%)": f"{rent_ativo:.2f}%",
                                    "P/VP": round(pvp, 2) if pvp else "-",
                                    "Dividend Yield (12m)": f"{round(dy, 2)}%" if dy else "-",
                                    "Saldo Bruto": saldo_no_app
                                })
                            except: pass
                        
                        if fund_fii_list:
                            st.session_state[f'fund_fii_cache_{conta_selecionada}'] = pd.DataFrame(fund_fii_list)
                            
                if f'fund_fii_cache_{conta_selecionada}' in st.session_state:
                    df_fund_fii = st.session_state[f'fund_fii_cache_{conta_selecionada}']
                    st.dataframe(df_fund_fii.drop(columns=["Saldo Bruto"]), use_container_width=True, hide_index=True)
                    
                    st.write("")
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_setor = px.pie(df_fund_fii, values='Saldo Bruto', names='Setor/Segmento', hole=0.4, title="Exposição por Segmento (R$)", color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_setor, use_container_width=True)
                    with col_g2:
                        fig_bar = px.bar(df_fund_fii.sort_values(by='Saldo Bruto', ascending=True), x='Saldo Bruto', y='Ticker', orientation='h', title="Peso por Fundo (R$)", color_discrete_sequence=['#006437'])
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                    st.markdown("---")
                    st.markdown("### 👔 Consultoria Especializada (Gestor de FIIs)")
                    
                    if st.button("🧠 Consultoria: Analisar Minha Carteira de FIIs", type="primary", key="consultoria_fii"):
                        if not api_key: st.error("Insira a API Key do Google.")
                        else:
                            with st.spinner("Analisando prêmio de risco, IFIX e correlação com a Selic..."):
                                try:
                                    ind = buscar_indicadores_macro()
                                    texto_tabela = df_fund_fii.drop(columns=["Saldo Bruto"]).to_csv(index=False, sep="|")
                                    
                                    genai.configure(api_key=api_key)
                                    modelo_ia = encontrar_modelo_flash()
                                    
                                    prompt_gestor = f"""
                                    Você é o Gestor Especialista em Fundos Imobiliários do Family Office Tepetos.
                                    Analise criticamente a Carteira de FIIs do seu cliente.
                                    
                                    CENÁRIO MACRO DE HOJE (ANCORAGEM):
                                    - Taxa Selic: {ind['SELIC']['v']}%
                                    - Inflação (IPCA): {ind['IPCA']['v']}%
                                    
                                    TABELA DO PORTFÓLIO (Rentabilidade, P/VP, DY):
                                    {texto_tabela}
                                    
                                    Sua análise deve focar em:
                                    1) Estratégia de Alocação: A carteira está muito focada em Papel (CRI) ou Tijolo? Com a Selic atual, essa estratégia faz sentido?
                                    2) Avaliação de Preço: Existem fundos com P/VP muito acima de 1.05 que deveriam ser parados de aportar ou reduzidos?
                                    3) Risco e Oportunidade: Avalie o prêmio de risco dos Dividendos frente à Selic.
                                    4) Sugestões de Estudo: Quais classes ou 2 FIIs específicos (Logística, Renda Urbana, Shopping, Papel High Grade) o cliente deveria estudar para balancear essa carteira?
                                    
                                    Escreva em formato Markdown profissional.
                                    Assine como 'Gestor de FIIs - Tepetos' Finance'.
                                    """
                                    res = genai.GenerativeModel(modelo_ia).generate_content(prompt_gestor)
                                    st.markdown(f"<div style='background-color: white; padding: 30px; border-left: 5px solid #006437; border-radius: 8px;'>{res.text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                                except Exception as e: st.error(f"Erro na consultoria: {e}")
                
                st.markdown("---")
                
                st.markdown("### 🧠 Deep Dive Masterclass ALL-IN-ONE (FIIs)")
                st.caption("Gera Gráficos Históricos, Tabela Básica, Qualitativa de Vacância/Imóveis, Sinais de Alerta e Valuation pelo Modelo de Gordon.")
                t_deep_fii_input = st.text_input("Digite o Ticker para o Raio-X do Imóvel/Papel (ex: HGLG11):", key="deep_fii_input").upper().strip()
                
                if st.button("🔎 Gerar Raio-X Integrado", key="btn_deep_fii") and t_deep_fii_input:
                    if not api_key: st.error("Insira a API Key.")
                    else:
                        with st.spinner("Analisando cotações, dividendos e precificando via Modelo de Gordon Matemático..."):
                            t_deep_fii = f"{t_deep_fii_input}.SA" if not t_deep_fii_input.endswith(".SA") else t_deep_fii_input
                            try:
                                tk = yf.Ticker(t_deep_fii)
                                
                                hist_5y = tk.history(period="5y")
                                if not hist_5y.empty:
                                    st.markdown(f"#### 📈 Evolução da Cotação (Últimos 5 Anos): {t_deep_fii_input}")
                                    fig_price = px.line(hist_5y.reset_index(), x='Date', y='Close', labels={'Date': 'Data', 'Close': 'Cotação (R$)'}, color_discrete_sequence=['#006437'])
                                    st.plotly_chart(fig_price, use_container_width=True)
                                    
                                divs = tk.dividends
                                if not divs.empty:
                                    divs_df = pd.DataFrame(divs).reset_index()
                                    divs_df.columns = ['Date', 'Dividends']
                                    divs_df['Year'] = divs_df['Date'].dt.year
                                    divs_yearly = divs_df.groupby('Year')['Dividends'].sum().reset_index()
                                    ano_atual = datetime.date.today().year
                                    divs_yearly = divs_yearly[divs_yearly['Year'] >= (ano_atual - 5)]
                                    if not divs_yearly.empty:
                                        st.markdown(f"#### 💰 Histórico de Pagamento de Rendimentos Anuais: {t_deep_fii_input}")
                                        fig_divs = px.bar(divs_yearly, x='Year', y='Dividends', labels={'Year': 'Ano', 'Dividends': 'Total Pago (R$)'}, color_discrete_sequence=['#FFD700'])
                                        fig_divs.update_layout(xaxis=dict(tickmode='linear', dtick=1))
                                        st.plotly_chart(fig_divs, use_container_width=True)
                                
                                info = tk.info
                                p_hoje = s_float(info.get('currentPrice', info.get('previousClose', 0)))
                                pvp = info.get('priceToBook', 0)
                                div_rate = s_float(info.get('trailingAnnualDividendRate'))
                                if div_rate > 0 and p_hoje > 0:
                                    dy = (div_rate / p_hoje) * 100
                                else:
                                    dy_raw = s_float(info.get('dividendYield', info.get('trailingAnnualDividendYield', 0)))
                                    dy = dy_raw if dy_raw > 0.3 else dy_raw * 100
                                ind = buscar_indicadores_macro()
                                
                                fundamentos_texto = f"""
                                - Cotação Atual: R$ {p_hoje:.2f}
                                - P/VP Atual: {round(pvp, 2)}
                                - Dividend Yield Projetado: {round(dy, 2)}%
                                """
                            except: 
                                p_hoje = 0.0
                                fundamentos_texto = "Dados fundamentalistas básicos indisponíveis."
                            
                            texto_dcf_para_ia = ""
                            df_dcf, _, _ = calcular_valuation_dcf_matematico(t_deep_fii_input, is_fii=True)
                            if df_dcf is not None and not df_dcf.empty:
                                texto_dcf_para_ia = df_dcf.to_csv(index=False, sep="|")
                                
                            data_atual = datetime.date.today().strftime('%d/%m/%Y')
                            genai.configure(api_key=api_key)
                            
                            try:
                                modelo_ia = encontrar_modelo_flash()
                                prompt_deepdive = f"""
                                Hoje é {data_atual}. Atue como Analista de Fundos Imobiliários e Head de Valuation.
                                Faça o Dossiê Supremo do FII {t_deep_fii_input} (Cotação atual: R$ {p_hoje:.2f}, Selic atual: {ind['SELIC']['v']}%).
                                
                                **DADOS COLETADOS (NÃO ALTERE ESTES NÚMEROS):**
                                Fundamentos: {fundamentos_texto}
                                Gordon Model Calculado pelo Sistema: {texto_dcf_para_ia}
                                
                                **ESTRUTURA OBRIGATÓRIA DO RELATÓRIO EM MARKDOWN:**
                                
                                ### 1. Visão Geral do Fundo e Segmento
                                (Explique a tese do fundo, se é papel/tijolo, híbrido, galpões, shoppings e como a Selic atual impacta esse segmento).
                                
                                ### 2. Análise Competitiva do Portfólio
                                (Qualidade dos imóveis, localização ou perfil de risco da carteira de CRIs).
                                
                                ### 3. Matriz SWOT e Alertas de Risco 🚨
                                (Forças, Fraquezas, Oportunidades. Aponte SINAIS DE ALERTA sobre Vacância Física/Financeira, concentração de inquilinos ou calotes).

                                ### 4. Gestão e Governança
                                (Avalie a qualidade do Administrador e do Gestor, taxa de gestão/performance).

                                ### 5. Tabela de Fundamentos Básicos
                                (Crie a tabela resumo de P/VP e DY com os dados coletados).
                                
                                ### 6. Evolução Histórica de Rendimentos
                                (Comente o histórico de dividendos que o usuário vê no gráfico acima. É consistente?).

                                ### 7. Valuation e Preço Justo (Modelo de Gordon)
                                (Crie a Tabela EXATA com os dados do Gordon Model acima). Colunas: `| Cenário | Retorno Exigido (%) | Cresc. Dividendo (g) | Preço Justo | Upside / Margem |`.
                                
                                ### 8. Parecer Final e Perspectivas
                                (A conclusão de ouro: este fundo compensa o prêmio de risco frente à Renda Fixa hoje?).
                                
                                Assine EXATAMENTE: 'Com carinho, Tepeto'
                                """
                                res = genai.GenerativeModel(modelo_ia).generate_content(prompt_deepdive)
                                st.success("Dossiê Deep Dive FII concluído!")
                                st.markdown(f"<div style='background-color: white; padding: 30px; border-left: 5px solid #006437; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>{res.text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                            except Exception as e: st.error(f"Erro no Deep Dive: {e}")

    # 🔥 VERSÃO 22.2: MORNING CALL COM CORES CORRETAS E BANCADA DO AGRO 🔥
    with tab_macro:
        st.header("🌍 Cenário Macro & Boletins Inteligentes")
        st.markdown("### Radar Econômico Global (Tempo Real)")
        indicadores = buscar_indicadores_macro()
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("💵 Dólar", f"R$ {indicadores['USD']['v']:.2f}", f"{indicadores['USD']['d']:.2f}%")
        col_m2.metric("💶 Euro", f"R$ {indicadores['EUR']['v']:.2f}", f"{indicadores['EUR']['d']:.2f}%")
        col_m3.metric("💷 Libra", f"R$ {indicadores['GBP']['v']:.2f}", f"{indicadores['GBP']['d']:.2f}%")
        col_m4.metric("🏦 Taxa Selic", f"{indicadores['SELIC']['v']}%", "BCB", delta_color="off")
        col_m5.metric("🛒 IPCA (Mês)", f"{indicadores['IPCA']['v']}%", "BCB", delta_color="off")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Termômetro das Bolsas Mundiais")
        col_m6, col_m7, col_m8 = st.columns(3)
        col_m6.metric("🇧🇷 IBOVESPA", f"{indicadores.get('IBOV', {}).get('v', 0):,.0f} pts", f"{indicadores.get('IBOV', {}).get('d', 0):.2f}%")
        col_m7.metric("🇺🇸 S&P 500", f"{indicadores.get('SP500', {}).get('v', 0):,.0f} pts", f"{indicadores.get('SP500', {}).get('d', 0):.2f}%")
        col_m8.metric("🇺🇸 NASDAQ", f"{indicadores.get('NASDAQ', {}).get('v', 0):,.0f} pts", f"{indicadores.get('NASDAQ', {}).get('d', 0):.2f}%")
        
        st.markdown("---")
        
        def calcular_rentabilidade_diaria_carteira(conta):
            with closing(get_conn()) as conn:
                df_cart = pd.read_sql("SELECT ticker, classe_ativo, saldo_bruto FROM carteira WHERE conta=?", conn, params=(conta,))
            
            if df_cart.empty: return None
            
            dados_ativos = []
            for _, row in df_cart.iterrows():
                t_orig = str(row['ticker']).strip().upper()
                c_ativo = str(row['classe_ativo']).title().strip()
                saldo = float(row['saldo_bruto'])
                if saldo <= 0: continue
                
                t_yf = t_orig
                if c_ativo in ["Ação", "Acao", "Fundo Imobiliário", "Fundo Imobiliario", "Bdr"] and t_orig[-1].isdigit() and not t_orig.endswith(".SA"):
                    t_yf = f"{t_orig}.SA"
                elif c_ativo == "Cripto" and not t_orig.endswith("-USD"):
                    t_yf = f"{t_orig}-USD"
                    
                try:
                    hist = yf.Ticker(t_yf).history(period="5d")
                    if not hist.empty and len(hist) >= 2:
                        p_hoje = hist['Close'].iloc[-1]
                        p_ontem = hist['Close'].iloc[-2]
                        var_pct = ((p_hoje / p_ontem) - 1) * 100
                        
                        dados_ativos.append({
                            'Ticker': t_orig,
                            'Classe': c_ativo,
                            'Variação (%)': var_pct,
                            'Saldo': saldo
                        })
                except: pass
            
            if not dados_ativos: return None
            
            df_dados = pd.DataFrame(dados_ativos)
            
            total_saldo = df_dados['Saldo'].sum()
            df_dados['Peso'] = df_dados['Saldo'] / total_saldo
            rent_global = (df_dados['Variação (%)'] * df_dados['Peso']).sum()
            
            df_br = df_dados[df_dados['Ticker'].str.endswith('.SA') & (df_dados['Classe'] == 'Ação')]
            rent_br = (df_br['Variação (%)'] * (df_br['Saldo'] / df_br['Saldo'].sum())).sum() if not df_br.empty else 0.0
            
            df_us = df_dados[~df_dados['Ticker'].str.endswith('.SA') & df_dados['Classe'].isin(['Ação', 'Bdr', 'Outros'])]
            rent_us = (df_us['Variação (%)'] * (df_us['Saldo'] / df_us['Saldo'].sum())).sum() if not df_us.empty else 0.0
            
            df_fii = df_dados[df_dados['Classe'].str.contains('Fundo Imob')]
            rent_fii = (df_fii['Variação (%)'] * (df_fii['Saldo'] / df_fii['Saldo'].sum())).sum() if not df_fii.empty else 0.0
            
            df_cripto = df_dados[df_dados['Classe'].str.contains('Cripto')]
            rent_cripto = (df_cripto['Variação (%)'] * (df_cripto['Saldo'] / df_cripto['Saldo'].sum())).sum() if not df_cripto.empty else 0.0
            
            return {
                "Global": rent_global,
                "Ações BR": rent_br if not df_br.empty else None,
                "Stocks EUA": rent_us if not df_us.empty else None,
                "FIIs": rent_fii if not df_fii.empty else None,
                "Cripto": rent_cripto if not df_cripto.empty else None,
                "df": df_dados
            }

        st.markdown("### 📊 Termômetro Diário da Sua Carteira (Fechamento Anterior)")
        with st.spinner("Calculando o fechamento do dia da sua carteira..."):
            rent_diaria = calcular_rentabilidade_diaria_carteira(conta_selecionada)
            
        if rent_diaria:
            cr1, cr2, cr3, cr4, cr5 = st.columns(5)
            
            def formata_rent(val):
                if val is None: return "N/A", "off"
                return f"{val:+.2f}%", "normal"
                
            v_glob, c_glob = formata_rent(rent_diaria["Global"])
            v_br, c_br = formata_rent(rent_diaria["Ações BR"])
            v_us, c_us = formata_rent(rent_diaria["Stocks EUA"])
            v_fii, c_fii = formata_rent(rent_diaria["FIIs"])
            v_cripto, c_cripto = formata_rent(rent_diaria["Cripto"])
            
            cr1.metric("Carteira Global", v_glob, v_glob, delta_color=c_glob)
            cr2.metric("Ações BR", v_br, v_br, delta_color=c_br)
            cr3.metric("Stocks EUA", v_us, v_us, delta_color=c_us)
            cr4.metric("FIIs", v_fii, v_fii, delta_color=c_fii)
            cr5.metric("Cripto", v_cripto, v_cripto, delta_color=c_cripto)
        else:
            st.info("Sua carteira está vazia ou os mercados não tiveram oscilação registrada no último pregão.")
            
        st.markdown("---")

        def get_portfolio_performance_string(conta, periodo="5d"):
            resumo = ""
            with closing(get_conn()) as conn:
                df_cart = pd.read_sql("SELECT ticker, classe_ativo FROM carteira WHERE conta=?", conn, params=(conta,))
            
            if df_cart.empty:
                return "A carteira está vazia neste momento."
                
            classes_dados = {}
            for _, row in df_cart.iterrows():
                t_orig = str(row['ticker']).strip().upper()
                c_ativo = str(row['classe_ativo']).title().strip()
                
                t_yf = t_orig
                if c_ativo in ["Ação", "Acao", "Fundo Imobiliário", "Fundo Imobiliario", "Bdr"] and t_orig[-1].isdigit() and not t_orig.endswith(".SA"):
                    t_yf = f"{t_orig}.SA"
                elif c_ativo == "Cripto" and not t_orig.endswith("-USD"):
                    t_yf = f"{t_orig}-USD"
                    
                try:
                    hist = yf.Ticker(t_yf).history(period=periodo)
                    if not hist.empty and len(hist) >= 2:
                        preco_atual = hist['Close'].iloc[-1]
                        var_pct = ((preco_atual / hist['Close'].iloc[-2]) - 1) * 100
                        
                        if c_ativo not in classes_dados:
                            classes_dados[c_ativo] = []
                            
                        classes_dados[c_ativo].append({
                            'ticker': t_orig,
                            'preco': preco_atual,
                            'var': var_pct
                        })
                except:
                    pass
                    
            if not classes_dados:
                return "Não foi possível extrair a variação recente via Yahoo Finance."
                
            for classe, ativos in classes_dados.items():
                resumo += f"\n[{classe}]\n"
                for a in ativos:
                    resumo += f"- {a['ticker']}: Preço R$ {a['preco']:.2f} | Variação Diária: {a['var']:+.2f}%\n"
                    
            return resumo

        meses_pt = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
        mes_atual = meses_pt[datetime.date.today().month - 1]
        ano_atual = datetime.date.today().year

        col_b0, col_b1, col_b2 = st.columns(3)
        with col_b0:
            btn_diario = st.button("☀️ Morning Call Diário", type="primary", use_container_width=True)
        with col_b1:
            btn_semanal = st.button("📊 Gerar Boletim Semanal", use_container_width=True)
        with col_b2:
            btn_mensal = st.button("📈 Gerar Resumo Mensal", use_container_width=True)
            
        if btn_diario:
            if not api_key: st.error("Insira a chave API da Google no menu lateral.")
            else:
                with st.spinner("Compilando as notícias globais e montando o seu Morning Call exclusivo..."):
                    try:
                        dados_performance = get_portfolio_performance_string(conta_selecionada, "5d")
                        
                        panorama_mundial = f"""
                        - Dólar: R$ {indicadores['USD']['v']:.2f} ({indicadores['USD']['d']:+.2f}%)
                        - Selic: {indicadores['SELIC']['v']}%
                        - IBOVESPA (BR): {indicadores.get('IBOV', {}).get('v', 0):,.0f} pts ({indicadores.get('IBOV', {}).get('d', 0):+.2f}%)
                        - S&P 500 (EUA): {indicadores.get('SP500', {}).get('v', 0):,.0f} pts ({indicadores.get('SP500', {}).get('d', 0):+.2f}%)
                        - NASDAQ (EUA): {indicadores.get('NASDAQ', {}).get('v', 0):,.0f} pts ({indicadores.get('NASDAQ', {}).get('d', 0):+.2f}%)
                        - NIKKEI (Japão): {indicadores.get('NIKKEI', {}).get('v', 0):,.0f} pts ({indicadores.get('NIKKEI', {}).get('d', 0):+.2f}%)
                        - HANG SENG (China): {indicadores.get('HANGSENG', {}).get('v', 0):,.0f} pts ({indicadores.get('HANGSENG', {}).get('d', 0):+.2f}%)
                        - STOXX 600 (Europa): {indicadores.get('STOXX600', {}).get('v', 0):,.0f} pts ({indicadores.get('STOXX600', {}).get('d', 0):+.2f}%)
                        - Petróleo Brent: $ {indicadores.get('BRENT', {}).get('v', 0):.2f} ({indicadores.get('BRENT', {}).get('d', 0):+.2f}%)
                        - Ouro: $ {indicadores.get('GOLD', {}).get('v', 0):.2f} ({indicadores.get('GOLD', {}).get('d', 0):+.2f}%)
                        - Cobre: $ {indicadores.get('COPPER', {}).get('v', 0):.2f} ({indicadores.get('COPPER', {}).get('d', 0):+.2f}%)
                        - Soja: $ {indicadores.get('SOYBEAN', {}).get('v', 0):.2f} ({indicadores.get('SOYBEAN', {}).get('d', 0):+.2f}%)
                        - Milho: $ {indicadores.get('CORN', {}).get('v', 0):.2f} ({indicadores.get('CORN', {}).get('d', 0):+.2f}%)
                        - Açúcar: $ {indicadores.get('SUGAR', {}).get('v', 0):.2f} ({indicadores.get('SUGAR', {}).get('d', 0):+.2f}%)
                        - Café: $ {indicadores.get('COFFEE', {}).get('v', 0):.2f} ({indicadores.get('COFFEE', {}).get('d', 0):+.2f}%)
                        - Bitcoin: $ {indicadores.get('BITCOIN', {}).get('v', 0):,.0f} ({indicadores.get('BITCOIN', {}).get('d', 0):+.2f}%)
                        """
                        
                        genai.configure(api_key=api_key)
                        modelo_boletim = encontrar_modelo_flash()
                        
                        prompt_diario = f"""Inicie EXATAMENTE com:
                        "Olá, Tepeto!
                        
                        **☀️ MORNING CALL TEPETOS FINANCE**
                        **Data:** {datetime.date.today().strftime('%d/%m/%Y')}"

                        Você é o Economista-Chefe do Family Office Tepetos. Escreva o resumo matinal dos mercados no estilo institucional 'Morning Call'.
                        Considere OBRIGATORIAMENTE os dados matemáticos abaixo para escrever o seu texto:
                        
                        {panorama_mundial}

                        Crie um relatório em Markdown elegante, direto e com a estrutura EXATA abaixo:

                        ### ⚡ Fast Track (Resumo de 1 Minuto)
                        (Crie 3 a 4 bullet points curtos com as manchetes e direcionamentos mais importantes do dia para o investidor ler rápido).

                        ### 🌍 Fechamento e Abertura dos Mercados
                        (Use os DADOS EXATOS que eu te passei acima para falar sobre as Bolsas dos EUA e do Brasil (Ibovespa), citando se subiram ou caíram e a porcentagem. Faça o mesmo para a Ásia e a Europa nesta manhã).

                        ### 🛢️ Commodities, Agro e Cripto
                        (Use os DADOS EXATOS passados acima para comentar as commodities (Petróleo, Ouro, Cobre e os produtos do Agronegócio como Soja, Milho, Açúcar e Café). Não precisa citar todos obrigatoriamente se não houver variação relevante, foque nos destaques que impactam a inflação e a bolsa brasileira. Cite sempre o preço de fechamento e a variação %. Comente também rapidamente sobre a variação do Bitcoin).

                        ### 📅 Agenda do Dia
                        (Aponte os principais indicadores econômicos, decisões de juros ou balanços importantes esperados para hoje no Brasil e no mundo).

                        ### 🏛️ Cenário Macro e Político
                        (Análise detalhada da economia e política global e brasileira. Como isso afeta os juros, o dólar e a inflação hoje?).

                        ### 🎯 Radar da sua Carteira
                        Abaixo estão os ativos presentes na carteira do cliente hoje e suas últimas variações:
                        [DADOS DA CARTEIRA]
                        {dados_performance}
                        [FIM DOS DADOS]
                        (Selecione alguns ativos ou setores específicos dessa lista que podem sofrer impactos diretos hoje devido às notícias ou cenário macroeconômico atual. Destaque perspectivas e comentários relevantes sobre eles. Não precisa citar todos, apenas os destaques).

                        IMPORTANTE - ASSINATURA E IDENTIDADE:
                        No final do texto, conclua EXATAMENTE com o seguinte texto:

                        "Na Tepetos Finance, nosso nome vem do ucraniano "tieptio", que significa "pintinho". Assim como um ninho, cuidamos do nosso patrimônio, dos nossos investimentos e do nosso futuro com muito amor, carinho e instinto de proteção, buscando sempre o melhor para o nosso crescimento e segurança.

                        Excelente dia de negócios!
                        Com carinho,
                        Tepetos"
                        """
                        
                        response_bol = genai.GenerativeModel(modelo_boletim).generate_content(prompt_diario)
                        st.success("Morning Call gerado com sucesso!")
                        st.markdown(f"<div style='background-color: white; padding: 30px; border-left: 5px solid #FFD700; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>{response_bol.text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                    except Exception as e: st.error(f"Erro ao gerar o Morning Call: {e}")

        if btn_semanal:
            if not api_key: st.error("Insira a chave API da Google no menu lateral.")
            else:
                with st.spinner("Extraindo a variação da última SEMANA da sua carteira e gerando relatório..."):
                    try:
                        dados_performance = get_portfolio_performance_string(conta_selecionada, "5d")
                        genai.configure(api_key=api_key)
                        modelo_boletim = encontrar_modelo_flash()
                        
                        prompt_semanal = f"""Inicie EXATAMENTE com:
                        "Olá, Tepeto!
                        
                        **BOLETIM SEMANAL DE MERCADO E CARTEIRA**
                        **Período:** Semana atual de {mes_atual} de {ano_atual}"

                        Você é um Economista-Chefe responsável por elaborar este Resumo para o 'Tepetos' Finance'.

                        Gere o restante do relatório com linguagem profissional, contendo:
                        1) Visão Macro da Semana: Dólar a R$ {indicadores['USD']['v']:.2f}, IBOV a {indicadores.get('IBOV', {}).get('v', 0):.0f} pts.
                        2) Mercados e Tendências da semana.
                        
                        3) Desempenho da Carteira na Semana:
                        [DADOS REAIS DA CARTEIRA NA ÚLTIMA SEMANA]
                        {dados_performance}
                        [FIM DOS DADOS]
                        - CRIE TABELAS SEPARADAS em Markdown para CADA Classe de Ativo encontrada nos dados (ex: uma tabela para Ações, outra para FIIs, etc.).
                        - As colunas das tabelas DEVEM SER: | Ativo | Preço Atual | Variação Semanal |.
                        - Comente rapidamente os principais destaques de alta e baixa DESSA SEMANA.
                        
                        4) Ações Práticas recomendadas para a próxima semana.

                        IMPORTANTE - ASSINATURA E IDENTIDADE:
                        No final do texto, conclua EXATAMENTE com o seguinte texto:

                        "Na Tepetos Finance, nosso nome vem do ucraniano "tieptio", que significa "pintinho". Assim como um ninho, cuidamos do nosso patrimônio, dos nossos investimentos e do nosso futuro com muito amor, carinho e instinto de proteção, buscando sempre o melhor para o nosso crescimento e segurança.

                        Com carinho,
                        Tepetos"
                        """
                        
                        response_bol = genai.GenerativeModel(modelo_boletim).generate_content(prompt_semanal)
                        st.success("Boletim Semanal gerado com sucesso!")
                        st.markdown(f"<div style='background-color: white; padding: 30px; border-left: 5px solid #006437; border-radius: 8px;'>{response_bol.text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                    except Exception as e: st.error(f"Erro ao gerar o relatório: {e}")

        if btn_mensal:
            if not api_key: st.error("Insira a chave API da Google no menu lateral.")
            else:
                with st.spinner("Extraindo a variação do último MÊS da sua carteira e gerando o dossiê completo..."):
                    try:
                        dados_performance = get_portfolio_performance_string(conta_selecionada, "1mo")
                        genai.configure(api_key=api_key)
                        modelo_boletim = encontrar_modelo_flash()
                        
                        prompt_mensal = f"""Inicie EXATAMENTE com:
                        "Olá, Tepeto!

                        **RESUMO MENSAL DE MERCADO E CARTEIRA**
                        **Período:** {mes_atual}, {ano_atual}"

                        Você é um Economista-Chefe responsável por elaborar um Resumo Mensal de Mercado e Carteira para o 'Tepetos' Finance'.

                        Gere um relatório estruturado com linguagem profissional, clara e analítica, contendo:

                        1) Panorama Macroeconômico:
                        - Resumo dos principais acontecimentos no Brasil e no mundo no mês.
                        - Política monetária (Fed e Copom).
                        - Inflação relevante (IPCA atual em {indicadores['IPCA']['v']}% e Selic em {indicadores['SELIC']['v']}%).
                        - Comportamento do dólar (Cotação em R$ {indicadores['USD']['v']:.2f}).
                        - Eventos geopolíticos importantes.

                        2) Desempenho dos Mercados:
                        - Principais índices globais (% no mês).
                        - Ibovespa (Atual: {indicadores.get('IBOV', {}).get('v', 0):.0f} pts).
                        - IFIX.
                        - Small caps.
                        - Volatilidade (VIX).

                        3) Commodities:
                        - Minério de ferro.
                        - Petróleo (Brent).
                        - Ouro.
                        - Cobre.

                        4) Criptomoedas:
                        - Bitcoin e Ethereum (Tendência e dominância).

                        5) Desempenho da Carteira (NO MÊS):
                        Utilize os dados REAIS extraídos da variação mensal dos ativos do usuário:
                        [DADOS AUTOMÁTICOS DA CARTEIRA - MÊS ATUAL]
                        {dados_performance}
                        [FIM DOS DADOS]
                        - CRIE TABELAS SEPARADAS em Markdown para CADA Classe de Ativo encontrada nos dados (ex: uma tabela para Ações, outra para FIIs, etc.).
                        - As colunas das tabelas DEVEM SER: | Ativo | Preço Atual | Variação no Mês |.
                        - Comente sobre as maiores altas e baixas NESTE MÊS específico baseando-se nas tabelas geradas.
                        - Faça uma análise interpretativa do desempenho geral da carteira frente ao mercado neste mês.

                        6) Ativos para Monitorar:
                        Identifique riscos e eventos relevantes para os ativos listados na carteira.

                        7) Conclusão Estratégica:
                        Apresente uma leitura estratégica do momento de mercado focando em análise macro e oportunidades.

                        Formato:
                        - Texto corrido, elegante e usando negritos.
                        - Objetivo e direto. Entre 800 e 1200 palavras.

                        IMPORTANTE - ASSINATURA E IDENTIDADE:
                        No final do texto, conclua EXATAMENTE com o seguinte texto:

                        "Na Tepetos Finance, nosso nome vem do ucraniano "tieptio", que significa "pintinho". Assim como um ninho, cuidamos do nosso patrimônio, dos nossos investimentos e do nosso futuro com muito amor, carinho e instinto de proteção, buscando sempre o melhor para o nosso crescimento e segurança.

                        Com carinho,
                        Tepetos"
                        """
                        
                        response_bol = genai.GenerativeModel(modelo_boletim).generate_content(prompt_mensal)
                        st.success("Dossiê Mensal gerado com sucesso!")
                        st.markdown(f"<div style='background-color: white; padding: 30px; border-left: 5px solid #006437; border-radius: 8px;'>{response_bol.text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                    except Exception as e: st.error(f"Erro ao gerar o relatório: {e}")