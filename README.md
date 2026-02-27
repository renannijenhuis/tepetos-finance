# 🐥 Tepetos' Finance v22.2 - CORRIGIDO 🐷💚

Sistema ERP de Gestão Financeira Pessoal com IA integrada (Google Gemini)

> **✅ VERSÃO CORRIGIDA** - Este é o código v22.2 COM AS 3 CORREÇÕES APLICADAS para funcionar perfeitamente no Streamlit Cloud com Python 3.13.

---

## 🚀 Funcionalidades

### 💳 Tesouraria & Despesas
- **Orçamento 60/20/20:** Controle suas finanças com metas inteligentes
- **Dashboard Executivo:** Visualize receitas, despesas e investimentos em tempo real
- **Importação Automática:** Leia extratos, PDFs e Excel com IA
- **Auditoria Inteligente:** CFO Virtual analisa seu orçamento

### 💼 Wealth Management
- **Carteira Universal:** Ações BR, Stocks US, FIIs, Renda Fixa, Cripto
- **Terminal Fundamentalista:** Análise completa com Tepetos Score
- **Planejamento FIRE:** Calculadora de independência financeira
- **Deep Dive com IA:** Relatórios completos com gráficos e valuation DCF
- **Morning Call Diário:** Boletins personalizados sobre seus ativos

---

## 📦 Tecnologias

- **Frontend:** Streamlit (interface moderna)
- **IA:** Google Gemini Flash
- **Dados Financeiros:** Yahoo Finance (yfinance)
- **Gráficos:** Plotly Express
- **Banco:** SQLite3 (local, seguro)

---

## ☁️ Deploy no Streamlit Cloud

### Passo 1: Preparar Repositório

1. Crie um repositório no GitHub
2. Faça upload destes arquivos:
   - `meu_app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`

### Passo 2: Deploy

1. Acesse [streamlit.io/cloud](https://streamlit.io/cloud)
2. Conecte seu repositório
3. Configure a API Key nas **Secrets**:
   ```toml
   GOOGLE_API_KEY = "SUA_CHAVE_AQUI"
   ```
4. Clique em **Deploy!**

---

## 🔧 Instalação Local

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/tepetos-finance.git
cd tepetos-finance

# Instale as dependências
pip install -r requirements.txt

# Configure sua API Key
# Crie o arquivo .streamlit/secrets.toml:
echo 'GOOGLE_API_KEY = "sua_chave"' > .streamlit/secrets.toml

# Execute
streamlit run meu_app.py
```

---

## ✅ O que foi corrigido na v22.2

Esta versão inclui **3 correções críticas** aplicadas:

1. **✅ Commodities simplificadas** - Removidos tickers problemáticos (NIKKEI, Soja, Milho, etc.)
2. **✅ Morning Call atualizado** - Sem referências a commodities indisponíveis
3. **✅ Try/except nos gráficos** - Proteção contra falhas nos Deep Dives

---

## 📁 Estrutura do Projeto

```
tepetos-finance/
├── meu_app.py              # Código principal (2752 linhas)
├── requirements.txt        # Dependências
├── .streamlit/
│   └── config.toml        # Tema verde Tepetos
├── financas.db            # Banco (gerado automaticamente)
└── README.md              # Esta documentação
```

---

## 🎯 Roadmap Futuro

- [ ] PostgreSQL (banco na nuvem)
- [ ] Multi-usuário com autenticação
- [ ] Backup Google Drive
- [ ] Integração Open Banking
- [ ] App mobile nativo

---

## 🐥 Sobre o Nome

"**Tepetos**" vem do ucraniano "тепто" (*tieptio*), que significa "**pintinho**". 

Como um ninho, cuidamos do nosso patrimônio com amor e proteção! 💚

---

## 📝 Licença

Uso pessoal livre. Para uso comercial, entre em contato.

---

**Desenvolvido com 💚 pela equipe Tepetos Finance**
