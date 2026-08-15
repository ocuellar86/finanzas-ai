import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from groq import Groq


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Finanzas AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# ESTILO
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    font-weight: 700;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.disclaimer {
    padding: 15px;
    border-radius: 10px;
    background-color: #fff7ed;
    border: 1px solid #fed7aa;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# CATÁLOGO DE ACTIVOS
# =========================================================

ACTIVOS = {

    "🇲🇽 México": {

        "Walmart de México": "WALMEX.MX",
        "FEMSA": "FEMSAUBD.MX",
        "Grupo Bimbo": "BIMBOA.MX",
        "Cemex": "CEMEXCPO.MX",
        "América Móvil": "AMXL.MX",
        "Grupo Aeroportuario del Pacífico": "GAPB.MX",
        "Grupo Aeroportuario del Sureste": "ASURB.MX",
        "Grupo Financiero Banorte": "GFNORTEO.MX",
        "Alfa": "ALFAA.MX",
        "Kimberly-Clark de México": "KIMBERA.MX",
    },

    "🇺🇸 Estados Unidos": {

        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "NVIDIA": "NVDA",
        "Amazon": "AMZN",
        "Alphabet": "GOOGL",
        "Meta": "META",
        "Tesla": "TSLA",
        "Coca-Cola": "KO",
        "McDonald's": "MCD",
        "JPMorgan": "JPM",
    },

    "📊 ETFs": {

        "VOO": "VOO",
        "SPY": "SPY",
        "QQQ": "QQQ",
        "IVV": "IVV",
        "VTI": "VTI",
        "SCHD": "SCHD",
        "IWM": "IWM",
    }
}


# =========================================================
# FUNCIONES FINANCIERAS
# =========================================================

@st.cache_data(ttl=900)
def obtener_historial(ticker, periodo):

    try:

        datos = yf.download(
            ticker,
            period=periodo,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if datos is None:
            return pd.DataFrame()

        if datos.empty:
            return pd.DataFrame()

        # En algunas versiones de yfinance
        # las columnas pueden venir como MultiIndex.
        if isinstance(datos.columns, pd.MultiIndex):

            try:
                datos.columns = datos.columns.get_level_values(0)
            except Exception:
                pass

        return datos

    except Exception as e:

        st.error(
            f"Error al obtener datos de Yahoo Finance: {e}"
        )

        return pd.DataFrame()


@st.cache_data(ttl=900)
def obtener_fundamentales(ticker):

    activo = yf.Ticker(ticker)

    try:
        info = activo.info

        return {
            "Nombre": info.get("longName"),
            "Moneda": info.get("currency"),
            "Sector": info.get("sector"),
            "Industria": info.get("industry"),
            "P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "P/B": info.get("priceToBook"),
            "Dividend Yield": info.get("dividendYield"),
            "Beta": info.get("beta"),
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "Capitalización": info.get("marketCap"),
        }

    except Exception:

        return {}


def calcular_indicadores(datos):

    datos = datos.copy()

    datos["Retorno"] = datos["Close"].pct_change()

    datos["SMA20"] = (
        datos["Close"]
        .rolling(20)
        .mean()
    )

    datos["SMA50"] = (
        datos["Close"]
        .rolling(50)
        .mean()
    )

    datos["SMA200"] = (
        datos["Close"]
        .rolling(200)
        .mean()
    )

    # RSI

    delta = datos["Close"].diff()

    ganancias = delta.clip(lower=0)
    perdidas = -delta.clip(upper=0)

    media_ganancias = ganancias.rolling(14).mean()
    media_perdidas = perdidas.rolling(14).mean()

    rs = media_ganancias / media_perdidas

    datos["RSI14"] = (
        100 - (100 / (1 + rs))
    )

    return datos


def calcular_metricas(datos):

    retornos = (
        datos["Close"]
        .pct_change()
        .dropna()
    )

    precio_actual = datos["Close"].iloc[-1]

    rendimiento = (
        datos["Close"].iloc[-1]
        /
        datos["Close"].iloc[0]
        - 1
    )

    volatilidad = (
        retornos.std()
        * np.sqrt(252)
    )

    maximo_acumulado = (
        datos["Close"]
        .cummax()
    )

    drawdown = (
        datos["Close"]
        /
        maximo_acumulado
        - 1
    )

    max_drawdown = drawdown.min()

    rendimiento_anual = (
        retornos.mean()
        * 252
    )

    tasa_libre_riesgo = 0.05

    sharpe = (
        rendimiento_anual
        -
        tasa_libre_riesgo
    ) / volatilidad

    return {
        "Precio": precio_actual,
        "Rendimiento": rendimiento,
        "Volatilidad": volatilidad,
        "Drawdown": max_drawdown,
        "Sharpe": sharpe,
        "RSI": datos["RSI14"].iloc[-1],
        "SMA20": datos["SMA20"].iloc[-1],
        "SMA50": datos["SMA50"].iloc[-1],
        "SMA200": datos["SMA200"].iloc[-1]
    }


# =========================================================
# IA
# =========================================================

def analizar_con_ia(ticker, metricas, fundamentales):

    if "GROQ_API_KEY" not in st.secrets:

        return (
            "La API de Groq todavía no está configurada. "
            "Puedes utilizar el análisis cuantitativo sin IA."
        )

    try:

        cliente = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

        prompt = f"""
        Eres un analista financiero especializado
        en mercados de México y Estados Unidos.

        Analiza el activo:

        TICKER:
        {ticker}

        MÉTRICAS CALCULADAS:

        {metricas}

        INFORMACIÓN FUNDAMENTAL:

        {fundamentales}

        Genera un análisis profesional utilizando
        únicamente la información proporcionada.

        Estructura:

        1. Resumen ejecutivo
        2. Rendimiento
        3. Riesgo
        4. Análisis técnico
        5. Análisis fundamental
        6. Interpretación del Sharpe
        7. Fortalezas
        8. Riesgos
        9. Conclusión

        No inventes datos.

        Diferencia claramente entre:
        - datos observados
        - cálculos
        - interpretación

        No presentes la respuesta como asesoría
        financiera personalizada.
        """

        respuesta = cliente.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content":
                    "Analista financiero cuantitativo."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )

        return respuesta.choices[0].message.content

    except Exception as e:

        return (
            f"No fue posible generar el análisis IA: {e}"
        )


# =========================================================
# ENCABEZADO
# =========================================================

st.title("📊 Finanzas AI")

st.write("🔧 PRUEBA DE CONEXIÓN A YAHOO FINANCE")

try:
    prueba = yf.download(
        "AAPL",
        period="5d",
        interval="1d",
        progress=False,
        threads=False
    )

    st.write("Resultado de Yahoo Finance:")
    st.write(prueba)

except Exception as e:
    st.error("ERROR REAL:")
    st.exception(e)

st.markdown(
    """
### Plataforma de análisis bursátil
**México 🇲🇽 | Estados Unidos 🇺🇸 | ETFs 📊**
"""
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Configuración")

mercado = st.sidebar.selectbox(
    "Mercado",
    list(ACTIVOS.keys())
)

empresa = st.sidebar.selectbox(
    "Activo",
    list(ACTIVOS[mercado].keys())
)

ticker = ACTIVOS[mercado][empresa]

periodo = st.sidebar.selectbox(
    "Periodo de análisis",
    {
        "1 mes": "1mo",
        "3 meses": "3mo",
        "6 meses": "6mo",
        "1 año": "1y",
        "2 años": "2y",
        "5 años": "5y"
    }
)

periodo_codigo = periodo

analizar = st.sidebar.button(
    "🔍 ANALIZAR ACTIVO",
    use_container_width=True,
    type="primary"
)


# =========================================================
# INFORMACIÓN DEL ACTIVO
# =========================================================

st.info(
    f"Activo seleccionado: **{empresa}**  |  "
    f"Ticker: **{ticker}**"
)


# =========================================================
# ANÁLISIS
# =========================================================

if analizar:

    with st.spinner(
        "Obteniendo información financiera..."
    ):

        datos = obtener_historial(
            ticker,
            periodo_codigo
        )

        fundamentales = obtener_fundamentales(
            ticker
        )

if datos.empty:

    st.error(
        f"No se encontraron datos para {ticker}."
    )

    st.warning(
        """
        Yahoo Finance no devolvió información para este ticker.
        Verifica el ticker o intenta nuevamente.
        """
    )

    st.stop()

    datos = calcular_indicadores(datos)

    metricas = calcular_metricas(datos)


    # =====================================================
    # KPIs
    # =====================================================

    st.subheader("📊 Indicadores principales")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Precio",
        f"{metricas['Precio']:,.2f}"
    )

    c2.metric(
        "Rendimiento",
        f"{metricas['Rendimiento']:.2%}"
    )

    c3.metric(
        "Volatilidad",
        f"{metricas['Volatilidad']:.2%}"
    )

    c4.metric(
        "Sharpe",
        f"{metricas['Sharpe']:.2f}"
    )


    # =====================================================
    # GRÁFICO
    # =====================================================

    st.subheader(
        "📈 Evolución del precio"
    )

    figura = go.Figure()

    figura.add_trace(
        go.Scatter(
            x=datos.index,
            y=datos["Close"],
            name="Precio",
            mode="lines"
        )
    )

    figura.add_trace(
        go.Scatter(
            x=datos.index,
            y=datos["SMA20"],
            name="SMA 20",
            mode="lines"
        )
    )

    figura.add_trace(
        go.Scatter(
            x=datos.index,
            y=datos["SMA50"],
            name="SMA 50",
            mode="lines"
        )
    )

    figura.add_trace(
        go.Scatter(
            x=datos.index,
            y=datos["SMA200"],
            name="SMA 200",
            mode="lines"
        )
    )

    figura.update_layout(
        height=500,
        xaxis_title="Fecha",
        yaxis_title="Precio",
        hovermode="x unified"
    )

    st.plotly_chart(
        figura,
        use_container_width=True
    )


    # =====================================================
    # RIESGO
    # =====================================================

    st.subheader("⚠️ Riesgo")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Volatilidad anualizada",
        f"{metricas['Volatilidad']:.2%}"
    )

    c2.metric(
        "Máximo Drawdown",
        f"{metricas['Drawdown']:.2%}"
    )

    c3.metric(
        "RSI",
        f"{metricas['RSI']:.2f}"
    )


    # =====================================================
    # FUNDAMENTALES
    # =====================================================

    st.subheader(
        "💰 Indicadores fundamentales"
    )

    f1, f2, f3, f4 = st.columns(4)

    f1.metric(
        "P/E",
        str(fundamentales.get("P/E", "N/D"))
    )

    f2.metric(
        "P/B",
        str(fundamentales.get("P/B", "N/D"))
    )

    f3.metric(
        "Beta",
        str(fundamentales.get("Beta", "N/D"))
    )

    dividend = fundamentales.get(
        "Dividend Yield"
    )

    if dividend is not None:
        dividend = dividend / 100

    f4.metric(
        "Dividend Yield",
        f"{dividend:.2%}"
        if dividend is not None
        else "N/D"
    )


    # =====================================================
    # TÉCNICO
    # =====================================================

    st.subheader(
        "📉 Análisis técnico"
    )

    tecnico = pd.DataFrame({

        "Indicador": [
            "RSI 14",
            "SMA 20",
            "SMA 50",
            "SMA 200"
        ],

        "Valor": [
            metricas["RSI"],
            metricas["SMA20"],
            metricas["SMA50"],
            metricas["SMA200"]
        ]

    })

    st.dataframe(
        tecnico,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # IA
    # =====================================================

    st.subheader(
        "🤖 Análisis mediante Inteligencia Artificial"
    )

    with st.spinner(
        "La IA está interpretando los resultados..."
    ):

        analisis = analizar_con_ia(
            ticker,
            metricas,
            fundamentales
        )

    st.markdown(
        analisis
    )


    # =====================================================
    # DATOS
    # =====================================================

    with st.expander(
        "📋 Ver datos históricos"
    ):

        st.dataframe(
            datos.tail(50),
            use_container_width=True
        )


# =========================================================
# AVISO
# =========================================================

st.divider()

st.markdown(
    """
<div class="disclaimer">

<strong>⚠️ Aviso:</strong><br>
Esta plataforma tiene fines educativos e informativos.
Los datos pueden presentar retrasos, errores o limitaciones
derivadas de las fuentes utilizadas. Los análisis generados
por inteligencia artificial no constituyen asesoría financiera,
recomendación de inversión ni garantía de rendimiento.

</div>
""",
    unsafe_allow_html=True
)
