import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from groq import Groq


# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
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

st.markdown(
    """
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

    .disclaimer {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


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
# PERIODOS
# =========================================================

PERIODOS = {

    "1 mes": "1mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "1 año": "1y",
    "2 años": "2y",
    "5 años": "5y"

}


# =========================================================
# OBTENER DATOS HISTÓRICOS
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

        # -------------------------------------------------
        # NORMALIZAR COLUMNAS DE YFINANCE
        # -------------------------------------------------

        if isinstance(datos.columns, pd.MultiIndex):

            # Para un solo ticker tomamos el primer nivel
            try:
                datos.columns = datos.columns.get_level_values(0)
            except Exception:
                datos.columns = [
                    columna[0]
                    if isinstance(columna, tuple)
                    else columna
                    for columna in datos.columns
                ]

        # -------------------------------------------------
        # ASEGURAR COLUMNAS NECESARIAS
        # -------------------------------------------------

        columnas_necesarias = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for columna in columnas_necesarias:

            if columna not in datos.columns:
                return pd.DataFrame()

        # -------------------------------------------------
        # LIMPIAR DATOS
        # -------------------------------------------------

        datos = datos.copy()

        datos = datos.dropna(
            subset=["Close"]
        )

        return datos

    except Exception:

        return pd.DataFrame()


# =========================================================
# OBTENER INFORMACIÓN FUNDAMENTAL
# =========================================================

@st.cache_data(ttl=1800)
def obtener_fundamentales(ticker):

    try:

        activo = yf.Ticker(ticker)

        info = activo.info

        return {

            "Nombre":
                info.get("longName"),

            "Moneda":
                info.get("currency"),

            "Sector":
                info.get("sector"),

            "Industria":
                info.get("industry"),

            "P/E":
                info.get("trailingPE"),

            "Forward P/E":
                info.get("forwardPE"),

            "P/B":
                info.get("priceToBook"),

            "Dividend Yield":
                info.get("dividendYield"),

            "Beta":
                info.get("beta"),

            "ROE":
                info.get("returnOnEquity"),

            "ROA":
                info.get("returnOnAssets"),

            "Capitalización":
                info.get("marketCap")

        }

    except Exception:

        return {}


# =========================================================
# CÁLCULO DE INDICADORES TÉCNICOS
# =========================================================

def calcular_indicadores(datos):

    datos = datos.copy()

    # -----------------------------------------------------
    # RETORNOS
    # -----------------------------------------------------

    datos["Retorno"] = (
        datos["Close"]
        .pct_change()
    )

    # -----------------------------------------------------
    # MEDIAS MÓVILES
    # -----------------------------------------------------

    datos["SMA20"] = (
        datos["Close"]
        .rolling(window=20)
        .mean()
    )

    datos["SMA50"] = (
        datos["Close"]
        .rolling(window=50)
        .mean()
    )

    datos["SMA200"] = (
        datos["Close"]
        .rolling(window=200)
        .mean()
    )

    # -----------------------------------------------------
    # RSI 14
    # -----------------------------------------------------

    delta = datos["Close"].diff()

    ganancias = delta.clip(
        lower=0
    )

    perdidas = -delta.clip(
        upper=0
    )

    media_ganancias = (
        ganancias
        .rolling(window=14)
        .mean()
    )

    media_perdidas = (
        perdidas
        .rolling(window=14)
        .mean()
    )

    # Evitar división por cero
    media_perdidas = media_perdidas.replace(
        0,
        np.nan
    )

    rs = (
        media_ganancias
        /
        media_perdidas
    )

    datos["RSI14"] = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    return datos


# =========================================================
# MÉTRICAS FINANCIERAS
# =========================================================

def calcular_metricas(datos):

    retornos = (
        datos["Close"]
        .pct_change()
        .dropna()
    )

    if len(datos) == 0:

        return {}

    # -----------------------------------------------------
    # PRECIO ACTUAL
    # -----------------------------------------------------

    precio_actual = float(
        datos["Close"].iloc[-1]
    )

    # -----------------------------------------------------
    # RENDIMIENTO DEL PERIODO
    # -----------------------------------------------------

    precio_inicial = float(
        datos["Close"].iloc[0]
    )

    if precio_inicial != 0:

        rendimiento = (
            precio_actual
            /
            precio_inicial
            - 1
        )

    else:

        rendimiento = np.nan

    # -----------------------------------------------------
    # VOLATILIDAD ANUALIZADA
    # -----------------------------------------------------

    if len(retornos) > 1:

        volatilidad = (
            retornos.std()
            *
            np.sqrt(252)
        )

    else:

        volatilidad = np.nan

    # -----------------------------------------------------
    # MAXIMUM DRAWDOWN
    # -----------------------------------------------------

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

    max_drawdown = float(
        drawdown.min()
    )

    # -----------------------------------------------------
    # SHARPE
    # -----------------------------------------------------

    tasa_libre_riesgo = 0.05

    if (
        len(retornos) > 1
        and pd.notna(volatilidad)
        and volatilidad != 0
    ):

        rendimiento_anualizado = (
            retornos.mean()
            *
            252
        )

        sharpe = (
            rendimiento_anualizado
            -
            tasa_libre_riesgo
        ) / volatilidad

    else:

        sharpe = np.nan

    # -----------------------------------------------------
    # INDICADORES TÉCNICOS
    # -----------------------------------------------------

    rsi = datos["RSI14"].iloc[-1]

    sma20 = datos["SMA20"].iloc[-1]

    sma50 = datos["SMA50"].iloc[-1]

    sma200 = datos["SMA200"].iloc[-1]

    return {

        "Precio":
            precio_actual,

        "Rendimiento":
            rendimiento,

        "Volatilidad":
            volatilidad,

        "Drawdown":
            max_drawdown,

        "Sharpe":
            sharpe,

        "RSI":
            rsi,

        "SMA20":
            sma20,

        "SMA50":
            sma50,

        "SMA200":
            sma200

    }


# =========================================================
# FORMATEAR NÚMEROS
# =========================================================

def formato_numero(valor, decimales=2):

    if valor is None:
        return "N/D"

    try:

        if pd.isna(valor):
            return "N/D"

        return f"{float(valor):,.{decimales}f}"

    except Exception:

        return "N/D"


def formato_porcentaje(valor):

    if valor is None:
        return "N/D"

    try:

        if pd.isna(valor):
            return "N/D"

        return f"{float(valor):.2%}"

    except Exception:

        return "N/D"


# =========================================================
# FORMATEAR DIVIDEND YIELD
# =========================================================

def normalizar_dividend_yield(valor):

    if valor is None:
        return None

    try:

        valor = float(valor)

        if pd.isna(valor):
            return None

        # Yahoo puede devolver el dato como:
        # 0.025 = 2.5%
        # o como 2.5 = 2.5%
        if valor > 1:

            valor = valor / 100

        return valor

    except Exception:

        return None


# =========================================================
# INTERPRETACIÓN DEL RSI
# =========================================================

def interpretar_rsi(rsi):

    if rsi is None or pd.isna(rsi):

        return (
            "No hay suficientes observaciones "
            "para calcular el RSI."
        )

    if rsi >= 70:

        return (
            "El RSI se encuentra en zona "
            "tradicionalmente considerada de "
            "sobrecompra."
        )

    elif rsi <= 30:

        return (
            "El RSI se encuentra en zona "
            "tradicionalmente considerada de "
            "sobreventa."
        )

    else:

        return (
            "El RSI se encuentra en una zona "
            "intermedia, sin señal extrema."
        )


# =========================================================
# INTERPRETACIÓN DEL SHARPE
# =========================================================

def interpretar_sharpe(sharpe):

    if sharpe is None or pd.isna(sharpe):

        return (
            "No fue posible calcular un Sharpe "
            "confiable para este periodo."
        )

    if sharpe >= 1:

        return (
            "El Sharpe es positivo y relativamente "
            "favorable en términos de rendimiento "
            "ajustado por riesgo."
        )

    elif sharpe >= 0:

        return (
            "El Sharpe es positivo, aunque el "
            "rendimiento ajustado por riesgo "
            "es moderado."
        )

    else:

        return (
            "El Sharpe es negativo, lo que indica "
            "un rendimiento inferior a la tasa "
            "libre de riesgo bajo esta metodología."
        )


# =========================================================
# IA - GROQ
# =========================================================

def analizar_con_ia(
    ticker,
    metricas,
    fundamentales
):

    if "GROQ_API_KEY" not in st.secrets:

        return (
            "### ℹ️ IA no configurada\n\n"
            "La plataforma está funcionando correctamente "
            "para el análisis cuantitativo, pero la API de "
            "Groq todavía no está configurada.\n\n"
            "Puedes agregar `GROQ_API_KEY` en los Secrets "
            "de Streamlit para habilitar el análisis mediante IA."
        )

    try:

        cliente = Groq(
            api_key=st.secrets[
                "GROQ_API_KEY"
            ]
        )

        prompt = f"""
Eres un analista financiero cuantitativo
especializado en los mercados de México
y Estados Unidos.

Analiza exclusivamente la información
proporcionada.

TICKER:
{ticker}

MÉTRICAS CUANTITATIVAS:
{metricas}

INFORMACIÓN FUNDAMENTAL:
{fundamentales}

Realiza un análisis estructurado con:

1. Resumen ejecutivo
2. Rendimiento observado
3. Riesgo
4. Análisis técnico
5. Análisis fundamental
6. Interpretación del Sharpe
7. Fortalezas
8. Riesgos
9. Conclusión

Reglas:

- No inventes datos.
- No agregues precios que no estén proporcionados.
- Distingue datos observados de cálculos.
- Explica las limitaciones de la información.
- No presentes el análisis como asesoría financiera personalizada.
- No garantices rendimientos futuros.
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

        return (
            respuesta
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return (
            "### ⚠️ No fue posible generar el análisis IA\n\n"
            f"Detalle técnico: `{e}`"
        )


# =========================================================
# ENCABEZADO
# =========================================================

st.title("📊 Finanzas AI")

st.markdown(
    """
### Plataforma de análisis bursátil

**México 🇲🇽 | Estados Unidos 🇺🇸 | ETFs 📊**

Consulta precios históricos, indicadores técnicos,
métricas de riesgo, información fundamental y análisis
mediante inteligencia artificial.
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
    list(
        ACTIVOS[mercado].keys()
    )
)

ticker = ACTIVOS[
    mercado
][empresa]

periodo_seleccionado = st.sidebar.selectbox(
    "Periodo de análisis",
    list(
        PERIODOS.keys()
    )
)

periodo_codigo = PERIODOS[
    periodo_seleccionado
]


analizar = st.sidebar.button(
    "🔍 ANALIZAR ACTIVO",
    use_container_width=True,
    type="primary"
)


# =========================================================
# GUARDAR SELECCIÓN EN SESSION STATE
# =========================================================

if analizar:

    st.session_state[
        "analizar_activo"
    ] = True

    st.session_state[
        "ticker"
    ] = ticker

    st.session_state[
        "empresa"
    ] = empresa

    st.session_state[
        "mercado"
    ] = mercado

    st.session_state[
        "periodo"
    ] = periodo_codigo

    st.session_state[
        "periodo_nombre"
    ] = periodo_seleccionado


# =========================================================
# DETERMINAR ACTIVO A ANALIZAR
# =========================================================

if st.session_state.get(
    "analizar_activo",
    False
):

    ticker_analisis = st.session_state[
        "ticker"
    ]

    empresa_analisis = st.session_state[
        "empresa"
    ]

    mercado_analisis = st.session_state[
        "mercado"
    ]

    periodo_analisis = st.session_state[
        "periodo"
    ]

    periodo_nombre_analisis = st.session_state[
        "periodo_nombre"
    ]

else:

    ticker_analisis = ticker

    empresa_analisis = empresa

    mercado_analisis = mercado

    periodo_analisis = periodo_codigo

    periodo_nombre_analisis = periodo_seleccionado


# =========================================================
# INFORMACIÓN DEL ACTIVO
# =========================================================

st.info(
    f"Activo seleccionado: "
    f"**{empresa_analisis}**  |  "
    f"Ticker: **{ticker_analisis}**  |  "
    f"Periodo: **{periodo_nombre_analisis}**"
)


# =========================================================
# ANÁLISIS
# =========================================================

if st.session_state.get(
    "analizar_activo",
    False
):

    # -----------------------------------------------------
    # OBTENER DATOS
    # -----------------------------------------------------

    with st.spinner(
        "📡 Obteniendo información financiera..."
    ):

        datos = obtener_historial(
            ticker_analisis,
            periodo_analisis
        )

        fundamentales = obtener_fundamentales(
            ticker_analisis
        )


    # -----------------------------------------------------
    # VALIDAR DATOS
    # -----------------------------------------------------

    if datos.empty:

        st.error(
            f"No se encontraron datos para "
            f"{ticker_analisis}."
        )

        st.warning(
            """
            Yahoo Finance no devolvió información
            para este activo.

            Verifica el ticker o intenta nuevamente.
            """
        )

        st.stop()


    # -----------------------------------------------------
    # CALCULAR INDICADORES
    # -----------------------------------------------------

    datos = calcular_indicadores(
        datos
    )

    metricas = calcular_metricas(
        datos
    )


    # =====================================================
    # KPIs
    # =====================================================

    st.subheader(
        "📊 Indicadores principales"
    )

    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Precio",
        formato_numero(
            metricas["Precio"]
        )
    )


    c2.metric(
        "Rendimiento",
        formato_porcentaje(
            metricas["Rendimiento"]
        )
    )


    c3.metric(
        "Volatilidad",
        formato_porcentaje(
            metricas["Volatilidad"]
        )
    )


    c4.metric(
        "Sharpe",
        formato_numero(
            metricas["Sharpe"]
        )
    )


    # =====================================================
    # GRÁFICO DE PRECIO
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
        hovermode="x unified",
        legend_title="Indicadores"
    )


    st.plotly_chart(
        figura,
        use_container_width=True
    )


    # =====================================================
    # RIESGO
    # =====================================================

    st.subheader(
        "⚠️ Riesgo"
    )

    r1, r2, r3 = st.columns(3)


    r1.metric(
        "Volatilidad anualizada",
        formato_porcentaje(
            metricas["Volatilidad"]
        )
    )


    r2.metric(
        "Máximo Drawdown",
        formato_porcentaje(
            metricas["Drawdown"]
        )
    )


    r3.metric(
        "RSI 14",
        formato_numero(
            metricas["RSI"]
        )
    )


    # =====================================================
    # INTERPRETACIÓN TÉCNICA
    # =====================================================

    st.subheader(
        "📉 Interpretación técnica"
    )

    t1, t2 = st.columns(2)


    with t1:

        st.write(
            "**RSI:**"
        )

        st.write(
            interpretar_rsi(
                metricas["RSI"]
            )
        )


    with t2:

        st.write(
            "**Sharpe:**"
        )

        st.write(
            interpretar_sharpe(
                metricas["Sharpe"]
            )
        )


    # =====================================================
    # MEDIAS MÓVILES
    # =====================================================

    st.subheader(
        "📐 Medias móviles"
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
    # FUNDAMENTALES
    # =====================================================

    st.subheader(
        "💰 Indicadores fundamentales"
    )

    f1, f2, f3, f4 = st.columns(4)


    f1.metric(
        "P/E",
        formato_numero(
            fundamentales.get(
                "P/E"
            )
        )
    )


    f2.metric(
        "P/B",
        formato_numero(
            fundamentales.get(
                "P/B"
            )
        )
    )


    f3.metric(
        "Beta",
        formato_numero(
            fundamentales.get(
                "Beta"
            )
        )
    )


    dividend = normalizar_dividend_yield(
        fundamentales.get(
            "Dividend Yield"
        )
    )


    f4.metric(
        "Dividend Yield",
        formato_porcentaje(
            dividend
        )
    )


    # -----------------------------------------------------
    # INFORMACIÓN GENERAL
    # -----------------------------------------------------

    st.subheader(
        "🏢 Información general"
    )


    informacion = pd.DataFrame({

        "Variable": [

            "Nombre",
            "Moneda",
            "Sector",
            "Industria",
            "Forward P/E",
            "ROE",
            "ROA",
            "Capitalización"

        ],

        "Valor": [

            fundamentales.get(
                "Nombre",
                "N/D"
            ),

            fundamentales.get(
                "Moneda",
                "N/D"
            ),

            fundamentales.get(
                "Sector",
                "N/D"
            ),

            fundamentales.get(
                "Industria",
                "N/D"
            ),

            formato_numero(
                fundamentales.get(
                    "Forward P/E"
                )
            ),

            formato_porcentaje(
                fundamentales.get(
                    "ROE"
                )
            ),

            formato_porcentaje(
                fundamentales.get(
                    "ROA"
                )
            ),

            formato_numero(
                fundamentales.get(
                    "Capitalización"
                ),
                0
            )

        ]

    })


    st.dataframe(
        informacion,
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
        "🤖 La IA está interpretando los resultados..."
    ):

        analisis = analizar_con_ia(
            ticker_analisis,
            metricas,
            fundamentales
        )


    st.markdown(
        analisis
    )


    # =====================================================
    # DATOS HISTÓRICOS
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
    derivadas de las fuentes utilizadas.

    Los análisis generados por inteligencia artificial no
    constituyen asesoría financiera, recomendación de inversión
    ni garantía de rendimiento.

    </div>
    """,
    unsafe_allow_html=True
)
