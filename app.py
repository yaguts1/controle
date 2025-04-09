import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sympy as sp

st.set_page_config(page_title="Simulador Interativo de Controle", layout="wide")
st.title("🎛️ Simulador Interativo de Controle (PD / PI / PD + PI)")

# === Função de transferência da planta ===
st.sidebar.title("🧮 Função de Transferência da Planta")
expr_input = st.sidebar.text_input("Expressão simbólica", value="1 / (s*(s+6)*(s+15))")
usar_expr = st.sidebar.checkbox("Usar expressão simbólica acima", value=True)

if usar_expr and expr_input:
    try:
        s_sym = sp.symbols('s')
        expr = sp.sympify(expr_input, locals={'s': s_sym})
        num_sym, den_sym = sp.fraction(expr)
        num_poly = sp.Poly(num_sym, s_sym)
        den_poly = sp.Poly(den_sym, s_sym)
        num = [float(c) for c in num_poly.all_coeffs()]
        den = [float(c) for c in den_poly.all_coeffs()]
        G = ctrl.TransferFunction(num, den)
        st.sidebar.success("Função convertida com sucesso.")
        st.sidebar.latex(f"G(s) = {sp.latex(expr)}")
    except Exception as e:
        st.sidebar.error(f"Erro ao interpretar a expressão: {e}")
        st.stop()
else:
    num_input = st.sidebar.text_input("Numerador (coeficientes)", value="1")
    den_input = st.sidebar.text_input("Denominador (coeficientes)", value="1, 21, 90, 0")
    try:
        num = [float(x.strip()) for x in num_input.split(",")]
        den = [float(x.strip()) for x in den_input.split(",")]
        G = ctrl.TransferFunction(num, den)
        st.sidebar.success("Função de transferência válida.")
    except Exception as e:
        st.sidebar.error(f"Erro na função de transferência: {e}")
        st.stop()

s = ctrl.TransferFunction.s

# === Requisitos de desempenho ===
st.sidebar.title("🎯 Requisitos de Desempenho")
OS_input = st.sidebar.slider("Ultrapassagem (%)", 1.0, 100.0, step=0.5, value=25.0)
tp_input = st.sidebar.slider("Tempo de Pico (s)", 0.1, 5.0, step=0.1, value=0.6)

OS = OS_input / 100
tp = tp_input
zeta = -np.log(OS) / np.sqrt(np.pi**2 + (np.log(OS))**2)
wn = np.pi / (tp * np.sqrt(1 - zeta**2))
sigma = -zeta * wn
wd = wn * np.sqrt(1 - zeta**2)
polo_desejado = complex(sigma, wd)

st.sidebar.markdown(f"""
**🧮 Polo desejado:**  
ζ = `{zeta:.4f}`  
ωₙ = `{wn:.2f}`  
Polo = `{polo_desejado.real:.2f} ± {polo_desejado.imag:.2f}j`
""")

# === Configuração do compensador ===
st.sidebar.title("⚙️ Compensador")
tipo = st.sidebar.radio("Tipo de Compensador", ["PD", "PI", "PD + PI"])
zero_PD = st.sidebar.slider("Zero do PD", -40.0, -0.1, step=0.1, value=-30.0)
zero_PI = st.sidebar.slider("Zero do PI", -3.0, -0.01, step=0.01, value=-0.1)
polo_PI = st.sidebar.slider("Polo do PI", -0.01, -0.00001, step=0.00001, value=-0.0067)
ganho_manual = st.sidebar.slider("Ganho K", 0.0, 2000.0, step=0.5, value=17.0)
auto_alinhamento = st.sidebar.button("🎯 Alinhar automaticamente")

# === Funções auxiliares ===
def criar_compensador(tipo, zp, zpi, ppi):
    if tipo == "PD":
        return s - zp
    elif tipo == "PI":
        return (s - zpi) / s
    else:
        return ((s - zp) * (s - zpi)) / (s - ppi)

def analisar_resposta(t, y):
    y_final = y[-1]
    overshoot = (np.max(y) - y_final) / y_final * 100
    tp_index = np.argmax(y)
    tp = t[tp_index]
    try:
        t10 = t[np.where(y >= 0.1 * y_final)[0][0]]
        t90 = t[np.where(y >= 0.9 * y_final)[0][0]]
        ts = t90 - t10
    except IndexError:
        ts = np.nan
    erro_ss = abs(1 - y_final)
    return {"overshoot": overshoot, "tempo_pico": tp, "tempo_subida": ts, "erro_ss": erro_ss}

def simular(tipo_local, K, zp, zpi, ppi):
    C = criar_compensador(tipo_local, zp, zpi, ppi)
    Gc = ctrl.series(C, G)
    Gmf = ctrl.feedback(K * Gc, 1)
    t, y = ctrl.step_response(Gmf)
    return t, y, Gmf, C

def encontrar_compensador_otimo(tipo, OS_max=0.25, tp_max=0.6, tolerancia=0.02):
    melhor_erro = float("inf")
    melhor_parametros = {}
    OS_limite = OS_max * (1 + tolerancia)
    tp_limite = tp_max * (1 + tolerancia)

    def eh_valido(t, y):
        analise = analisar_resposta(t, y)
        return analise["overshoot"] / 100 <= OS_limite and analise["tempo_pico"] <= tp_limite

    if tipo == "PD":
        for z in np.linspace(-5, -35, 30):
            C = s - z
            G_C = ctrl.series(C, G)
            rlist, klist = ctrl.root_locus(G_C, plot=False)
            for polos, K in zip(rlist, klist):
                for p in polos:
                    if np.imag(p) < 0.1 or abs(p - polo_desejado) > 5:
                        continue
                    Gmf = ctrl.feedback(K * G_C, 1)
                    t, y = ctrl.step_response(Gmf)
                    if eh_valido(t, y):
                        erro = abs(p - polo_desejado)
                        if erro < melhor_erro:
                            melhor_parametros = {"zero_PD": z, "ganho": K, "erro": erro}
                            melhor_erro = erro

    elif tipo == "PI":
        for z in np.linspace(-0.01, -3, 30):
            C = (s - z) / s
            G_C = ctrl.series(C, G)
            rlist, klist = ctrl.root_locus(G_C, plot=False)
            for polos, K in zip(rlist, klist):
                for p in polos:
                    if np.imag(p) < 0.1 or abs(p - polo_desejado) > 5:
                        continue
                    Gmf = ctrl.feedback(K * G_C, 1)
                    t, y = ctrl.step_response(Gmf)
                    if eh_valido(t, y):
                        erro = abs(p - polo_desejado)
                        if erro < melhor_erro:
                            melhor_parametros = {"zero_PI": z, "ganho": K, "erro": erro}
                            melhor_erro = erro

    elif tipo == "PD + PI":
        for zp in np.linspace(-5, -30, 20):
            for zpi in np.linspace(-0.05, -3, 20):
                C = criar_compensador("PD + PI", zp, zpi, polo_PI)
                G_C = ctrl.series(C, G)
                rlist, klist = ctrl.root_locus(G_C, plot=False)
                for polos, K in zip(rlist, klist):
                    for p in polos:
                        if np.imag(p) < 0.1 or abs(p - polo_desejado) > 5:
                            continue
                        Gmf = ctrl.feedback(K * G_C, 1)
                        t, y = ctrl.step_response(Gmf)
                        if eh_valido(t, y):
                            erro = abs(p - polo_desejado)
                            if erro < melhor_erro:
                                melhor_parametros = {
                                    "zero_PD": zp,
                                    "zero_PI": zpi,
                                    "polo_PI": polo_PI,
                                    "ganho": K,
                                    "erro": erro
                                }
                                melhor_erro = erro
    return melhor_parametros if melhor_parametros else None

# === Alinhamento automático ===
if auto_alinhamento:
    melhores = encontrar_compensador_otimo(tipo, OS_max=OS, tp_max=tp)
    if melhores:
        zero_PD = melhores.get("zero_PD", zero_PD)
        zero_PI = melhores.get("zero_PI", zero_PI)
        polo_PI = melhores.get("polo_PI", polo_PI)
        ganho = melhores.get("ganho", ganho_manual)
        erro_polo = melhores.get("erro", None)
        st.sidebar.success("Parâmetros ajustados com sucesso!")
        st.sidebar.markdown(
            f"📌 Zero PD = {zero_PD:.3f}\n"
            f"📌 Zero PI = {zero_PI:.4f}\n"
            f"📌 Polo PI = {polo_PI:.5f}\n"
            f"📌 K = {ganho:.2f}\n"
            f"📏 Erro até polo desejado = {erro_polo:.4f}"
        )
    else:
        st.sidebar.error("⚠️ Nenhuma solução válida encontrada.")
        ganho = ganho_manual
else:
    ganho = ganho_manual

# === Simulação ===
t, y, Gmf, C = simular(tipo, ganho, zero_PD, zero_PI, polo_PI)
analise = analisar_resposta(t, y)
polos = Gmf.poles()
G_C = ctrl.series(C, G)

# === Gráficos ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Lugar das Raízes (LGR)")
    fig1, ax1 = plt.subplots()
    ctrl.rlocus(G_C, ax=ax1)
    ax1.plot(polo_desejado.real, polo_desejado.imag, 'rx', label="Polo desejado", markersize=10)
    for i, p in enumerate(polos):
        ax1.plot(p.real, p.imag, 'go')
        ax1.text(p.real + 0.5, p.imag, f"P{i+1}", color="green")
    ax1.set_xlim(min(-50, polo_desejado.real - 5), max(5, polo_desejado.real + 10))
    ax1.set_ylim(-abs(polo_desejado.imag) - 5, abs(polo_desejado.imag) + 5)
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader("📈 Resposta ao Degrau")
    fig2, ax2 = plt.subplots()
    ax2.plot(t, y)
    ax2.set_title(
        f"{tipo} — K={ganho:.1f} | OS={analise['overshoot']:.1f}% | "
        f"Tp={analise['tempo_pico']:.2f}s | Ts={analise['tempo_subida']:.2f}s\n"
        f"Erro estacionário = {analise['erro_ss']:.4f}"
    )
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Saída")
    ax2.grid(True)
    ax2.set_ylim(0, 1.3)
    fig2.tight_layout()
    st.pyplot(fig2)

# === Comparação ===
st.subheader("📊 Comparação entre Compensadores")
fig3, ax3 = plt.subplots()
for tipo_i in ["PD", "PI", "PD + PI"]:
    t_i, y_i, _, _ = simular(tipo_i, ganho, zero_PD, zero_PI, polo_PI)
    ax3.plot(t_i, y_i, label=tipo_i)
ax3.set_xlabel("Tempo (s)")
ax3.set_ylabel("Saída")
ax3.set_title("Comparação entre Compensadores")
ax3.grid(True)
ax3.legend()
fig3.tight_layout()
st.pyplot(fig3)

st.markdown("---")
st.caption("Desenvolvido com ❤️ usando Streamlit, Sympy, Control e Matplotlib.")
