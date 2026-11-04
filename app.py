from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from src.environment import SCENARIOS
from src.metrics import run_scenario, table_rows
from src.plotstyle import plot_curves

st.set_page_config(page_title="adaptive trial lab", layout="wide")
css = (Path(__file__).parent / "assets" / "style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

@st.cache_data(show_spinner="running trials")
def simulate(scenario, horizon, n_trials, seed):
    return run_scenario(scenario, horizon, n_trials, seed)

def table_html(rows):
    head = "<tr><th>algorithm</th><th>final regret</th><th>patients on best arm, %</th></tr>"
    body = "".join(f"<tr><td>{n}</td><td>{r:.1f} ± {rc:.1f}</td><td>{b:.1f} ± {bc:.1f}</td></tr>"
                   for n, r, rc, b, bc in rows)
    return f'<table class="tbl">{head}{body}</table>'


st.markdown(
    '<div class="wordmark">adaptive trial lab</div>'
    '<h1 class="headline">learning which treatment works while patients are already in the trial</h1>'
    '<p class="sub">four ways to allocate patients, compared over many simulated trials.</p>',
    unsafe_allow_html=True)

st.markdown('<hr class="rule"><div class="section">controls</div>', unsafe_allow_html=True)
with st.form("controls", border=False):
    c1, c2, c3, c4 = st.columns(4, gap="large")
    scenario = c1.selectbox("scenario", SCENARIOS)
    horizon = c2.slider("patients per trial", 200, 3000, 1000, step=100)
    n_trials = c3.slider("independent trials", 50, 500, 300, step=50)
    seed = c4.number_input("seed", min_value=0, value=0, step=1)
    st.form_submit_button("run")

results = simulate(scenario, horizon, n_trials, int(seed))
x = np.arange(1, horizon + 1)
change = horizon // 3 if scenario == "drift" else None

st.markdown('<hr class="rule"><div class="section">result</div>', unsafe_allow_html=True)
st.markdown('<div class="caption">cumulative regret, mean with 95% interval</div>', unsafe_allow_html=True)
fig = plot_curves({k: v["regret"] for k, v in results.items()}, x, "patients treated", "regret", vline=change)
st.pyplot(fig)
plt.close(fig)
st.markdown('<div class="caption">share of patients given the best treatment so far, %</div>', unsafe_allow_html=True)
fig = plot_curves({k: v["best"] for k, v in results.items()}, x, "patients treated", "% on best arm", vline=change)
st.pyplot(fig)
plt.close(fig)

st.markdown('<hr class="rule"><div class="section">numbers</div>', unsafe_allow_html=True)
st.markdown(table_html(table_rows(results)), unsafe_allow_html=True)
st.markdown(
    '<hr class="rule"><div class="section">method</div><div class="method">'
    "<p>each patient receives one treatment and is either cured or not, with a fixed hidden probability.</p>"
    "<p>regret is the cure probability lost by not always using the best treatment, summed over patients.</p>"
    "<p>every setting is repeated in independent trials. lines show the mean, bands show mean ± 1.96 standard errors.</p>"
    "<p>in the drift scenario the best treatment loses its effect a third of the way through.</p></div>",
    unsafe_allow_html=True)
