import pandas as pd
import os
import streamlit as st
import altair as alt
import re
from io import StringIO

dossier = r"D:\OneToMasterThemAll\Speciation"


st.set_page_config(layout="wide")
alt.data_transformers.disable_max_rows()

def extraire_numero(nom_fichier):
    match = re.search(r'(\d+)', nom_fichier)
    return int(match.group(1)) if match else float("inf")

def charger_sorties(dossier):
    if not os.path.exists(dossier):
        return []

    fichiers = sorted(
        [f for f in os.listdir(dossier) if f.endswith(".txt")],
        key=extraire_numero
    )

    dfs = []

    for i, f in enumerate(fichiers, start=1):
        path = os.path.join(dossier, f)

        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                comment="%",
                header=0
            )
        except Exception as e:
            st.warning(f"Erreur lecture {f} : {e}")
            continue
        if df.empty:
            continue
        df["t"] = i
        dfs.append(df)

    return dfs

dfs = charger_sorties(dossier)

st.title("Breakthrough concentration")

if not dfs:
    st.error("No files have been loaded. Are u sure of the folder path?")
    st.stop()

cols = [c for c in dfs[0].columns if c not in ["t", "x"]]

cols_to_plot = st.multiselect(
    "Select your species :",
    cols,
    default=cols[:1]
)

if not cols_to_plot:
    st.warning("Select at least one species.")
    st.stop()

data = []
for i, df in enumerate(dfs):
    last = df.iloc[-1]

    for col in cols_to_plot:
        data.append({
            "fichier": i,
            "valeur": last[col],
            "espece": col
        })

plot_df = pd.DataFrame(data)
chart = alt.Chart(plot_df).mark_line(point=True).encode(
    x=alt.X("fichier:Q", title="Time-step"),
    y=alt.Y("valeur:Q", title="Concentration (mol/kgw)"),

    color=alt.Color(
        "espece:N",
        title="Species",
        scale=alt.Scale(scheme="category20")
    ),

    tooltip=["espece", "fichier", "valeur"]
).interactive()

st.altair_chart(chart, use_container_width=True)
csv = StringIO()
plot_df.to_csv(csv, index=False)

st.download_button(
    "Download data in .csv format",
    data=csv.getvalue(),
    file_name="breakthroughProfile.csv",
    mime="text/csv"
)