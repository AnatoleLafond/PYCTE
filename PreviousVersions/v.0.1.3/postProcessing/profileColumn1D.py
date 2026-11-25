import pandas as pd
import os
import re
import streamlit as st
import altair as alt
from io import StringIO

dossier = "/media/al274877/EMTEC C350/calDoloPflotran/Speciation" # insert path leading to Speciation/primarySpecies/couplingHistory folder


st.set_page_config(layout="wide")
alt.data_transformers.disable_max_rows()

def extraire_numero(nom_fichier):
    match = re.search(r'(\d+)', nom_fichier)
    return int(match.group(1)) if match else float("inf")

@st.cache_data(show_spinner="reading output file...")
def charger_sorties(dossier):
    if not os.path.exists(dossier):
        st.error("Folder not found")
        return []

    fichiers_txt = sorted(
        [f for f in os.listdir(dossier) if f.endswith(".txt")],
        key=extraire_numero
    )

    dfs = []

    for i, nom_fichier in enumerate(fichiers_txt):
        chemin = os.path.join(dossier, nom_fichier)

        df = pd.read_csv(
            chemin,
            sep=r"\s+",
            comment="%",
            header=0
        )

        if df.empty:
            continue

        df.insert(0, "t", i)
        dfs.append(df)

    return dfs


df_output = charger_sorties(dossier)

if not df_output:
    st.stop()

st.title("Spatial profile")

index_df = st.slider(
    "Select time step",
    0,
    len(df_output) - 1,
    0
)

df_selected = df_output[index_df]

all_columns = [col for col in df_selected.columns if col not in ['t', 'x']]

cols_to_plot = st.multiselect(
    "Select species :",
    all_columns,
    default=all_columns[:1]
)

if not cols_to_plot:
    st.warning("Pls select at least one species.")
    st.stop()

plot_data = []

use_x = "x" in df_selected.columns

for col in cols_to_plot:
    for i in range(len(df_selected)):
        plot_data.append({
            "x": df_selected["x"].iloc[i] if use_x else i,
            "value": df_selected[col].iloc[i],
            "column": col
        })

plot_df = pd.DataFrame(plot_data)

log_scale = st.toggle("Log. scale", value=False)
if log_scale:  # filtrer uniquement si log activé
    plot_df = plot_df[plot_df["value"] > 0]
scale_type = "log" if log_scale else "linear"

chart = alt.Chart(plot_df).mark_line().encode(
    x=alt.X("x", title="x (m)"),
    y=alt.Y("value", title="Concentration (mol/kgw)", scale=alt.Scale(type=scale_type)),
    color="column",
    tooltip=["column", "x", "value"]
).interactive()

st.altair_chart(chart, use_container_width=True)

csv_buffer = StringIO()
plot_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="Download data",
    data=csv_buffer.getvalue(),
    file_name="picctsOutput.csv",
    mime="text/csv"
)