import pandas as pd
import plotly.express as px
from jinja2 import Template

record_df = pd.read_csv("./results-umap-hdbscan/record.csv")


# FUNCTIONS ====================================================================
def export(fig, save_path: str):
    input_template_path = "./results-umap-hdbscan-figures/template.html"
    saving_kwargs = {
        "full_html": False,
        "auto_play": False,
        "include_plotlyjs": False,
        "include_mathjax": False,
        "config": {"responsive": True},
    }

    with open(save_path, "w", encoding="utf-8") as output_file:
        with open(input_template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({"fig": fig.to_html(**saving_kwargs)}))


# CREATE FIGURES FOR UMAP ======================================================
configurations_to_load = []
for (language, model, n_neighbors), sub_df in record_df.groupby(
    ["language", "model", "n_neighbors"]
):
    configurations_to_load += [
        {
            "language": language,
            "model": model,
            "n_neighbors": n_neighbors,
            "index_to_load": sub_df.index[0],
        }
    ]
configurations_to_load = pd.DataFrame(configurations_to_load)
configurations_to_load = configurations_to_load.sort_values("n_neighbors")

for (language, model), sub_config in configurations_to_load.groupby(
    ["language", "model"]
):
    print(language, model)
    df_to_plot = None
    for i in range(len(sub_config)):
        index_to_load = sub_config.iloc[i]["index_to_load"]
        n_neighbors = sub_config.iloc[i]["n_neighbors"]
        df_temp_loop = pd.read_csv(
            f"./results-umap-hdbscan/results-{index_to_load}.csv"
        ).sample(frac=0.5)
        df_temp_loop.loc[:, "n_neighbors"] = [n_neighbors] * len(df_temp_loop)

        if df_to_plot is None:
            df_to_plot = df_temp_loop
        else:
            df_to_plot = pd.concat((df_to_plot, df_temp_loop))

    fig = px.scatter(
        df_to_plot, x="x", y="y", color="oai_first_name", animation_frame="n_neighbors"
    ).update_layout(
        showlegend=False, height=500, width=750, title={"text": f"{model} ({language})"}
    )

    export(fig, f"./results-umap-hdbscan-figures/figure-umap-{language}-{model}.html")

del fig, df_to_plot, df_temp_loop, language, model, sub_config, n_neighbors, sub_df