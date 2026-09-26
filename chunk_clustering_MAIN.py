import os
from dotenv import load_dotenv
import pandas as pd
import json
import fasttext
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
from clearml import Task

load_dotenv()

path = "G:/My Drive/Haifa/MIDRASH"

config = {
    "task_name": "CPD_optimised_clustering_num",
    "model_path": "G:/.shortcut-targets-by-id/1K7DwC8_x5-sEu7JpzotvjF1nWhGNkGuj/Zohar/fasttext_model.bin",
    "result_dir": "G:/My Drive/Haifa/MIDRASH/Results",
    "data_dir": "G:/My Drive/Haifa/MIDRASH/SefariasData/rabbinic",
    "metadata_path": "Inputs/avg_text_length_per_file.xlsx",
    "dataset_choice": "Experiment 1",
    "num_clusters": 10,
    "apply_smoothing": False,
    "table_output_name": "rabani_unsmoothed_table",
    "visualization_output_name": "rabani_unsmoothed_visualization",
}


class chunks_clusters:
    def __init__(
        self,
        task_name,
        model_path,
        result_dir,
        data_dir,
        metadata_path,
        dataset_choice,
        num_clusters,
        apply_smoothing,
        table_output_name,
        visualization_output_name,
    ):

        self.task = Task.init(project_name="MIDRASH", task_name=task_name)
        self.config = self.task.connect(config)
        print("started clearml task")

        self.apply_smoothing = apply_smoothing
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.options = {
            "Experiment 1": "Experiment 1: Rabbinic Literature",
            "Experiment 2": "Experiment 2: Midrashic Literature",
            "Experiment 3": "Experiment 3: Tanhuma",
            "Experiment 4": "Experiment 4: Tanhuma",
            "Experiment 5": "Experiment 5: Midrashic Literature w/o anthologies",
            "Experiment 6": "Experiment 6: Early Main Literature (halakhic, amoraic, tanhumah)",
        }

        self.dataset_name = self.options[dataset_choice]
        self.table_output = os.path.join(result_dir, f"{table_output_name}.csv")
        self.visualization_output = os.path.join(
            result_dir, f"{visualization_output_name}.html"
        )

        print("created all relevant paths")

        self.metadata = self.read_metadata()
        print("read metadata")
        self.dataset = self.metadata[self.metadata[self.dataset_name] == "1"].copy()
        print("created relevant dataset")
        self.df_chunks = self.create_df_chunks()

        self.model = fasttext.load_model(model_path)
        print("loaded model")
        self.df_chunks["vec"] = self.df_chunks["chunk"].apply(
            lambda x: self.model.get_sentence_vector(x)
        )
        print("created vectors")
        print(self.df_chunks.head())

        self.df_chunks["scluster"] = self.perform_clustering(num_clusters)
        print("created clusters")
        print(self.df_chunks.head())

        self.save_table()
        print("saved table")
        self.save_ngram_frequencies()
        print("saved ngram frequencies")
        self.create_and_save_plot()
        print("created and saved plot")
        self.task.flush(wait_for_uploads=True)
        self.task.close()
        print("closed clearml task")

    def read_metadata(self):
        return pd.read_excel(self.metadata_path, dtype=str)

    def create_df_chunks(self):

        df_chunk_records = []

        self.dataset["hierarchy_list"] = self.dataset["hierarchy"].apply(
            lambda x: tuple(int(part.split("-")[0]) for part in str(x).split("."))
        )

        self.dataset = self.dataset.sort_values(by="hierarchy_list").reset_index(
            drop=True
        )

        for _, row in self.dataset.iterrows():
            filename = row["filename"]
            hierarchy = row["hierarchy"]
            hierarchy_list = row["hierarchy_list"]

            full_text = []

            if filename in os.listdir(self.data_dir):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                obj = json.loads(line)
                                sentence = obj.get("sentence", "")
                                full_text.extend(sentence.split())
                            except json.JSONDecodeError as e:
                                print(f"Error in file {filename}: {e}")

                chunks = [
                    " ".join(full_text[i : i + 50])
                    for i in range(0, len(full_text), 50)
                ]

                for idx, chunk in enumerate(chunks):
                    df_chunk_records.append(
                        {
                            "filename": filename,
                            "hierarchy": hierarchy,
                            "hierarchy_list": hierarchy_list,
                            "chunk": chunk,
                            "chunk_number": idx + 1,
                        }
                    )

        df_chunks = pd.DataFrame(df_chunk_records)
        df_chunks = df_chunks.sort_values(
            by=["hierarchy_list", "filename", "chunk_number"]
        ).reset_index(drop=True)
        df_chunks["chronological_index"] = df_chunks.index

        return df_chunks

    def perform_clustering(self, num_clusters):
        XX = np.stack(self.df_chunks["vec"].values)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        if self.apply_smoothing:
            weights = np.array([0.2, 0.5, 0.6, 0.7, 0.8, 1, 0.8, 0.7, 0.6, 0.5, 0.2])
            features = np.array(
                [np.convolve(x, weights, mode="same") / sum(weights) for x in XX.T]
            ).T
        else:
            features = XX
        return kmeans.fit_predict(features)

    def save_ngram_frequencies(self):
        from collections import Counter

        ngrams_dir = os.path.join(os.path.dirname(self.table_output), "ngrams")
        os.makedirs(ngrams_dir, exist_ok=True)

        for cluster_id in sorted(self.df_chunks["scluster"].unique()):
            cluster_chunks = self.df_chunks[self.df_chunks["scluster"] == cluster_id]["chunk"]

            unigrams = Counter()
            bigrams = Counter()
            trigrams = Counter()

            for chunk in cluster_chunks:
                tokens = str(chunk).split()
                unigrams.update(tokens)
                bigrams.update(zip(tokens, tokens[1:]))
                trigrams.update(zip(tokens, tokens[1:], tokens[2:]))

            output_path = os.path.join(ngrams_dir, f"cluster_{cluster_id}_ngrams.xlsx")
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                pd.DataFrame(unigrams.most_common(), columns=["מילה", "תדירות"]).to_excel(
                    writer, sheet_name="מילים", index=False
                )
                pd.DataFrame(
                    [(" ".join(k), v) for k, v in bigrams.most_common()],
                    columns=["זוג", "תדירות"],
                ).to_excel(writer, sheet_name="זוגות", index=False)
                pd.DataFrame(
                    [(" ".join(k), v) for k, v in trigrams.most_common()],
                    columns=["שלשה", "תדירות"],
                ).to_excel(writer, sheet_name="שלשות", index=False)

            self.task.upload_artifact(
                name=f"NGrams Cluster {cluster_id}", artifact_object=output_path
            )
            print(f"saved ngrams for cluster {cluster_id} → {output_path}")

    def save_table(self):
        self.df_chunks["vec"] = self.df_chunks["vec"].apply(
            lambda x: x.tolist()
        )  # Convert to list for JSON serialization
        self.df_chunks["vec"] = self.df_chunks["vec"].astype(str)

        self.df_chunks.to_csv(self.table_output, index=False)

        self.task.get_logger().report_table(
            title="Chunks Dataframe",
            series=self.dataset_name,
            iteration=0,
            table_plot=self.df_chunks.head(7),
        )
        print("reported clearml table")

        self.task.upload_artifact(name="Cluster Table", artifact_object=self.df_chunks)

    def create_and_save_plot(self):
        self.df_chunks["chunk_index"] = (
            self.df_chunks.groupby("filename").cumcount() + 1
        )
        self.df_chunks["text_start"] = self.df_chunks["chunk"].apply(
            lambda x: str(x)[:50]
        )
        self.df_chunks["collection"] = self.df_chunks["filename"].apply(
            lambda x: str(x)[: x.find("-.")]
        )
        self.df_chunks["chronological_index"] = self.df_chunks.index

        # Calculate 2D histogram
        hist, xedges, yedges = np.histogram2d(
            self.df_chunks["chronological_index"],
            self.df_chunks["scluster"],
            bins=[50, 50],
        )

        # Find bin indices for each point
        x_bin_indices = np.digitize(self.df_chunks["chronological_index"], xedges) - 1
        y_bin_indices = np.digitize(self.df_chunks["scluster"], yedges) - 1

        # Ensure indices are within valid range
        x_bin_indices[x_bin_indices >= hist.shape[0]] = hist.shape[0] - 1
        y_bin_indices[y_bin_indices >= hist.shape[1]] = hist.shape[1] - 1

        # Get density for each point
        density = hist[x_bin_indices, y_bin_indices]

        self.df_chunks["density"] = density

        # Normalize density for consistent marker size
        self.df_chunks["density_normalized"] = (
            self.df_chunks["density"] - self.df_chunks["density"].min()
        ) / (self.df_chunks["density"].max() - self.df_chunks["density"].min())
        self.df_chunks["marker_size"] = (
            self.df_chunks["density_normalized"] * 1
        )  # Adjust the range as needed

        # Plot with uniform marker color and size reflecting density
        fig = px.scatter(
            self.df_chunks,
            x="chronological_index",
            y="scluster",
            color="scluster",
            size="marker_size",
            custom_data=["collection", "chunk_index", "text_start"],
            title="Scatter Plot of Clusters over Time with Density",
            labels={"scluster": "Cluster ID", "file_index": "file index"},
        )

        fig.update_layout(
            width=1200,  # Adjust the width as needed
            height=600,  # Adjust the height to make it longer on the y-axis
        )

        fig.update_traces(
            marker=dict(opacity=0.8, line=dict(width=0)),
            hovertemplate="<b>chunk_index:</b> %{x}<br>"
            + "<b>Cluster:</b> %{y}<br>"
            + "<b>Collection:</b> %{customdata[0]}<br>"
            + "<b>Text:</b> %{customdata[2]}<br>"
            + "<b>Density:</b> %{marker.size}<br>",
        )

        fig.write_html(self.visualization_output, auto_open=False)

        self.task.upload_artifact(
            name="Cluster Visualization HTML", artifact_object=self.visualization_output
        )

        fig_plot = px.scatter(
            self.df_chunks,
            x="chronological_index",
            y="scluster",
            color="scluster",
            custom_data=["collection", "chunk_index"],
            title="Scatter Plot of Clusters over Time",
            labels={"scluster": "Cluster ID"},
        )

        fig_plot.update_layout(width=1200, height=600)

        fig_plot.update_traces(
            marker=dict(size=8, opacity=0.8, line=dict(width=0)),  # גודל אחיד
            hovertemplate="<b>chunk_index:</b> %{customdata[1]}<br>"
            + "<b>Cluster:</b> %{y}<br>"
            + "<b>Collection:</b> %{customdata[0]}<br>",
        )

        self.task.get_logger().report_plotly(
            title="Cluster Analysis with Density",
            series=self.dataset_name,
            iteration=0,
            figure=json.loads(fig.to_json()),
        )

        self.task.get_logger().report_plotly(
            title="Cluster Analysis",
            series=self.dataset_name,
            iteration=0,
            figure=json.loads(fig_plot.to_json()),
        )

        print("reported clearml plot")
        self.task.get_logger().flush()


if __name__ == "__main__":
    chunks_clusters(**config)
