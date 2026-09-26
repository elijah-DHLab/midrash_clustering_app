import os
from dotenv import load_dotenv
import pandas as pd
import json
import fasttext
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from clearml import Task

load_dotenv()

config = {
    "task_name": "cluster_optimization",
    "model_path": "G:/.shortcut-targets-by-id/1K7DwC8_x5-sEu7JpzotvjF1nWhGNkGuj/Zohar/fasttext_model.bin",
    "result_dir": "G:/My Drive/Haifa/MIDRASH/Results",
    "data_dir": "G:/My Drive/Haifa/MIDRASH/SefariasData/rabbinic",
    "metadata_path": "Inputs/avg_text_length_per_file.xlsx",
    "dataset_choice": "Experiment 1",
    "apply_smoothing": False,
    "k_min": 2,
    "k_max": 30,
    "k_step": 1,
    # silhouette_sample_size: None = use all points (slow); set e.g. 2000 for large datasets
    "silhouette_sample_size": 2000,
}


class ClusterOptimizer:
    def __init__(
        self,
        task_name,
        model_path,
        result_dir,
        data_dir,
        metadata_path,
        dataset_choice,
        apply_smoothing,
        k_min,
        k_max,
        k_step,
        silhouette_sample_size,
    ):
        self.task = Task.init(project_name="MIDRASH", task_name=task_name)
        self.config = self.task.connect(config)
        print("started clearml task")

        self.apply_smoothing = apply_smoothing
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.result_dir = result_dir
        self.silhouette_sample_size = silhouette_sample_size

        self.options = {
            "Experiment 1": "Experiment 1: Rabbinic Literature",
            "Experiment 2": "Experiment 2: Midrashic Literature",
            "Experiment 3": "Experiment 3: Tanhuma",
            "Experiment 4": "Experiment 4: Tanhuma",
            "Experiment 5": "Experiment 5: Midrashic Literature w/o anthologies",
            "Experiment 6": "Experiment 6: Early Main Literature (halakhic, amoraic, tanhumah)",
        }

        self.dataset_name = self.options[dataset_choice]

        self.metadata = pd.read_excel(self.metadata_path, dtype=str)
        self.dataset = self.metadata[self.metadata[self.dataset_name] == "1"].copy()
        print(f"loaded {len(self.dataset)} documents")

        self.features = self._build_features(model_path)
        print(f"built feature matrix: {self.features.shape}")

        k_range = range(k_min, k_max + 1, k_step)
        inertias, silhouettes = self._sweep(k_range)

        self._report(k_range, inertias, silhouettes)

        self.task.flush(wait_for_uploads=True)
        self.task.close()
        print("done")

    def _build_features(self, model_path):
        self.dataset["hierarchy_list"] = self.dataset["hierarchy"].apply(
            lambda x: tuple(int(part.split("-")[0]) for part in str(x).split("."))
        )
        self.dataset = self.dataset.sort_values(by="hierarchy_list").reset_index(drop=True)

        model = fasttext.load_model(model_path)
        print("loaded fasttext model")

        vecs = []
        for _, row in self.dataset.iterrows():
            filename = row["filename"]
            if filename not in os.listdir(self.data_dir):
                continue
            filepath = os.path.join(self.data_dir, filename)
            full_text = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            full_text.extend(obj.get("sentence", "").split())
                        except json.JSONDecodeError:
                            pass
            chunks = [" ".join(full_text[i: i + 50]) for i in range(0, len(full_text), 50)]
            for chunk in chunks:
                vecs.append(model.get_sentence_vector(chunk))

        XX = np.stack(vecs)

        if self.apply_smoothing:
            weights = np.array([0.2, 0.5, 0.6, 0.7, 0.8, 1, 0.8, 0.7, 0.6, 0.5, 0.2])
            XX = np.array(
                [np.convolve(x, weights, mode="same") / sum(weights) for x in XX.T]
            ).T

        return XX

    def _sweep(self, k_range):
        inertias = []
        silhouettes = []

        for k in k_range:
            print(f"fitting k={k}...")
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(self.features)
            inertias.append(km.inertia_)

            sil = silhouette_score(
                self.features,
                labels,
                sample_size=self.silhouette_sample_size,
                random_state=42,
            )
            silhouettes.append(sil)
            print(f"  inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

        return inertias, silhouettes

    def _report(self, k_range, inertias, silhouettes):
        k_list = list(k_range)

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Elbow Method (Inertia)", "Silhouette Score"),
            vertical_spacing=0.15,
        )

        fig.add_trace(
            go.Scatter(x=k_list, y=inertias, mode="lines+markers", name="Inertia"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=k_list, y=silhouettes, mode="lines+markers", name="Silhouette"),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
        fig.update_xaxes(title_text="Number of Clusters (k)", row=2, col=1)
        fig.update_yaxes(title_text="Inertia", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=2, col=1)

        fig.update_layout(
            title_text=f"Cluster Optimization — {self.dataset_name}",
            height=800,
            width=1000,
            showlegend=False,
        )

        best_k_sil = k_list[int(np.argmax(silhouettes))]
        print(f"\nBest k by silhouette score: {best_k_sil} (score={max(silhouettes):.4f})")

        output_path = os.path.join(self.result_dir, "cluster_optimization.html")
        fig.write_html(output_path, auto_open=False)
        print(f"saved plot to {output_path}")

        self.task.get_logger().report_plotly(
            title="Cluster Optimization",
            series=self.dataset_name,
            iteration=0,
            figure=json.loads(fig.to_json()),
        )

        results_df = pd.DataFrame({"k": k_list, "inertia": inertias, "silhouette": silhouettes})
        self.task.upload_artifact(name="Optimization Results", artifact_object=results_df)

        self.task.get_logger().report_scalar(
            "Best k (silhouette)", "k", best_k_sil, iteration=0
        )
        self.task.get_logger().report_scalar(
            "Best silhouette score", "score", max(silhouettes), iteration=0
        )


if __name__ == "__main__":
    ClusterOptimizer(**config)
