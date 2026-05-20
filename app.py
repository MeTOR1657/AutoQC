import os
import shutil
import warnings
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pandas as pd
import requests
import torch

from scipy.stats import entropy
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

DATASET_FOLDER = "datasets"
OUTPUT_FOLDER = "outputs"
TEMP_FOLDER = "temp"

SAMPLE_RATE = 8000

MAX_WORKERS = os.cpu_count()

# =========================================================
# CREATE FOLDERS
# =========================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    TEMP_FOLDER,
    exist_ok=True
)

# =========================================================
# LOAD SILERO VAD
# =========================================================

print("\nLOADING SILERO VAD...")

vad_model, utils = torch.hub.load(
    'snakers4/silero-vad',
    'silero_vad',
    trust_repo=True,
    force_reload=False
)

(
    get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks
) = utils

print("VAD LOADED")

# =========================================================
# AUDIO FUNCTIONS
# =========================================================

def normalize_audio(y):

    max_val = np.max(
        np.abs(y)
    )

    if max_val > 0:

        y = y / max_val

    return y


def calculate_rms(y):

    return float(
        np.mean(
            np.abs(y)
        )
    )


def calculate_clipping_ratio(y):

    return float(

        np.mean(
            np.abs(y) > 0.99
        )
    )


def calculate_zcr(y):

    zcr = librosa.feature.zero_crossing_rate(
        y
    )

    return float(
        np.mean(zcr)
    )


def calculate_spectral_centroid(
    y,
    sr
):

    centroid = (
        librosa.feature
        .spectral_centroid(
            y=y,
            sr=sr
        )
    )

    return float(
        np.mean(centroid)
    )


def calculate_entropy(y):

    hist, _ = np.histogram(
        y,
        bins=256,
        density=True
    )

    hist = hist + 1e-10

    return float(
        entropy(hist)
    )


def calculate_dynamic_range(y):

    peak = np.max(
        np.abs(y)
    )

    rms = np.sqrt(
        np.mean(y ** 2)
    )

    if rms == 0:
        return 0

    return float(
        20 * np.log10(
            peak / rms
        )
    )


# =========================================================
# VAD
# =========================================================

def run_vad(audio_path):

    try:

        wav = read_audio(
            audio_path,
            sampling_rate=SAMPLE_RATE
        )

        speech_timestamps = (

            get_speech_timestamps(
                wav,
                vad_model,
                sampling_rate=SAMPLE_RATE
            )
        )

        total_speech = 0

        for ts in speech_timestamps:

            total_speech += (

                ts["end"] -
                ts["start"]
            )

        total_duration = len(wav)

        if total_duration == 0:

            return 0, 1

        speech_ratio = (
            total_speech /
            total_duration
        )

        silence_ratio = (
            1 - speech_ratio
        )

        return (
            speech_ratio,
            silence_ratio
        )

    except:

        return 0.5, 0.5


# =========================================================
# DOWNLOAD AUDIO
# =========================================================

def download_audio(url):

    try:

        filename = os.path.basename(
            url.split("?")[0]
        )

        temp_audio_path = (
            os.path.join(
                TEMP_FOLDER,
                filename
            )
        )

        if not os.path.exists(
            temp_audio_path
        ):

            r = requests.get(
                url,
                timeout=30
            )

            with open(
                temp_audio_path,
                "wb"
            ) as f:

                f.write(r.content)

        return temp_audio_path

    except:

        return None


# =========================================================
# PROCESS AUDIO
# =========================================================

def process_audio(audio_path):

    try:

        y, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        duration = (
            librosa.get_duration(
                y=y,
                sr=sr
            )
        )

        y = normalize_audio(y)

        # BASIC
        rms_volume = (
            calculate_rms(y)
        )

        clipping_ratio = (
            calculate_clipping_ratio(y)
        )

        # EXTRA
        zcr = calculate_zcr(y)

        spectral_centroid = (
            calculate_spectral_centroid(
                y,
                sr
            )
        )

        entropy_value = (
            calculate_entropy(y)
        )

        dynamic_range = (
            calculate_dynamic_range(y)
        )

        # VAD
        speech_ratio, silence_ratio = (
            run_vad(audio_path)
        )

        # FILE SIZE
        file_size_kb = (

            os.path.getsize(
                audio_path
            ) / 1024
        )

        return {

            "audio":
                audio_path,

            "duration":
                duration,

            "rms_volume":
                rms_volume,

            "clipping_ratio":
                clipping_ratio,

            "speech_ratio":
                speech_ratio,

            "silence_ratio":
                silence_ratio,

            "file_size_kb":
                file_size_kb,

            "zcr":
                zcr,

            "spectral_centroid":
                spectral_centroid,

            "entropy":
                entropy_value,

            "dynamic_range":
                dynamic_range
        }

    except Exception as e:

        print(
            f"ERROR: {audio_path}"
        )

        print(e)

        return None


# =========================================================
# PROCESS ROW
# =========================================================

def process_row(args):

    row, audio_col, user_col = args

    try:

        audio_path = str(
            row[audio_col]
        )

        # USER
        user = "unknown"

        if user_col is not None:

            user = str(
                row[user_col]
            )

        # URL AUDIO
        if audio_path.startswith(
            "http"
        ):

            audio_path = (
                download_audio(
                    audio_path
                )
            )

            if audio_path is None:

                return None

        # LOCAL AUDIO
        if not os.path.exists(
            audio_path
        ):

            return None

        result = process_audio(
            audio_path
        )

        if result is not None:

            result["user"] = user

        return result

    except:

        return None


# =========================================================
# FIND CSV FILES
# =========================================================

csv_files = glob(
    f"{DATASET_FOLDER}/*.csv"
)

print(
    f"\nFOUND {len(csv_files)} CSV FILES"
)

# =========================================================
# PROCESS CSV
# =========================================================

for csv_file in csv_files:

    print(
        f"\nPROCESSING: {csv_file}"
    )

    try:

        df = pd.read_csv(csv_file)

        # =================================================
        # AUDIO COLUMN
        # =================================================

        audio_col = None

        for c in df.columns:

            if "audio" in c.lower():

                audio_col = c
                break

        if audio_col is None:

            print(
                "NO AUDIO COLUMN"
            )

            continue

        # =================================================
        # USER COLUMN
        # =================================================

        user_col = None

        for c in df.columns:

            if c.lower() in [

                "user",
                "speaker",
                "name",
                "userid"

            ]:

                user_col = c
                break

        # =================================================
        # ROWS
        # =================================================

        rows = [

            (
                row,
                audio_col,
                user_col
            )

            for _, row
            in df.iterrows()
        ]

        # =================================================
        # MULTIPROCESS
        # =================================================

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            results = list(

                tqdm(

                    executor.map(
                        process_row,
                        rows
                    ),

                    total=len(rows)
                )
            )

        # =================================================
        # CLEAN RESULTS
        # =================================================

        all_features = [

            r for r in results
            if r is not None
        ]

        if len(all_features) == 0:

            print("NO FEATURES")

            continue

        # =================================================
        # DATAFRAME
        # =================================================

        feature_df = pd.DataFrame(
            all_features
        )

        # =================================================
        # FEATURES
        # =================================================

        feature_columns = [

            "duration",

            "rms_volume",

            "clipping_ratio",

            "speech_ratio",

            "silence_ratio",

            "file_size_kb",

            "zcr",

            "spectral_centroid",

            "entropy",

            "dynamic_range"
        ]

        X = feature_df[
            feature_columns
        ].fillna(0)

        # =================================================
        # NORMALIZE
        # =================================================

        scaler = StandardScaler()

        X_scaled = (
            scaler.fit_transform(X)
        )

        # =================================================
        # KMEANS
        # =================================================

        kmeans = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )

        feature_df["cluster"] = (

            kmeans.fit_predict(
                X_scaled
            )
        )

        # =================================================
        # DISTANCE TO CENTER
        # =================================================

        distances = cdist(
            X_scaled,
            kmeans.cluster_centers_
        )

        feature_df[
            "distance_to_center"
        ] = [

            distances[i][cluster]

            for i, cluster in enumerate(
                feature_df["cluster"]
            )
        ]

        # =================================================
        # OUTLIER SCORE
        # =================================================

        feature_df[
            "outlier_score"
        ] = (

            feature_df[
                "distance_to_center"
            ]

            /

            feature_df[
                "distance_to_center"
            ].max()
        )

        # =================================================
        # Z SCORE
        # =================================================

        for col in feature_columns:

            mean = (
                feature_df[col]
                .mean()
            )

            std = (
                feature_df[col]
                .std()
            )

            if std == 0:
                std = 1

            feature_df[
                f"{col}_zscore"
            ] = (

                feature_df[col] -
                mean

            ) / std

        # =================================================
        # PERCENTILES
        # =================================================

        for col in feature_columns:

            feature_df[
                f"{col}_percentile"
            ] = (

                feature_df[col]
                .rank(pct=True)
            )

        # =================================================
        # CLUSTER SCORE
        # =================================================

        cluster_scores = {}

        for cluster_id in (
            feature_df["cluster"]
            .unique()
        ):

            cluster_df = feature_df[

                feature_df["cluster"]
                == cluster_id
            ]

            score = (

                cluster_df[
                    "speech_ratio"
                ].mean() * 0.30 +

                cluster_df[
                    "rms_volume"
                ].mean() * 0.15 +

                cluster_df[
                    "dynamic_range"
                ].mean() * 0.10 +

                cluster_df[
                    "duration"
                ].mean() * 0.05 -

                cluster_df[
                    "silence_ratio"
                ].mean() * 0.15 -

                cluster_df[
                    "clipping_ratio"
                ].mean() * 0.10 -

                cluster_df[
                    "entropy"
                ].mean() * 0.10 -

                cluster_df[
                    "distance_to_center"
                ].mean() * 0.05
            )

            cluster_scores[
                cluster_id
            ] = score

        # =================================================
        # BEST / WORST CLUSTER
        # =================================================

        best_cluster = max(
            cluster_scores,
            key=cluster_scores.get
        )

        worst_cluster = min(
            cluster_scores,
            key=cluster_scores.get
        )

        # =================================================
        # MAP QC
        # =================================================

        def map_qc(cluster):

            if cluster == best_cluster:

                return "good"

            elif cluster == worst_cluster:

                return "bad"

            else:

                return "review"

        feature_df["qc"] = (

            feature_df["cluster"]
            .apply(map_qc)
        )

        # =================================================
        # OUTPUT
        # =================================================

        dataset_name = (

            os.path.splitext(

                os.path.basename(
                    csv_file
                )

            )[0]
        )

        dataset_output = os.path.join(
            OUTPUT_FOLDER,
            dataset_name
        )

        os.makedirs(
            dataset_output,
            exist_ok=True
        )

        # =================================================
        # CREATE QC FOLDERS
        # =================================================

        for qc_type in [

            "good",
            "bad",
            "review"
        ]:

            os.makedirs(

                f"{dataset_output}/{qc_type}",

                exist_ok=True
            )

        # =================================================
        # COPY AUDIO
        # =================================================

        for _, row in (
            feature_df.iterrows()
        ):

            audio_path = row["audio"]

            qc = row["qc"]

            user = str(
                row["user"]
            )

            filename = os.path.basename(
                audio_path
            )

            user_folder = os.path.join(

                dataset_output,
                qc,
                user
            )

            os.makedirs(
                user_folder,
                exist_ok=True
            )

            target = os.path.join(
                user_folder,
                filename
            )

            try:

                shutil.copy(
                    audio_path,
                    target
                )

            except:
                pass

        # =================================================
        # EXPORT CSV
        # =================================================

        for qc_type in [

            "good",
            "bad",
            "review"
        ]:

            feature_df[

                feature_df["qc"]
                == qc_type

            ].to_csv(

                f"{dataset_output}/{qc_type}.csv",

                index=False
            )

        # =================================================
        # FULL REPORT CSV
        # =================================================

        feature_df.to_csv(

            f"{dataset_output}/full_report.csv",

            index=False
        )

        # =================================================
        # SUMMARY
        # =================================================

        print("\nQC RESULT")

        print(
            feature_df["qc"]
            .value_counts()
        )

        print(
            f"\nOUTPUT: {dataset_output}"
        )

    except Exception as e:

        print(e)

print("\nDONE")