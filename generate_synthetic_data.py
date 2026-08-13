import numpy as np
import pandas as pd

def generate_synthetic_committee_data(n_patients=300, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n_patients)

    def noisy_vote(label, flip_prob):
        flip = rng.random(len(label)) < flip_prob
        return np.where(flip, 1 - label, label)

    vote_normal = noisy_vote(label, flip_prob=0.12)
    vote_coop = noisy_vote(label, flip_prob=0.11)

    gender = rng.choice(["Male", "Female"], size=n_patients)
    ethnicity = rng.choice(["Group A", "Group B"], size=n_patients, p=[0.7, 0.3])
    education = rng.choice(["High School", "Associate/Bachelor", "Graduate"], size=n_patients)
    diagnosis = rng.choice(["Diagnosis A", "Diagnosis B", "Diagnosis C"], size=n_patients)
    age = rng.integers(18, 80, n_patients)
    adi_score = rng.uniform(0, 100, n_patients)

    df = pd.DataFrame({
        "label": label,
        "vote_Normal": vote_normal,
        "vote_Normal Co-Op": vote_coop,
        "gender": gender,
        "ethnicity": ethnicity,
        "education": education,
        "diagnosis": diagnosis,
        "age": age,
        "adi_score": adi_score,
    })

    df["age_group"] = pd.cut(
        df["age"], bins=[0, 18, 35, 50, 65, 100],
        labels=["<18", "18-34", "35-49", "50-64", "65+"]
    )
    df["adi_group"] = pd.qcut(
        df["adi_score"], q=4,
        labels=["Q1 (least deprived)", "Q2", "Q3", "Q4 (most deprived)"]
    )

    return df

if __name__ == "__main__":
    df = generate_synthetic_committee_data()
    print(df.head())