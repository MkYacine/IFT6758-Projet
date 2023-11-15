from src.experiments.utils import *
from src.features.feature_engineering_2 import select_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import src.features.feature_selection as fsel


def main():
    data_2017021065 = pd.read_csv("data\datasets\csv_files\\2017021065.csv")

    exp = create_experiment("wpg_v_wsh_2017021065")
    exp.log_dataframe_profile(
        data_2017021065,
        name="wpg_v_wsh_2017021065",  # keep this name
        dataframe_format="csv",  # ensure you set this flag!
    )
    exp.end()


if __name__ == "__main__":
    main()
