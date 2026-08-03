from itertools import combinations
import config
import train
import evaluation


ENCODINGS = [
    "use_one_line",
    "use_cycle",
    "use_lehmer",
    "use_inversion"
]

def generate_configs():
    configs = []
    for size in range(1,5):
        for combo in combinations(
            ENCODINGS,
            size
        ):
            current = {
                "use_one_line": False,
                "use_cycle": False,
                "use_lehmer": False,
                "use_inversion": False
            }
            for item in combo:
                current[item] = True
            configs.append(current)
    return configs


def main():

    configs = generate_configs()
    print(
        f"Total experiments: {len(configs)}"
    )

    for i, experiment_config in enumerate(configs):
        print("\n====================")
        print(
            f"Experiment {i+1}/{len(configs)}"
        )
        print(experiment_config)
        config.ENCODING_CONFIG = experiment_config
        train.train()
        evaluation.main()


if __name__ == "__main__":
    main()