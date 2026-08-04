import glob
import info_feeding_test.config as config
import info_feeding_test.evaluation as evaluation


def get_config_from_name(name):

    config_dict = {
        "use_one_line": False,
        "use_cycle": False,
        "use_lehmer": False,
        "use_inversion": False
    }


    if "ONE" in name:
        config_dict["use_one_line"] = True

    if "CYCLE" in name:
        config_dict["use_cycle"] = True

    if "LEHMER" in name:
        config_dict["use_lehmer"] = True

    if "INV" in name:
        config_dict["use_inversion"] = True


    return config_dict



def main():

    models = glob.glob(
        "*_model.pt"
    )


    print(
        f"Found {len(models)} models"
    )


    for model_file in models:

        experiment_name = (
            model_file
            .replace("_model.pt", "")
        )


        print("\n====================")
        print(
            "Evaluating:",
            experiment_name
        )


        experiment_config = get_config_from_name(
            experiment_name
        )


        config.ENCODING_CONFIG = experiment_config


        evaluation.main()



if __name__ == "__main__":
    main()