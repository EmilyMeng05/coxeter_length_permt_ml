ENCODING_CONFIG = {
    "use_one_line": True,
    "use_cycle": True,
    "use_lehmer": True,
    "use_inversion": True
}



def create_experiment_name(config):
    names = []
    if config["use_one_line"]:
        names.append("ONE")
    if config["use_cycle"]:
        names.append("CYCLE")
    if config["use_lehmer"]:
        names.append("LEHMER")
    if config["use_inversion"]:
        names.append("INV")

    return (
        f"E{len(names)}_"
        + "_".join(names)
    )