import argparse
import os


def get_name(folder, string, idx):
    return os.path.join(folder, f"{string}.run_{idx}.logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True, type=str)
    parser.add_argument("-s", "--string", required=True, type=str)
    args = parser.parse_args()

    run_id = 1
    if os.path.exists(get_name(args.folder, args.string, run_id)):
        run_id += 1
    print(get_name(args.folder, args.string, run_id))
