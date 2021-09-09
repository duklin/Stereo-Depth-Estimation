import os
import glob
import shutil
import zipfile
from argparse import ArgumentParser


def preprocess(root: str, kitti_archive: str):
    """Create more verbose and filtered subfolders dedicated for this project
    `image_2` and `image_3` are renamed into `image_left` and `image_right` respectively 
    in both `training` and `testing` folders. Only needed images are copied, the ones
    of the form xxxxxx_10.png. The following structure is created:
    - <root>/ (in this case the `dataset` folder)
        - testing/
            - image_left/
                - xxxxxx_10.png
            - image_right/
                - xxxxxx_10.png
        - training/
            - image_left/
                - xxxxxx_10.png
            - image_right/
                - xxxxxx_10.png
            - disp_occ_left/
                - xxxxxx_10.png
    """

    with zipfile.ZipFile(kitti_archive, "r") as zip_file:
        zip_file.extractall(root)

    original_dirs = [
        "testing/image_2",
        "testing/image_3",
        "training/image_2",
        "training/image_3",
        "training/disp_occ_0",
    ]
    dest_dirs = [
        "testing/image_left",
        "testing/image_right",
        "training/image_left",
        "training/image_right",
        "training/disp_occ_left",
    ]

    for orig_dir, dest_dir in zip(original_dirs, dest_dirs):
        orig_dir = os.path.join(root, orig_dir)
        dest_dir = os.path.join(root, dest_dir)

        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)

        for img_path in glob.glob(f"{orig_dir}/*10.png"):
            shutil.copy(img_path, dest_dir)


def main(args):
    preprocess(args.root, args.kitti_archive)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--kitti-archive")
    args = parser.parse_args()
    main(args)
