"""Get all pretrained checkpoints from lagrangebench."""

import shutil
import os
import argparse

import gdown


IDs = {
    "gns_tgv2d": "19TO4PaFGcryXOFFKs93IniuPZKEcaJ37",
    "segnn_tgv2d": "1llGtakiDmLfarxk6MUAtqj6sLleMQ7RL",
    "gns_rpf2d": "1uYusVlP1ykUNuw58vo7Wss-xyTMmopAn",
    "segnn_rpf2d": "108dZVWs2qxAvKiboeEBW-nIcv-aslhYP",
    "gns_ldc2d": "1JvdsW0H6XrgC2_cwV3pP66cAm9j1-AXc",
    "segnn_ldc2d": "1D_wgs2pD9pTXoJK76yi-R0K2tY_T6lPn",
    "gns_dam2d": "16bJz3VfSMxOG1II8kCg5DlzGhjvdip2p",
    "segnn_dam2d": "1_6rHxK81vzrdIMPtJ7rIkeoUgsTeKmSn",
    "gns_tgv3d": "1DEkXxrebS9eyLSMlc_ztHrqlh29NgLXC",
    "segnn_tgv3d": "1ivJnHTgfbQ0IJujc5O0CUoQNiGU4zi_d",
    "gns_rpf3d": "1yo-qgShLd1sgS1u5zkMXdJvhuPBwEQQE",
    "segnn_rpf3d": "1Qczh3Z_z0grTuRuPDHyiYLzV1zg7Liz9",
    "gns_ldc3d": "1b3IIkxk5VcWiT8Oyqg1wex8-ZfJv2g_v",
    "segnn_ldc3d": "1ZIg7FXc1l3C4ekc9WvVvjHEl5KKxOA_U",
}


def download_checkpoint(checkpoint_name, checkpoint_id, output_dir):
    """Download a checkpoint from lagrangebench."""

    # download from gdrive
    zip_path = os.path.join(output_dir, checkpoint_name + ".zip")
    gdown.download(id=checkpoint_id, output=zip_path, quiet=False)

    # unzip
    ckp_dir = os.path.join(output_dir, checkpoint_name)
    shutil.unpack_archive(filename=zip_path, extract_dir=ckp_dir)

    # remove the zip file
    os.remove(zip_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="ckp/pretrained")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for checkpoint_name, checkpoint_id in IDs.items():
        download_checkpoint(checkpoint_name, checkpoint_id, args.output_dir)

    print("All checkpoints downloaded.")
