import optix as ox
import cupy as cp
import numpy as np
import argparse
import logging
import sys
import imageio
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Denoise images using the builtin Optix denoiser")

    parser.add_argument('color_file', help="The input color image to denoise (exr format)")
    parser.add_argument('-n', '--normal', type=str, required=False,
                        help="Additional normals image (exr format)")
    parser.add_argument('-a', '--albedo', type=str, required=False,
                        help="Additional albedo image (exr format)")
    parser.add_argument('-f', '--flow', type=str, required=False,
                        help=f"Additional flow image for temporal denoising")
    parser.add_argument('-o', '--out', type=str, default="denoised.exr", required=False,
                        help="Output file. Defaults to \"denoised.exr\"")
    parser.add_argument('-F', '--Frames', type=int, nargs=2, required=False,
                        help="First-Last frame number of a sequence")
    parser.add_argument('-e', '--exposure', type=float, required=False,
                        help="apply exposure on output images")
    parser.add_argument('-t', '--tilesize', type=int, nargs=2, required=False,
                        help="Use tiling to save GPU memory")
    parser.add_argument('-z', action='store_true', required=False,
                        help="Apply flow to input images (no denoising) and write output")
    parser.add_argument('-k', action='store_true', required=False,
                        help="Use kernel prediction model even if there are no AOVs")

    args = parser.parse_args()

    # a color image is always required
    color_image = imageio.read(args.color_file).get_data(0)
    normal_image = None
    albedo_image = None
    flow_image = None
    if args.normal is not None:
        normal_image = imageio.read(args.normal).get_data(0)

    if args.albedo is not None:
        albedo_image = imageio.read(args.albedo).get_data(0)

    # setup optix context and denoiser

    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=True, log_callback_function=logger, log_callback_level=3)

    model_kind = ox.DenoiserModelKind.HDR

    denoiser = ox.Denoiser(ctx, model_kind=model_kind, guide_albedo=args.albedo is not None, guide_normals=args.normal is not None, kp_mode=args.k)

    ret = denoiser.invoke(color_image, albedo=albedo_image if args.albedo else None, normals=normal_image if args.normal else None)
    ret = cp.asnumpy(ret)

    fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)
    axs[0].imshow(np.clip(color_image, 0, 255).astype(np.uint8))
    axs[0].set_title("original")

    ret = np.clip(ret, 0, 255).astype(np.uint8)
    axs[1].imshow(ret)
    axs[1].set_title("denoised")
    plt.show()